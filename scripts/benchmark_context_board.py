"""Synthetic timing replay. Without --port, prepare only. Never uploads firmware."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from replay_context_board import parse_result, write_command, receive, send_block, compare
from replay_context_board import fixtures, host_results, python_check
from check_probability_parity import canonical_feature_function
from verify_features import parse_c_header
from parity_criteria import ROOT, CONTRACT_SHA256, feature_check
from qualify_numerical_parity import local_output

CLOCK_HZ = 64000000
IDENTITY = 'TIMING_REPLAY_V1 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR14=0'
FROZEN_CANDIDATE_SHA256 = '1cfd4ed28b5aad114d1064a1a7323673b8725e434700e7b6303b5009ace74d0b'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_session(connection, inputs, expected, repetitions, evidence):
    write_command(connection, 'H\n')
    if receive(connection) != IDENTITY:
        raise RuntimeError('Unexpected firmware identity; no sample or reset commands sent')
    evidence.update(blocks=[], calibration=[], memory=[])
    write_command(connection, 'M\n')
    evidence['memory'].append({'stage': 'before', **parse_memory(receive(connection))})
    for _ in range(8):
        write_command(connection, 'C\n')
        words = receive(connection).split()
        if len(words) != 3 or words[0] != 'CAL':
            raise ValueError('Malformed calibration response')
        cal = parse_timed('TIMED ' + ' '.join(words[1:]) + ' BUFFER 250', 250)
        evidence['calibration'].append({key: value for key, value in cal.items() if key != 'decision'})
    for repetition in range(repetitions):
        for name, samples in inputs.items():
            write_command(connection, 'R\n')
            if receive(connection) != 'RESET 0':
                raise RuntimeError('Reset not acknowledged')
            # Neither an ATTR14 request nor an incomplete block may change state.
            for command in ('A\n', 'B 00000000\n'):
                write_command(connection, command)
                if receive(connection) != 'ERR invalid_command':
                    raise RuntimeError('Timing image accepted an unsupported command')
            for offset in range(0, len(samples), 250):
                total = offset + 250
                row = parse_timed(send_block(connection, samples[offset:total]), total)
                row.update(case=name, repetition=repetition, total_samples=total)
                evidence['blocks'].append(row)
                if row['decision'] is not None:
                    reference = expected[name][(total - 750) // 250]
                    checked = compare(row['decision'], reference)
                    features = feature_check(reference['values'][1:], row['decision']['values'][1:])
                    row.update(functional_check=checked, feature_check=features,
                               feature_absolute_errors=features['absolute_errors'])
                    if not checked['pass'] or not features['pass']:
                        raise RuntimeError('Timed decision differs from the frozen host pipeline')
            write_command(connection, 'M\n')
            evidence['memory'].append({'stage': 'after_case', 'case': name, 'repetition': repetition,
                                       **parse_memory(receive(connection))})


def parse_memory(line):
    words = line.split()
    if len(words) != 6 or words[0] != 'MEMORY':
        raise ValueError('Malformed memory response')
    reserved, peak, arena, used, free = map(int, words[1:])
    if not (0 < reserved <= 262144 and -1 <= peak <= reserved and
            0 <= used <= arena <= 262144 and 0 <= free <= arena):
        raise ValueError('Inconsistent memory response')
    return {'main_stack_reserved_bytes': reserved, 'main_stack_peak_used_bytes': None if peak == -1 else peak,
            'heap_arena_snapshot_bytes': arena, 'heap_used_snapshot_bytes': used,
            'heap_free_in_arena_snapshot_bytes': free}


def parse_timed(line, total_samples):
    words = line.split()
    if len(words) < 5 or words[0] != 'TIMED' or total_samples < 250 or total_samples % 250:
        raise ValueError('Malformed timing response')
    cycles, elapsed_ms = int(words[1]), int(words[2])
    if not 0 < cycles < 2**32 or not 0 <= elapsed_ms < 60000:
        raise ValueError('Invalid or potentially wrapped timing interval')
    decision = None
    if total_samples < 750:
        if len(words) != 5 or words[3] != 'BUFFER' or int(words[4]) != total_samples:
            raise ValueError('Unexpected warmup response')
    else:
        if words[3] != 'RESULT':
            raise ValueError('Missing decision')
        decision = parse_result(words[4:], hexadecimal=True)
        if decision['meta'] != [(total_samples - 750) // 250, total_samples,
                                total_samples - 750, total_samples - 250, 250]:
            raise ValueError('Stale or inconsistent decision metadata')
    return {'cycles': cycles, 'elapsed_ms': elapsed_ms,
            'microseconds': cycles * 1000000. / CLOCK_HZ, 'decision': decision}


def verify_build(receipt_path, upload_path=None):
    candidate_path = ROOT / 'outputs/parity/qualification/frozen-candidate-v1.json'
    if sha256(candidate_path) != FROZEN_CANDIDATE_SHA256:
        raise ValueError('Frozen candidate changed')
    candidate = json.loads(candidate_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt['candidate_sha256'] != FROZEN_CANDIDATE_SHA256 or candidate['contract_sha256'] != CONTRACT_SHA256:
        raise ValueError('Frozen numerical contract changed')
    # A build receipt must never replace a frozen expected digest, including
    # when the two manifests refer to the same absolute source path.
    for manifest in (candidate['source_sha256'], receipt['source_sha256']):
        for path, digest in manifest.items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f'Build source changed: {path}')
    if sha256(ROOT / receipt['image_path']) != receipt['image_sha256']:
        raise ValueError('Archived timing image changed')
    if upload_path:
        upload = json.loads(upload_path.read_text())
        if (not upload['uploaded_and_verified'] or upload['image_sha256'] != receipt['image_sha256'] or
                upload['candidate_sha256'] != FROZEN_CANDIDATE_SHA256):
            raise ValueError('Upload receipt does not identify the archived timing image')
    return receipt


def timing_summary(blocks):
    groups = {
        'warmup_blocks': [r for r in blocks if r['decision'] is None],
        'first_decision_after_reset': [r for r in blocks if r['decision'] and r['decision']['meta'][0] == 0],
        'subsequent_decisions': [r for r in blocks if r['decision'] and r['decision']['meta'][0] > 0],
    }
    result = {}
    for label, rows in groups.items():
        values = np.array([r['microseconds'] for r in rows])
        result[label] = None if not len(values) else {
            'count': len(values), 'min_us': float(values.min()), 'median_us': float(np.median(values)),
            'p95_us': float(np.percentile(values, 95, method='linear')), 'max_us': float(values.max())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--build-receipt', required=True, type=Path)
    parser.add_argument('--port')
    parser.add_argument('--upload-receipt', type=Path)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100:
        parser.error('--repetitions must be in 1..100')
    if args.port and not args.upload_receipt:
        parser.error('--port requires a receipt for the authorized, verified timing-image upload')
    output = local_output(args.output)
    if output.exists():
        raise ValueError('Output already exists; refusing to overwrite')
    build = verify_build(args.build_receipt, args.upload_receipt)
    samples_by_case = fixtures()
    parameters = {key: np.asarray(value, dtype=np.float32).astype(np.float64) for key, value in
                  parse_c_header(ROOT / 'firmware/Arduino Nano 33/include/kan_model_weights.h').items()}
    expected = {name: host_results(args.probe.resolve(), samples) for name, samples in samples_by_case.items()}
    checks = {name: python_check(samples, expected[name], parameters) for name, samples in samples_by_case.items()}
    # Retain every original raw-unit feature error as well as the adopted budget.
    compute = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype='bandpass', fs=50., output='sos')
    for name, rows in expected.items():
        for i, row in enumerate(rows):
            start, emitted = i * 250, i * 250 + 750
            left = max(0, start - 250)
            filtered = signal.sosfiltfilt(sos, samples_by_case[name][left:emitted].astype(np.float64))
            features = list(compute(filtered[start - left:start - left + 500], 50.).values())
            checked = feature_check(features, row['values'][1:])
            checks[name][i]['features'] = checked
            checks[name][i]['pass'] = checks[name][i]['pass'] and checked['pass']
    host_pass = all(r['pass'] for rows in checks.values() for r in rows)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_not_run',
        'parity_complete': False, 'board_run_completed': False, 'physical_gpio_verified': False,
        'all_14_attributions_observed': False, 'host_pass': host_pass, 'host_checks': checks,
        'reference': expected, 'protocol': IDENTITY, 'repetitions': args.repetitions,
        'build_receipt_sha256': sha256(args.build_receipt), 'image_sha256': build['image_sha256'],
        'probe_sha256': sha256(args.probe), 'runner_sha256': sha256(__file__),
        'candidate_sha256': FROZEN_CANDIDATE_SHA256, 'contract_sha256': CONTRACT_SHA256,
        'input_sha256': {name: hashlib.sha256(samples.tobytes()).hexdigest() for name, samples in samples_by_case.items()},
        'latency_summary': None, 'observations': {},
        'scope': 'Accelerated synthetic USB replay. DWT covers all 250 pushes per block, including the normal decision and XAI when due. Parsing/printing/memory queries are outside the interval; interrupts remain enabled. Raw timings include instrumentation overhead. No sensor/BLE latency, all-14 diagnostic recomputation, recorded validation reuse, or complete parity.',
        'memory_scope': 'Current loop-thread stack watermark since boot, if supported; newlib heap snapshots at query times. Neither is total peak RAM; ISR/other-thread stacks and transient heap peaks are not measured.',
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'report.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    if not host_pass:
        report['status'] = 'host_failed'
        save()
        return 1
    if not args.port:
        print(f'Prepared {sum(map(len, expected.values()))} synthetic decisions; board not run. {report_path}')
        return 0
    import serial
    report.update(status='running', port=args.port, upload_receipt_sha256=sha256(args.upload_receipt))
    save()
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            replay_session(connection, samples_by_case, expected, args.repetitions, report['observations'])
        report.update(status='completed', board_run_completed=True,
                      latency_summary=timing_summary(report['observations']['blocks']))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print(f'Timing replay completed; physical GPIO comparison remains pending. {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
