"""Timing transport tests; simulated records are not device measurements."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_board_parity import encode
from benchmark_context_board import IDENTITY, parse_timed, parse_memory, replay_session
import benchmark_context_board as benchmark


class TimingProtocolTests(unittest.TestCase):
    def test_build_receipt_cannot_override_frozen_source_digest(self):
        # An isolated filesystem/configuration boundary; no production files edited.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'pipeline.h'
            source.write_text('frozen implementation')
            candidate = root / 'outputs/parity/qualification/frozen-candidate-v1.json'
            candidate.parent.mkdir(parents=True)
            candidate.write_text(json.dumps({'contract_sha256': benchmark.CONTRACT_SHA256,
                'source_sha256': {str(source): benchmark.sha256(source)}}))
            frozen_hash = benchmark.sha256(candidate)
            source.write_text('changed implementation')
            image = root / 'firmware.bin'
            image.write_bytes(b'image')
            receipt = root / 'receipt.json'
            receipt.write_text(json.dumps({'candidate_sha256': frozen_hash,
                'source_sha256': {str(source): benchmark.sha256(source)},
                'image_path': str(image), 'image_sha256': benchmark.sha256(image)}))
            with patch.object(benchmark, 'ROOT', root), patch.object(benchmark, 'FROZEN_CANDIDATE_SHA256', frozen_hash):
                with self.assertRaises(ValueError):
                    benchmark.verify_build(receipt)

    def decision(self):
        fields = ['0', '750', '0', '500', '250']
        fields += [encode(.4)] + [encode(1.)] * 14
        fields += ['0', '0', '0', '2', encode(.05), '3', encode(.04), '4', encode(.03)]
        return 'TIMED 64000 1 RESULT ' + ' '.join(fields)

    def test_cycle_records_decode_with_explicit_warmup_and_decision_scope(self):
        warmup = parse_timed('TIMED 64 0 BUFFER 250', 250)
        self.assertEqual(warmup['cycles'], 64)
        self.assertIsNone(warmup['decision'])
        result = parse_timed(self.decision(), 750)
        self.assertEqual(result['microseconds'], 1000.)
        self.assertEqual(result['decision']['meta'], [0, 750, 0, 500, 250])
        self.assertAlmostEqual(result['decision']['values'][0], .4)

    def test_malformed_stale_or_invalid_clock_records_are_rejected(self):
        examples = [('', 750), ('TIMED 64 0 BUFFER 250', 500),
                    ('TIMED 64 0 BUFFER 750', 750), (self.decision(), 1000),
                    (self.decision().replace('TIMED', 'RESULT', 1), 750),
                    (self.decision().replace('64000', '0', 1), 750),
                    (self.decision().replace('64000', '4294967296', 1), 750),
                    (self.decision().replace('64000 1', '64000 60000', 1), 750),
                    (self.decision().replace('64000 1', '64000 -1', 1), 750),
                    (self.decision() + ' extra', 750)]
        for line, total in examples:
            with self.subTest(line=line[:40], total=total), self.assertRaises(ValueError):
                parse_timed(line, total)

    def test_memory_distinguishes_stack_watermark_from_heap_snapshot(self):
        memory = parse_memory('MEMORY 32768 8192 16384 12000 4384')
        self.assertEqual(memory['main_stack_peak_used_bytes'], 8192)
        self.assertEqual(memory['heap_used_snapshot_bytes'], 12000)
        self.assertIsNone(parse_memory('MEMORY 32768 -1 16384 12000 4384')['main_stack_peak_used_bytes'])
        for line in ['MEMORY 100 101 16384 12000 4384', 'MEMORY 100 -2 10 5 5',
                     'MEMORY 100 50 10 11 0', 'MEMORY 100 50 10 5']:
            with self.assertRaises(ValueError):
                parse_memory(line)

    def test_wrong_firmware_is_refused_before_any_sample_or_reset_command(self):
        class Port:
            def __init__(self): self.writes = []
            def write(self, data): self.writes.append(data); return len(data)
            def flush(self): pass
            def readline(self): return b'CONTEXT_PARITY_V1\n'
        port = Port()
        with patch('replay_context_board.time.sleep'), self.assertRaises(RuntimeError):
            replay_session(port, {}, {}, 1, {})
        self.assertEqual(b''.join(port.writes), b'H\n')

    def test_session_keeps_raw_measurements_separate_from_warmup_and_overhead(self):
        responses = [IDENTITY, 'MEMORY 32768 8192 16384 12000 4384']
        responses += ['CAL 64 0'] * 8
        responses += ['RESET 0', 'ERR invalid_command', 'ERR invalid_command',
                      'TIMED 64 0 BUFFER 250', 'TIMED 64 0 BUFFER 500', self.decision(),
                      'MEMORY 32768 9000 16384 12000 4384']
        class Port:
            def write(self, data): return len(data)
            def flush(self): pass
            def readline(self): return (responses.pop(0) + '\n').encode()
        evidence = {}
        expected = parse_timed(self.decision(), 750)['decision']
        with patch('replay_context_board.time.sleep'):
            replay_session(Port(), {'example': [0.] * 750}, {'example': [expected]}, 1, evidence)
        self.assertFalse(responses)
        self.assertEqual(len(evidence['blocks']), 3)
        self.assertEqual(len(evidence['calibration']), 8)
        self.assertEqual(len(evidence['memory']), 2)
        self.assertEqual(evidence['blocks'][-1]['microseconds'], 1000.)
        self.assertEqual(evidence['blocks'][-1]['feature_absolute_errors'], [0.] * 14)
        # A failed functional check retains the offending timing/decision record.
        responses[:] = [IDENTITY, 'MEMORY 32768 8192 16384 12000 4384'] + ['CAL 64 0'] * 8
        responses += ['RESET 0', 'ERR invalid_command', 'ERR invalid_command',
                      'TIMED 64 0 BUFFER 250', 'TIMED 64 0 BUFFER 500',
                      self.decision().replace(encode(.4), encode(.6), 1)]
        failed = {}
        with patch('replay_context_board.time.sleep'), self.assertRaises(RuntimeError):
            replay_session(Port(), {'example': [0.] * 750}, {'example': [expected]}, 1, failed)
        self.assertFalse(failed['blocks'][-1]['functional_check']['pass'])


if __name__ == '__main__':
    unittest.main()
