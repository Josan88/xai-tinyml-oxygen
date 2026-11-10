"""Protocol-only tests with a fake serial transport; never board evidence."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replay_board_parity as replay


class ReplayProtocolTests(unittest.TestCase):
    def test_fp32_hex_roundtrip(self):
        for value in (0., -0., 1.25, -999., 1e20):
            word = replay.encode(value)
            self.assertEqual(len(word), 8)
            self.assertEqual(replay.encode(replay.decode(word)), word)
        with self.assertRaises(ValueError):
            replay.decode("100000000")

    def run_fake(self, board_probability):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, probe = root / "windows.csv", root / "probe.exe"
            fixture.write_text(','.join(['0'] * 500) + '\n')
            probe.write_bytes(b'fake reference executable')
            actual = [board_probability] + [0.] * 42
            lines = [b'PARITY_READY_V1\n',
                     ('RESULT ' + ' '.join(replay.encode(x) for x in actual) + ' 0 0\n').encode()]
            writes = []
            class FakeSerial:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def readline(self): return lines.pop(0)
                def write(self, payload):
                    writes.append(payload)
                    return len(payload)
            argv = ['replay', '--probe', str(probe), '--windows', str(fixture), '--port', 'FAKE', '--output', str(root)]
            reference = subprocess.CompletedProcess([], 0, stdout=' '.join(['0'] * 43) + '\n')
            with patch.object(sys, 'argv', argv), patch.object(replay.subprocess, 'run', return_value=reference), \
                 patch.dict(sys.modules, {'serial': types.SimpleNamespace(Serial=lambda *a, **k: FakeSerial())}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = replay.main()
            self.assertEqual(len(b"".join(writes).split()), 500)
            report = json.loads((root / 'board_result.json').read_text())
            return code, report

    def test_successful_protocol(self):
        code, report = self.run_fake(0.)
        self.assertEqual(code, 0)
        self.assertTrue(report['board_probability_verified'])

    def test_failed_comparison_is_not_verified(self):
        code, report = self.run_fake(.1)
        self.assertEqual(code, 1)
        self.assertFalse(report['board_probability_verified'])


if __name__ == '__main__':
    unittest.main()
