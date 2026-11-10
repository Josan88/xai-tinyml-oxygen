"""v2 board drivers against a simulated CONTEXT_PARITY_V2 / TIMING_REPLAY_V2 board. No serial port.

The fake board computes decisions with the Python deployed-path reference and answers in the
firmware's exact wire format (FP32 hex words), so these tests exercise the drivers' framing,
parsing, state checks and frozen-criteria judging end to end.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replay_context_board_v2 as board  # noqa: E402
import benchmark_context_board_v2 as timing  # noqa: E402
import v2_deployed_reference as ref  # noqa: E402
from replay_board_parity import encode, decode  # noqa: E402

P = ref.load_parameters()


def cpp_like_rows(python, segment=0):
    """Host-probe row layout from a Python stream, rounded to FP32 like the firmware output."""
    rows = []
    f32 = lambda v: float(np.float32(v))  # noqa: E731
    for i in range(len(python["probability"])):
        top = python["top3"][i]
        rows.append([segment, *python["meta"][i], f32(python["probability"][i]), *map(f32, python["features"][i]),
                     int(python["candidate"][i]), int(python["confirmed"][i]), int(python["streak"][i]),
                     *[v for k in top for v in (int(k), f32(python["deltas"][i][k]))], int(python["valid"][i]),
                     *map(f32, python["deltas"][i])])
    return np.array(rows, dtype=np.float64)


class FakeBoard:
    def __init__(self, identity=board.IDENTITY, corrupt_delta=None, timing_image=False):
        self.identity, self.corrupt, self.timing = identity, corrupt_delta, timing_image
        self.pending, self.lines = b"", []
        self.reset()

    def reset(self):
        self.ir, self.red, self.deltas = [], [], None

    def reset_input_buffer(self):
        self.lines = []

    def flush(self):
        pass

    def write(self, data):
        self.pending += data
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            self.lines.append(self.handle(line.decode().strip()))
        return len(data)

    def readline(self):
        return (self.lines.pop(0) + "\n").encode() if self.lines else b""

    def result_words(self, s, i):
        words = [str(int(v)) for v in s["meta"][i]] + [encode(float(s["probability"][i]))]
        words += [encode(float(v)) for v in s["features"][i]]
        words += [str(int(s["candidate"][i])), str(int(s["confirmed"][i])), str(int(s["streak"][i]))]
        for k in s["top3"][i]:
            words += [str(int(k)), encode(float(s["deltas"][i][k]))]
        return words + [str(int(s["valid"][i]))]

    def handle(self, line):
        command, *words = line.split()
        if command == "H":
            return self.identity
        if command == "R" and not words:
            self.reset()
            return "RESET 0"
        if command == "A" and not self.timing:
            if self.deltas is None:
                return "ERR no_decision"
            return "ATTR16 " + " ".join([str(self.deltas[0])] + [encode(float(v)) for v in self.deltas[1]])
        if command == "B" and len(words) == 500 and all(len(w) == 8 for w in words):
            values = [decode(w) for w in words]
            self.ir += values[0::2]
            self.red += values[1::2]
            total = len(self.ir)
            prefix = "TIMED 16000000 250 " if self.timing else ""
            if total < 750:
                return prefix + f"BUFFER {total}"
            s = ref.stream(np.array(self.ir, dtype=np.float32), np.array(self.red, dtype=np.float32), P)
            i = len(s["probability"]) - 1
            deltas = s["deltas"][i].copy()
            if self.corrupt is not None:
                deltas[self.corrupt] += 2e-5
            self.deltas = (i, deltas)
            return prefix + "RESULT " + " ".join(self.result_words(s, i))
        if command == "M" and self.timing:
            return "MEMORY 32768 31308 1040 1040 0"
        if command == "C" and self.timing:
            return "CAL 6 0"
        return "ERR invalid_command"


class V2BoardProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = board.fixtures()
        cls.python = {n: ref.stream(ir, red, P) for n, (ir, red) in cls.fixtures.items()}

    def replay(self, fake, name):
        ir, red = self.fixtures[name]
        python = self.python[name]
        return board.replay_stream(fake, ir, red, cpp_like_rows(python), python)

    def test_all_fixture_decisions_pass_through_the_wire_format(self):
        fake = FakeBoard()
        self.assertEqual(board.receive_after(fake, "H\n"), board.IDENTITY)
        board.protocol_self_checks(fake)
        total = 0
        for name in self.fixtures:
            results = self.replay(fake, name)
            total += len(results)
            self.assertTrue(all(r["pass"] for r in results), name)
        self.assertEqual(total, 21)
        self.assertTrue(any(s["confirmed"].any() for s in self.python.values()))
        self.assertTrue(any((~s["candidate"]).any() for s in self.python.values()))

    def test_untransmitted_delta_error_fails(self):
        results = self.replay(FakeBoard(corrupt_delta=15), "risk_after_reset")
        self.assertFalse(any(r["pass"] for r in results))
        self.assertIn("ir_perfusion_index", results[0]["vs_python"]["attributions"]["failed_deltas"])

    def test_result_and_delta_framing(self):
        with self.assertRaises(ValueError):
            board.parse_result(["0"] * 31)
        with self.assertRaises(ValueError):
            board.parse_all_deltas("ATTR14 0 " + " ".join(["00000000"] * 14), 0)
        with self.assertRaises(ValueError):
            board.parse_all_deltas("ATTR16 1 " + " ".join(["00000000"] * 16), 0)
        self.assertEqual(len(board.block_command(*board.pair(1.2, 250, .6)).split()), 501)
        with self.assertRaises(ValueError):
            board.block_command(np.zeros(249), np.zeros(249))

    def test_identity_strings_differ_from_v1(self):
        self.assertTrue(board.IDENTITY.startswith("CONTEXT_PARITY_V2 "))
        self.assertTrue(timing.IDENTITY.startswith("TIMING_REPLAY_V2 "))
        self.assertNotIn("CONTEXT_PARITY_V1", board.IDENTITY)

    def test_timed_responses(self):
        fake = FakeBoard(identity=timing.IDENTITY, timing_image=True)
        self.assertEqual(board.receive_after(fake, "A\n"), "ERR invalid_command")
        ir, red = self.fixtures["risk_after_reset"]
        rows, python = cpp_like_rows(self.python["risk_after_reset"]), self.python["risk_after_reset"]
        for offset in range(0, len(ir), 250):
            total = offset + 250
            row = timing.parse_timed(board.receive_after(fake, board.block_command(ir[offset:total], red[offset:total])), total)
            self.assertEqual(row["microseconds"], 250000.)
            if row["decision"]:
                index = (total - 750) // 250
                self.assertTrue(timing.functional_check(row["decision"], rows[index], python, index)["pass"])
        with self.assertRaises(ValueError):
            timing.parse_timed("TIMED 0 0 BUFFER 250", 250)


if __name__ == "__main__":
    unittest.main()
