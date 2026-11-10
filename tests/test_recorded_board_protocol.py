"""Protocol-only tests with a fake serial transport; never board evidence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_board_parity import encode, decode
from qualify_recorded_board import split_blocks, expected_final, evaluate_case, replay_case, identify
from replay_context_board import parse_result


def features():
    return np.array([0., 5., -10., 10., 20., 5., 0., 0., 12500., 1., 1.171875, 20., 10., 2.])


def deltas():
    return np.array([.05, .04, .03, .02, .01] + [0.] * 9)


def native_row(count, probability, feat, attr, first=None):
    emissions, offset = (1, 0) if count == 750 else (2, 250)
    first = probability if first is None else first
    candidate, first_c = int(probability >= .5), int(first >= .5)
    streak = (1 + first_c if emissions == 2 else 1) if candidate else 0
    top = np.argsort(-np.abs(attr), kind="stable")[:3]
    row = np.zeros(546)
    row[1], row[2] = emissions, first
    row[3:8] = [emissions - 1, count, offset, offset + 500, 250]
    row[8], row[9:23] = probability, feat
    row[23:26] = [candidate, int(streak >= 2), streak]
    for i, index in enumerate(top):
        row[26 + 2 * i], row[27 + 2 * i] = index, attr[index]
    row[32:46] = attr
    return row


def result_line(meta, values, states, top):
    words = ["RESULT"] + [str(x) for x in meta] + [encode(v) for v in values]
    words += [str(x) for x in states]
    for index, delta in top:
        words += [str(index), encode(delta)]
    return " ".join(words)


def parsed(row, count):
    expected = expected_final(row, count)
    return expected, result_line(expected["meta"], expected["values"], expected["states"], expected["top"])


class Fake:
    def __init__(self, lines):
        self.lines = list(lines)
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return (self.lines.pop(0) + "\n").encode()


class RecordedBoardProtocolTests(unittest.TestCase):
    def test_split_blocks_and_native_timing(self):
        raw = np.arange(1000, dtype=np.float32)
        self.assertEqual(len(split_blocks(raw, 750)), 3)
        self.assertEqual(len(split_blocks(raw, 1000)), 4)
        with self.assertRaises(ValueError):
            split_blocks(raw, 500)
        row = native_row(1000, .6, features(), deltas(), first=.4)
        expected = expected_final(row, 1000)
        self.assertEqual(expected["meta"], [1, 1000, 250, 750, 250])
        self.assertEqual(expected["states"], [1, 0, 1])

    def test_evaluate_retains_original_absolute_failure(self):
        feat, attr = features(), deltas()
        row = native_row(750, .4, feat, attr)
        expected = expected_final(row, 750)
        board = parse_result(result_line(expected["meta"], expected["values"], expected["states"], expected["top"]).split()[1:], True)
        board["all_deltas"] = attr.tolist()
        python_feat = feat.copy()
        python_feat[8] = feat[8] + .001
        result = evaluate_case(python_feat, .4, attr, row, 750, [board], True)
        self.assertTrue(result["pass"])
        self.assertFalse(result["original_absolute_1e_4_pass"])
        self.assertGreater(result["board_vs_python"]["features"]["absolute_errors"][8], 1e-4)

    def test_identify_mismatch_sends_no_sample_block(self):
        fake = Fake(["WRONG"])
        with patch("replay_context_board.time.sleep"), self.assertRaises(RuntimeError):
            identify(fake)
        self.assertNotIn(b"B ", b"".join(fake.writes))

    def test_replay_750_and_1000_and_no_decision_after_reset(self):
        feat, attr = features(), deltas()
        with patch("replay_context_board.time.sleep"):
            row = native_row(750, .4, feat, attr)
            expected, line = parsed(row, 750)
            attr14 = "ATTR14 0 " + " ".join(encode(v) for v in attr)
            fake = Fake(["RESET 0", "ERR no_decision", "BUFFER 250", "BUFFER 500", line, attr14])
            rows, payload = replay_case(fake, np.zeros(1000, np.float32), 750)
            self.assertTrue(payload)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["all_deltas"], [decode(encode(v)) for v in attr])
            first = native_row(1000, .6, feat, attr, first=.4)
            expected = expected_final(first, 1000)
            warmup = result_line([0, 750, 0, 500, 250], [.4] + feat.tolist(), [0, 0, 0], expected["top"])
            final = result_line(expected["meta"], expected["values"], expected["states"], expected["top"])
            attr14 = "ATTR14 1 " + " ".join(encode(v) for v in attr)
            fake = Fake(["RESET 0", "ERR no_decision", "BUFFER 250", "BUFFER 500", warmup, final, attr14])
            rows, payload = replay_case(fake, np.zeros(1000, np.float32), 1000)
            self.assertTrue(payload)
            result = evaluate_case(feat, .6, attr, first, 1000, rows, payload)
            self.assertTrue(result["pass"])
            fake = Fake(["RESET 0", "ATTR14 0 " + " ".join(encode(0.) for _ in range(14))])
            with self.assertRaises(RuntimeError):
                replay_case(fake, np.zeros(750, np.float32), 750)


if __name__ == "__main__":
    unittest.main()
