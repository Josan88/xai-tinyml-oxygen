"""Transport/validation tests only; these are not board execution evidence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_context_board import parse_result, compare, write_command, parse_all_deltas
from replay_board_parity import encode


class ContextProtocolTests(unittest.TestCase):
    def test_all_deltas_reject_stale_truncated_and_invalid_responses(self):
        words = ["ATTR14", "3"] + [encode(.01 * i) for i in range(14)]
        self.assertEqual(len(parse_all_deltas(" ".join(words), 3)), 14)
        examples = [words[:-1], ["ATTR14", "2"] + words[2:], words[:2] + ["7FC00000"] + words[3:]]
        for bad in examples:
            with self.assertRaises(ValueError):
                parse_all_deltas(" ".join(bad), 3)

    def fields(self):
        return ['0', '750', '0', '500', '250'] + ['0.4'] + ['67208718.5'] * 14 + ['0', '0', '0', '2', '.05', '3', '.04', '4', '.03']

    def test_decimal_host_and_hex_board_compare_as_fp32(self):
        words = self.fields()
        board = words.copy()
        for i in list(range(5, 20)) + [24, 26, 28]:
            board[i] = encode(float(board[i]))
        result = compare(parse_result(board, True), parse_result(words))
        self.assertTrue(result['pass'])
        self.assertTrue(result['strict_raw_feature_pass'])

    def test_wrong_timing_or_state_fails(self):
        expected = parse_result(self.fields())
        for index, value in [(1, '500'), (4, '0'), (21, '1'), (22, '2')]:
            changed = self.fields()
            changed[index] = value
            self.assertFalse(compare(parse_result(changed), expected)['pass'])

    def test_truncated_duplicate_rank_and_nan_rejected(self):
        examples = [self.fields()[:-1], self.fields(), self.fields()]
        examples[1][25] = examples[1][23]
        examples[2][5] = 'nan'
        for words in examples:
            with self.assertRaises(ValueError):
                parse_result(words)

    def test_chunked_writes_and_short_write_failure(self):
        class Port:
            writes = []
            def write(self, data):
                self.writes.append(data)
                return len(data)
            def flush(self): pass
        port = Port()
        with patch('replay_context_board.time.sleep'):
            write_command(port, 'B ' + '01234567 ' * 250 + '\n')
        self.assertTrue(all(len(data) <= 64 for data in port.writes))
        self.assertEqual(b''.join(port.writes), ('B ' + '01234567 ' * 250 + '\n').encode())
        port.write = lambda data: 0
        with self.assertRaises(RuntimeError):
            write_command(port, 'H\n')


if __name__ == '__main__':
    unittest.main()
