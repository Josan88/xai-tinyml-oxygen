"""Serial helpers shared by the context replay and timing drivers (v2, P5, edge KAN).

The line protocol is the one in src/context_*_replay*.cpp: ASCII commands ending in '\n', written in
64-byte USB chunks; one response line per command. Blocks carry 250 IR/red pairs as FP32 hex words.
Moved here from replay_context_board_v2.py and replay_edge_kan_board.py without behaviour changes.
"""
import time

from replay_board_parity import encode

BLOCK_PAIRS = 250


def write_command(connection, command):
    data = command.encode("ascii")
    for offset in range(0, len(data), 64):
        piece = data[offset:offset + 64]
        if connection.write(piece) != len(piece):
            raise RuntimeError("Short USB write")
        connection.flush()
        time.sleep(.01)


def receive(connection, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = connection.readline().decode("ascii").strip()
        if line:
            return line
    raise RuntimeError("Diagnostic response timed out")


def receive_after(connection, command, timeout=15):
    write_command(connection, command)
    return receive(connection, timeout)


def block_command(ir, red):
    if len(ir) != BLOCK_PAIRS or len(red) != BLOCK_PAIRS:
        raise ValueError("Block must contain exactly 250 IR/red pairs")
    words = []
    for a, b in zip(ir, red):
        words += [encode(float(a)), encode(float(b))]
    return "B " + " ".join(words) + "\n"
