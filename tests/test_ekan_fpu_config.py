"""Self-contained host tests for the opt-in edge-KAN hardware-FPU config and aggregate summary.

No serial port, no PlatformIO build, and no OpenOximetry dataset required.
"""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import archive_ekan_fpu_build as fpu_build  # noqa: E402
import record_v2_upload_receipt as receipts  # noqa: E402
import replay_edge_kan_board as replay  # noqa: E402

FW = ROOT / "firmware" / "Arduino Nano 33"
FPU_FLAGS = FW / "fpu_flags.py"
INI = FW / "platformio.ini"
PLAN_DOC = ROOT / "docs" / "module8-review-fpu-plan.md"
SUMMARY_JSON = ROOT / "outputs" / "edge-kan-fpu" / "summary.json"

FPU_ENVS = {
    "nano33ble_ekan_context_parity_fpu": "nano33ble_ekan_context_parity",
    "nano33ble_ekan_pipeline_timing_fpu": "nano33ble_ekan_pipeline_timing",
    "nano33ble_ekan_context_timing_fpu": "nano33ble_ekan_context_timing",
}

FORBIDDEN_PATTERNS = (
    re.compile(r"(?i)(?:[a-z]:[/\\]|/[a-z]/)users[/\\]"),
    re.compile(r"(?i)users[/\\]jason"),
    re.compile(r"(?i)laptop-[a-z0-9]+"),
    re.compile(r"(?i)hermes-[a-z0-9]+"),
    re.compile(r"(?i)claude-" + "science"),
    re.compile(r"(?i)[a-z]:\\fyrp-" + "xai-tinyml"),
)


def _ini_sections(text):
    sections = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^\[([^\]]+)\]\s*$", line.strip())
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def _synthetic_log(env, *, banner=True, soft=False, fpv4=True):
    lines = [f"Processing {env} (platform: nordicnrf52; board: nano33ble; framework: arduino)"]
    if banner:
        lines.append(fpu_build.BANNER + " to 3 build environments")
    flags = "-mfloat-abi=softfp" if not soft else "-mfloat-abi=soft"
    if fpv4:
        flags += " -mfpu=fpv4-sp-d16"
    lines.append(f"arm-none-eabi-g++ -c -Os {flags} src/main.cpp -o main.o")
    lines.append(f"arm-none-eabi-g++ -Os {flags} @link_args.txt -o firmware.elf")
    lines.append("======================== [SUCCESS] Took 1.00 seconds ========================")
    return "\n".join(lines) + "\n"


def _iter_items(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _iter_items(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_items(item)


class FpuFirmwareConfigTests(unittest.TestCase):
    def test_fpu_flags_script_switches_soft_to_softfp_fpv4(self):
        self.assertTrue(FPU_FLAGS.exists())
        text = FPU_FLAGS.read_text(encoding="utf-8")
        self.assertIn("-mfloat-abi=soft", text)
        self.assertIn("-mfloat-abi=softfp", text)
        self.assertIn("-mfpu=fpv4-sp-d16", text)

    def test_platformio_opt_in_fpu_envs_leave_soft_float_envs_unchanged(self):
        sections = _ini_sections(INI.read_text(encoding="utf-8"))
        for fpu_env, base_env in FPU_ENVS.items():
            key = f"env:{fpu_env}"
            self.assertIn(key, sections)
            self.assertIn(f"extends = env:{base_env}", sections[key])
            self.assertIn("extra_scripts = post:fpu_flags.py", sections[key])

        for soft_env in (
            "env:nano33ble",
            "env:nano33ble_ekan_f1a",
            "env:nano33ble_ekan_context_parity",
            "env:nano33ble_ekan_pipeline_timing",
            "env:nano33ble_ekan_context_timing",
            "env:nano33ble_ekan_f1a_ref0",
            "env:nano33ble_ekan_context_parity_ref0",
            "env:nano33ble_ekan_pipeline_timing_ref0",
        ):
            self.assertIn(soft_env, sections)
            self.assertNotIn("fpu_flags.py", sections[soft_env])


class FpuBuildAndReceiptTests(unittest.TestCase):
    def test_archive_check_log_accepts_valid_and_rejects_bad_logs(self):
        env = "nano33ble_ekan_pipeline_timing_fpu"
        checked = fpu_build.check_log(env, _synthetic_log(env))
        self.assertEqual(checked["compile_lines"], 1)
        self.assertEqual(checked["lines_with_mfloat_abi_soft"], 0)
        self.assertEqual(checked["lines_lacking_softfp_fpu"], 0)

        with self.assertRaises(SystemExit):
            fpu_build.check_log(env, _synthetic_log(env, banner=False))
        with self.assertRaises(SystemExit):
            fpu_build.check_log(env, _synthetic_log(env, soft=True))
        with self.assertRaises(SystemExit):
            fpu_build.check_log(env, _synthetic_log(env, fpv4=False))

    def test_receipt_registry_includes_all_fpu_archives(self):
        for env in FPU_ENVS:
            key = f"outputs/firmware-ekan-fpu/{env}"
            self.assertEqual(receipts.EKAN_FPU_ARCHIVES.get(key), env)
            self.assertEqual(receipts.ARCHIVES.get(key), env)


class FpuReplayDriverTests(unittest.TestCase):
    def test_configure_fpu_updates_state_and_restores(self):
        saved = (
            replay.PREREG,
            replay.PIPELINE_PREREG,
            replay.OUT_ROOT,
            replay.ARCHIVE_ROOT,
            replay.FPU,
            replay.PIPELINE_ENV,
            copy.deepcopy(replay.PROFILES),
        )
        try:
            replay.configure_fpu()
            self.assertTrue(replay.FPU)
            self.assertEqual(replay.PREREG, ROOT / "docs/module8-review-fpu-plan.md")
            self.assertEqual(replay.PIPELINE_PREREG, ROOT / "docs/module8-review-fpu-plan.md")
            self.assertEqual(replay.OUT_ROOT, ROOT / "outputs/edge-kan-fpu")
            self.assertEqual(replay.ARCHIVE_ROOT, ROOT / "outputs/firmware-ekan-fpu")
            self.assertEqual(replay.PIPELINE_ENV, "nano33ble_ekan_pipeline_timing_fpu")
            self.assertEqual(replay.PROFILES["parity"][0], "nano33ble_ekan_context_parity_fpu")
            self.assertEqual(replay.PROFILES["timing"][0], "nano33ble_ekan_context_timing_fpu")
        finally:
            (
                replay.PREREG,
                replay.PIPELINE_PREREG,
                replay.OUT_ROOT,
                replay.ARCHIVE_ROOT,
                replay.FPU,
                replay.PIPELINE_ENV,
                replay.PROFILES,
            ) = saved

    def test_cli_rejects_ref0_and_fpu_together(self):
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts/replay_edge_kan_board.py"), "host-check", "--ref0", "--fpu"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("--ref0 and --fpu are separate jobs", done.stderr)


class FpuAggregateSummaryTests(unittest.TestCase):
    def test_summary_json_matches_reported_metrics_and_is_scrubbed(self):
        raw = SUMMARY_JSON.read_text(encoding="utf-8")
        plan_text = PLAN_DOC.read_text(encoding="utf-8")
        for label, text in (("summary.json", raw), ("module8-review-fpu-plan.md", plan_text)):
            for pat in FORBIDDEN_PATTERNS:
                self.assertIsNone(pat.search(text), f"{label} matched {pat.pattern}")

        data = json.loads(raw)
        for section in (data, data["board_pipeline_20261119"], data["board_timing_20261119"]):
            for forbidden in ("functional", "timing_blocks", "decisions", "blocks", "calibration"):
                self.assertNotIn(forbidden, section)

        for key, val in _iter_items(data):
            self.assertNotIn(key, ("functional", "timing_blocks", "blocks"))
            if key == "decisions":
                self.assertIsInstance(val, int)

        pipe_summary = data["board_pipeline_20261119"]["summary"]
        self.assertEqual(pipe_summary["functional_decisions"], 1243)
        self.assertEqual(pipe_summary["functional_failed"], 0)
        sub = pipe_summary["latency"]["subsequent_decisions"]
        self.assertEqual(round(sub["median_us"] / 1000, 2), 19.71)
        self.assertEqual(round(sub["p95_us"] / 1000, 2), 19.77)

        timing_summary = data["board_timing_20261119"]["summary"]
        self.assertEqual(timing_summary["checks_failed"], 0)
        self.assertEqual(round(timing_summary["edge_forward_min_cycles"]["median_us"] / 1000, 2), 0.49)
        self.assertEqual(round(timing_summary["edge_xai_incremental_min_cycles"]["median_us"] / 1000, 2), 3.00)


if __name__ == "__main__":
    unittest.main()
