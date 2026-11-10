"""Image-archive selection for the v2 receipt writer and board drivers. No serial port, no build.

Defaults must still resolve the 15 Oct images; --archive must resolve each r2 archive through its
SHA256SUMS and refuse a wrong env or a non-r2 directory.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import record_v2_upload_receipt as receipts  # noqa: E402
import replay_context_board_v2 as board  # noqa: E402

OLD = {"nano33ble_v2_context_parity": "52bdf7913c2d58925f1d7179f266392fd7891fbc9df01204793abc5ac535a9a0",
       "nano33ble_v2_context_timing": "1091aadcfe832408f16f17b644744717d0707544aba3f5c29a743c1ef57c4623",
       "nano33ble": "1f2f95dbd287a862e4086327f55ef3db213a31de6b70891c9deb2dcc140c07e0"}


class ArchiveSelectionTests(unittest.TestCase):
    def test_defaults_resolve_the_15_oct_images(self):
        for env, digest in OLD.items():
            path, recorded, _ = receipts.archived_image(env)
            self.assertEqual(recorded, digest)
            self.assertEqual(receipts.sha256(ROOT / path), digest)

    def test_each_r2_archive_matches_its_sha256sums(self):
        for key, env in receipts.R2_ARCHIVES.items():
            path, recorded, sums = receipts.archived_image(env, ROOT / key)
            self.assertEqual(path, f"{key}/firmware.bin")
            self.assertEqual(sums, ROOT / key / "SHA256SUMS")
            self.assertEqual(receipts.sha256(ROOT / path), recorded)
            self.assertNotIn(recorded, OLD.values())

    def test_wrong_env_or_directory_is_refused(self):
        with self.assertRaises(ValueError):
            receipts.archived_image("nano33ble_v2_f1a", ROOT / "outputs/firmware-v2-prepared-r2")
        with self.assertRaises(ValueError):
            receipts.archived_image("nano33ble_v2_context_parity", ROOT / "outputs/firmware-v2-prepared")

    def test_spo2_archive_resolves_and_refuses_the_r2_env(self):
        key = "outputs/firmware-v2-f1a-spo2maxim"
        path, recorded, sums = receipts.archived_image("nano33ble_v2_f1a_spo2", ROOT / key)
        self.assertEqual(path, f"{key}/firmware.bin")
        self.assertEqual(sums, ROOT / key / "SHA256SUMS")
        self.assertEqual(receipts.sha256(ROOT / path), recorded)
        self.assertNotEqual(recorded, receipts.archived_image("nano33ble_v2_f1a", ROOT / "outputs/firmware-v2-f1a-r2")[1])
        with self.assertRaises(ValueError):
            receipts.archived_image("nano33ble_v2_f1a", ROOT / key)
        with self.assertRaises(ValueError):
            receipts.archived_image("nano33ble_v2_f1a_spo2", ROOT / "outputs/firmware-v2-f1a-r2")

    def test_receipt_from_r2_archive(self):
        image = ROOT / "outputs/timing-v2-prepared-r2/firmware.bin"
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "bossac.log"
            log.write_text(f"Write {image.stat().st_size} bytes to flash (x pages)\nVerify successful\n")
            out = Path(tmp) / "receipt.json"
            done = subprocess.run([sys.executable, str(ROOT / "scripts/record_v2_upload_receipt.py"),
                                   "--env", "nano33ble_v2_context_timing", "--archive", "outputs/timing-v2-prepared-r2",
                                   "--bossac-log", str(log), "--bootloader-port", "COM7", "--output", str(out)],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            receipt = json.loads(out.read_text())
            self.assertEqual(receipt["image_sha256"], receipts.sha256(image))
            self.assertEqual(receipt["build_receipt"], str(Path("outputs/timing-v2-prepared-r2/SHA256SUMS")))

    def test_old_prepared_set_still_verifies_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload = Path(tmp) / "upload.json"
            upload.write_text(json.dumps({"uploaded_and_verified": True, "image_sha256": OLD["nano33ble_v2_context_parity"]}))
            manifest = board.verify_prepared(upload)
            self.assertEqual(manifest["image_sha256"], OLD["nano33ble_v2_context_parity"])
            upload.write_text(json.dumps({"uploaded_and_verified": True,
                                          "image_sha256": receipts.archived_image("nano33ble_v2_context_parity",
                                                                                  ROOT / "outputs/firmware-v2-prepared-r2")[1]}))
            with self.assertRaises(ValueError):
                board.verify_prepared(upload)

    def test_timing_refuses_r2_image_with_old_host_rows(self):
        output = ROOT / "outputs/__test_timing_never_created"
        done = subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_context_board_v2.py"),
                               "--archive", "outputs/timing-v2-prepared-r2", "--output", str(output)],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("both be r2", done.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
