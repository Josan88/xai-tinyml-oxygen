"""Write the upload receipt the v2 board drivers require, from a saved bossac log. No serial port.

Run only after Jason has flashed an archived image himself (docs/v2-flash-runbook.md). Checks
that the image is one of the archived v2 (or v1 rollback) images, that the bossac log reports
'Verify successful', and that the byte count bossac wrote equals the image size.

Without --archive the image comes from the 15 Oct build receipts (unchanged behaviour). With
--archive it comes from one of the r2 archives (22 Oct, fixed numeric source) or the label-0
explainer-reference archives (outputs/firmware-v2-ref0/<env>, 30 Oct) or the P5 archives
(outputs/firmware-p5/<env>, 30 Oct) or the SpO2 est archive (outputs/firmware-v2-f1a-spo2maxim,
5 Nov), checked against that archive's SHA256SUMS; --env must be the env that archive holds.

--dry-run opens nothing and writes nothing: it checks the archive and image hash and, if
--bossac-log is given, the log, then prints what the receipt would say.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = (ROOT / "outputs/firmware-v2-prepared/build-receipt.json",
            ROOT / "outputs/firmware-v1-rollback/rollback-receipt.json")
# r2 archive directory -> the env it holds (outputs/*-r2/README.md).
R2_ARCHIVES = {"outputs/firmware-v2-prepared-r2": "nano33ble_v2_context_parity",
               "outputs/timing-v2-prepared-r2": "nano33ble_v2_context_timing",
               "outputs/firmware-v2-f1a-r2": "nano33ble_v2_f1a"}
# Label-0-median explainer-reference images (docs/xai-reference-label0-runbook.md).
REF0_ARCHIVES = {"outputs/firmware-v2-ref0/nano33ble_v2_f1a_ref0": "nano33ble_v2_f1a_ref0",
                 "outputs/firmware-v2-ref0/nano33ble_v2_context_parity_ref0": "nano33ble_v2_context_parity_ref0"}
# P5 images (job F2, ADR-0008; docs/xai-reference-label0-runbook.md, section P5).
P5_ARCHIVES = {f"outputs/firmware-p5/{env}": env for env in
               ("nano33ble_p5_f1a", "nano33ble_p5_context_parity", "nano33ble_p5_context_timing")}
# SpO2 est from the Maxim reference algorithm (ADR-0010, docs/spo2-maxim-runbook.md). Not the paper's image.
SPO2_ARCHIVES = {"outputs/firmware-v2-f1a-spo2maxim": "nano33ble_v2_f1a_spo2"}
ARCHIVES = {**R2_ARCHIVES, **REF0_ARCHIVES, **P5_ARCHIVES, **SPO2_ARCHIVES}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_key(archive):
    """Repo-relative POSIX path of a known (r2 or ref0) archive directory, or ValueError."""
    try:
        key = Path(archive).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(f"{archive} is not inside the repository") from None
    if key not in ARCHIVES:
        raise ValueError(f"{archive} is not a known archive ({', '.join(ARCHIVES)})")
    return key


def summed_sha256(sums, name):
    """The SHA-256 a SHA256SUMS file records for one file name."""
    for line in Path(sums).read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1].strip().lstrip("*") == name:
            return parts[0].lower()
    raise ValueError(f"{sums} has no entry for {name}")


def archived_image(env, archive=None):
    """(image path relative to ROOT, recorded SHA-256, receipt file) for an archived env.

    archive None: the 15 Oct build receipts. archive set: an r2 or ref0 directory and its SHA256SUMS.
    Does not hash the image; callers compare it to the recorded value.
    """
    if archive is None:
        for path in RECEIPTS:
            images = json.loads(path.read_text())["images"]
            if env in images:
                return images[env]["image_path"], images[env]["firmware_bin_sha256"], path
        raise ValueError(f"{env} is not an archived image")
    key = archive_key(archive)
    if ARCHIVES[key] != env:
        raise ValueError(f"{key} holds {ARCHIVES[key]}, not {env}")
    sums = ROOT / key / "SHA256SUMS"
    return f"{key}/firmware.bin", summed_sha256(sums, "firmware.bin"), sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="archived env name, e.g. nano33ble_v2_context_parity")
    parser.add_argument("--archive", type=Path, help="r2 or ref0 archive directory, e.g. outputs/firmware-v2-prepared-r2 "
                                                     "(default: the 15 Oct build receipts)")
    parser.add_argument("--bossac-log", type=Path, required=False)
    parser.add_argument("--bootloader-port", help="port bossac wrote through, as typed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="check archive, image and (optional) log; write nothing")
    args = parser.parse_args()
    if not args.dry_run and not (args.bossac_log and args.bootloader_port and args.output):
        parser.error("--bossac-log, --bootloader-port and --output are required (except with --dry-run)")
    if args.output and args.output.exists():
        parser.error("Receipt exists; refusing to overwrite")
    try:
        image_path, image_sha256, receipt_path = archived_image(args.env, args.archive)
    except ValueError as error:
        parser.error(str(error))
    entry = {"image_path": image_path, "firmware_bin_sha256": image_sha256}
    image = ROOT / entry["image_path"]
    if sha256(image) != entry["firmware_bin_sha256"]:
        parser.error("Archived image no longer matches its receipt")
    if args.dry_run and not args.bossac_log:
        print(json.dumps({"dry_run": True, "env": args.env, "image_path": entry["image_path"],
                          "image_sha256": entry["firmware_bin_sha256"], "image_bytes": image.stat().st_size,
                          "build_receipt": str(receipt_path.relative_to(ROOT).as_posix()), "archive_and_image_ok": True,
                          "bossac_log": "not given"}, indent=2))
        return 0
    log = args.bossac_log.read_text(errors="replace")
    written = re.search(r"Write (\d+) bytes to flash", log)
    verified = "Verify successful" in log
    if not verified or not written or int(written.group(1)) != image.stat().st_size:
        parser.error("bossac log does not show a verified write of this image's size")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "env": args.env, "image_sha256": entry["firmware_bin_sha256"],
                          "image_bytes": image.stat().st_size, "bossac_log_ok": True, "receipt_written": False}, indent=2))
        return 0
    receipt = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "env": args.env,
               "uploaded_and_verified": True, "image_path": entry["image_path"], "image_sha256": entry["firmware_bin_sha256"],
               "image_bytes": image.stat().st_size, "bossac_log": str(args.bossac_log.resolve()),
               "bossac_log_sha256": sha256(args.bossac_log), "bootloader_port": args.bootloader_port,
               "build_receipt": str(receipt_path.relative_to(ROOT)), "build_receipt_sha256": sha256(receipt_path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
