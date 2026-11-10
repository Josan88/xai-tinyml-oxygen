"""Archive one compiled PlatformIO env in the r2 layout. Compile-side only: never flashes, no port.

Writes a fresh <out>/ with
  firmware.bin          copied from firmware/Arduino Nano 33/.pio/build/<env>/ (hash-checked)
  build.log             the clean `pio run -e <env>` log that produced it
  source.patch          git diff <base-commit> HEAD of the firmware tree (plus tests/v2_stream_probe.cpp)
  source-change.patch   optional: the same diff from --change-base, i.e. only this job's change
  build-receipt.json    status prepared_not_flashed, image/ELF SHA-256, RAM/flash lines, commit, identity
  README.md             copied from --readme
  SHA256SUMS            "<sha256> *<name>" for every file above (record_v2_upload_receipt.py reads it)
Refuses an existing <out>, a firmware tree that differs from HEAD, or a log for another env.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from archive_v2_firmware import linker_sizes  # noqa: E402

FW = ROOT / "firmware/Arduino Nano 33"
TRACKED = ["firmware/Arduino Nano 33", "tests/v2_stream_probe.cpp"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-commit", required=True, help="source.patch base, e.g. the commit a shipped image was built from")
    parser.add_argument("--change-base", help="optional base for source-change.patch (this job's change only)")
    parser.add_argument("--identity", required=True, help="how the image identifies itself (free text)")
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error(f"{args.out} exists; refusing to overwrite")
    dirty = git("status", "--porcelain", "--", *TRACKED).strip()
    if dirty:
        parser.error(f"Firmware tree differs from HEAD:\n{dirty}")
    log = args.build_log.read_text(encoding="utf-8", errors="replace")
    sizes = linker_sizes(log)
    if list(sizes) != [args.env] or not re.search(rf"^{re.escape(args.env)}\s+SUCCESS", log, flags=re.M):
        parser.error(f"{args.build_log} is not one successful build of {args.env}")
    built = FW / ".pio/build" / args.env
    args.out.mkdir(parents=True)
    shutil.copy2(built / "firmware.bin", args.out / "firmware.bin")
    if sha256(built / "firmware.bin") != sha256(args.out / "firmware.bin"):
        raise RuntimeError("Copy mismatch")
    shutil.copy2(args.build_log, args.out / "build.log")
    (args.out / "source.patch").write_text(git("diff", args.base_commit, "HEAD", "--", *TRACKED), encoding="utf-8", newline="\n")
    if args.change_base:
        (args.out / "source-change.patch").write_text(git("diff", args.change_base, "HEAD", "--", *TRACKED),
                                                      encoding="utf-8", newline="\n")
    head = git("rev-parse", "HEAD").strip()
    packages = re.findall(r"^ - (\S+) @ (\S+)", log, flags=re.M)
    receipt = {"id": f"{args.env}-prepared", "status": "prepared_not_flashed", "flashed": False,
               "created_utc": datetime.now(timezone.utc).isoformat(), "env": args.env, "git_commit": head,
               "firmware_tree_clean": True, "base_commit": args.base_commit, "change_base": args.change_base,
               "image_path": str((args.out / "firmware.bin").relative_to(ROOT).as_posix()),
               "firmware_bin_sha256": sha256(args.out / "firmware.bin"), "image_bytes": (args.out / "firmware.bin").stat().st_size,
               "firmware_elf_sha256_not_archived": sha256(built / "firmware.elf"), **sizes[args.env],
               "build_log_ram_flash_lines": [line.strip() for line in log.splitlines() if line.startswith(("RAM:", "Flash:"))],
               "packages": dict(packages), "identity": args.identity,
               "scope": "Compiled, not flashed, not run on the board."}
    with (args.out / "build-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    shutil.copy2(args.readme, args.out / "README.md")
    names = sorted(p.name for p in args.out.iterdir())
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256(args.out / n)} *{n}\n" for n in names), encoding="utf-8", newline="\n")
    print(json.dumps({k: receipt[k] for k in ("env", "firmware_bin_sha256", "image_bytes", "flash_bytes", "static_ram_bytes",
                                               "git_commit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
