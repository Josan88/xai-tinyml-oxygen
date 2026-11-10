"""Archive the compiled v2 images (ADR-0006) and the v1 rollback image. Never flashes.

Copies firmware.bin/.elf from firmware/Arduino Nano 33/.pio/build/<env>/ for the v2 envs into
outputs/firmware-v2-prepared/<env>/ and the v1 live env into outputs/firmware-v1-rollback/,
and writes a build receipt with image, source and build-log SHA-256 and the linker sizes.
Write-once: refuses to touch an existing output directory. Status is prepared_not_flashed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware/Arduino Nano 33"
V2_ENVS = ("nano33ble_v2", "nano33ble_v2_context_parity", "nano33ble_v2_context_timing")
V1_LIVE = "nano33ble"
IDENTITY = {
    "nano33ble_v2": "serial banner 'Model: TinyKAN v2 (16 inputs ...)'; Feat line has 16 names; BLE name OxygenRiskDemo",
    "nano33ble_v2_context_parity": "CONTEXT_PARITY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED",
    "nano33ble_v2_context_timing": "TIMING_REPLAY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED",
    "nano33ble": "v1 live image: Feat line has 14 names, no 'Model: TinyKAN v2' banner; BLE name OxygenRiskDemo",
}
SOURCES = ["include/feature_extraction.h", "include/kan_inference.h", "include/kan_model_weights.h",
           "include/kan_model_weights_v2.h", "include/oxygen_risk_pipeline.h", "include/timing_pipeline.h",
           "include/timing_pipeline_v2.h", "src/main.cpp", "src/context_parity_replay_v2.cpp",
           "src/context_timing_replay_v2.cpp", "platformio.ini"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def linker_sizes(log_text):
    sizes = {}
    envs =re.findall(r"Processing (\S+) \(", log_text)
    ram = re.findall(r"RAM:.*?used (\d+) bytes from (\d+)", log_text)
    flash = re.findall(r"Flash:.*?used (\d+) bytes from (\d+)", log_text)
    for env, r, f in zip(envs, ram, flash):
        sizes[env] = {"flash_bytes": int(f[0]), "flash_total": int(f[1]), "static_ram_bytes": int(r[0]), "ram_total": int(r[1])}
    return sizes


def archive(env, out_dir):
    out_dir.mkdir(parents=True)
    entry = {}
    for name in ("firmware.bin", "firmware.elf"):
        source = FW / ".pio/build" / env / name
        shutil.copy2(source, out_dir / name)
        if sha256(source) != sha256(out_dir / name):
            raise RuntimeError(f"Copy mismatch: {source}")
        entry[name.replace(".", "_") + "_sha256"] = sha256(out_dir / name)
    entry["image_path"] = str((out_dir / "firmware.bin").relative_to(ROOT)).replace("\\", "/")
    entry["image_bytes"] = (out_dir / "firmware.bin").stat().st_size
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-log", type=Path, required=True)
    args = parser.parse_args()
    v2_root, v1_root = ROOT / "outputs/firmware-v2-prepared", ROOT / "outputs/firmware-v1-rollback"
    for path in (v2_root, v1_root):
        if path.exists():
            parser.error(f"{path} exists; refusing to overwrite")
    log_text = args.build_log.read_text(encoding="utf-8", errors="replace")
    sizes = linker_sizes(log_text)
    if any(env not in sizes for env in V2_ENVS + (V1_LIVE,)):
        parser.error("Build log does not cover every archived env")
    created = datetime.now(timezone.utc).isoformat()
    sources = {f"firmware/Arduino Nano 33/{p}": sha256(FW / p) for p in SOURCES}
    images = {}
    for env in V2_ENVS:
        images[env] = {**archive(env, v2_root / env), **sizes[env], "identity": IDENTITY[env]}
    receipt = {"id": "nano33ble-v2-images-prepared", "status": "prepared_not_flashed", "flashed": False,
               "created_utc": created, "model": "kan_model_weights_v2.h (16 inputs, ADR-0006)",
               "build_flag": "-DOXY_MODEL_V2", "images": images, "source_sha256": sources,
               "build_log": str(args.build_log.resolve()), "build_log_sha256": sha256(args.build_log),
               "toolchain": "PlatformIO nordicnrf52 / framework-arduino-mbed, -Os -std=gnu++14 (as v1)",
               "scope": "Compiled, not flashed, not run on the board. Board parity, timing, peak RAM and a live sit are outstanding."}
    with (v2_root / "build-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    (v2_root / "SHA256SUMS").write_text("".join(f"{v['firmware_bin_sha256']}  {env}/firmware.bin\n" for env, v in images.items()))
    v1 = {**archive(V1_LIVE, v1_root / V1_LIVE), **sizes[V1_LIVE], "identity": IDENTITY[V1_LIVE]}
    rollback = {"id": "nano33ble-v1-live-rollback", "status": "archived_not_flashed", "created_utc": created,
                "images": {V1_LIVE: v1}, "source_sha256": sources, "build_log_sha256": sha256(args.build_log),
                "provenance": "Rebuilt from the v1 sources; byte-identical to the pre-change HEAD 1dd1c74 build (build/v2/v1-baseline-bin-sha256.txt). Whether it equals the image currently on the board is not verified (no board access).",
                "baseline_build_log": str((ROOT / "build/v2/v1-baseline-build.log").resolve())}
    with (v1_root / "rollback-receipt.json").open("x") as stream:
        json.dump(rollback, stream, indent=2)
        stream.write("\n")
    (v1_root / "SHA256SUMS").write_text(f"{v1['firmware_bin_sha256']}  {V1_LIVE}/firmware.bin\n")
    print(json.dumps({env: {k: v[k] for k in ("firmware_bin_sha256", "flash_bytes", "static_ram_bytes")}
                      for env, v in {**images, V1_LIVE: v1}.items()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
