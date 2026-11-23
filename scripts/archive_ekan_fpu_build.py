"""Archive one edge-KAN FPU build and check it really uses the FPU (Module 8 review M11, docs/module8-review-fpu-plan.md).

Compile-side only: never flashes, opens no port. Run after a clean verbose build in your own terminal:

    pio run -v -e nano33ble_ekan_pipeline_timing_fpu > build/ekan-fpu/nano33ble_ekan_pipeline_timing_fpu.log 2>&1
    python scripts/archive_ekan_fpu_build.py --env nano33ble_ekan_pipeline_timing_fpu \
        --build-log build/ekan-fpu/nano33ble_ekan_pipeline_timing_fpu.log

Writes a fresh outputs/firmware-ekan-fpu/<env>/ in the same layout as outputs/firmware-ekan/<env>/
(firmware.bin, firmware.elf, build.log, README.md, SHA256SUMS) plus fpu-check.json. Refuses to write if
  - the output directory exists, the env is not one of the three *_fpu envs, or the log is for another env;
  - the log lacks the fpu_flags.py banner, has no compiler lines (not a -v build), or any compiler/linker line
    carries -mfloat-abi=soft or lacks -mfloat-abi=softfp / -mfpu=fpv4-sp-d16;
  - the ELF disassembly (arm-none-eabi-objdump from the PlatformIO toolchain) shows no single-precision VFP
    arithmetic in the project code, or the image is byte-identical to the soft-float archive.
fpu-check.json records the same disassembly counts for the archived soft-float image of the base env.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware/Arduino Nano 33"
OUT_ROOT = ROOT / "outputs/firmware-ekan-fpu"
SOFT_ROOT = ROOT / "outputs/firmware-ekan"
ENVS = {"nano33ble_ekan_context_parity_fpu": "nano33ble_ekan_context_parity",
        "nano33ble_ekan_pipeline_timing_fpu": "nano33ble_ekan_pipeline_timing",
        "nano33ble_ekan_context_timing_fpu": "nano33ble_ekan_context_timing"}
BANNER = "fpu_flags.py: softfp + fpv4-sp-d16 applied"
COMPILER = re.compile(r"arm-none-eabi-(g\+\+|gcc)(\.exe)?\"?\s")
VFP_ARITH = re.compile(r"\tv(add|sub|mul|div|fma|fms|fnma|fnms|nmul|neg|abs|sqrt|cvt|cmp|cmpe)\.f32\b")
SOFT_CALL = re.compile(r"<__aeabi_f(add|sub|mul|div|cmp\w*|2iz|2uiz|2d)>|<__aeabi_(i2f|ui2f|d2f)>")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def objdump():
    home = Path(os.environ.get("PLATFORMIO_CORE_DIR", Path.home() / ".platformio"))
    exe = home / "packages/toolchain-gccarmnoneeabi/bin/arm-none-eabi-objdump.exe"
    if not exe.exists():
        exe = exe.with_suffix("")
    if not exe.exists():
        raise SystemExit(f"arm-none-eabi-objdump not found under {home}/packages/toolchain-gccarmnoneeabi/bin")
    return exe


def disasm_counts(elf, tool):
    """Count single-precision VFP arithmetic and soft-float helper calls, overall and in edge-KAN/pipeline code."""
    text = subprocess.run([str(tool), "-d", "-C", str(elf)], check=True, capture_output=True, text=True).stdout
    total_vfp = total_soft = 0
    per_function, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            current = m.group(1)
            continue
        vfp, soft = bool(VFP_ARITH.search(line)), bool(SOFT_CALL.search(line))
        if vfp or soft:
            total_vfp += vfp
            total_soft += soft
            f = per_function.setdefault(current, [0, 0])
            f[0] += vfp
            f[1] += soft
    project = {k: v for k, v in per_function.items()
               if re.search(r"edge_kan|EdgeKan|ekan|OxygenRisk|extract_features|window_mean|sosfilt|filtfilt|Kahan", k or "")}
    return {"vfp_f32_arith_instructions": total_vfp, "soft_float_helper_calls": total_soft,
            "project_vfp_f32_arith_instructions": sum(v[0] for v in project.values()),
            "project_soft_float_helper_calls": sum(v[1] for v in project.values()),
            "project_functions_counted": len(project)}


def check_log(env, log):
    first = next((ln for ln in log.splitlines() if ln.startswith("Processing ")), "")
    if not first.startswith(f"Processing {env} "):
        raise SystemExit(f"Build log is not for {env}: {first[:80]!r}")
    if BANNER not in log:
        raise SystemExit("fpu_flags.py banner missing: the post-script did not run")
    if "[SUCCESS]" not in log and "SUCCESS" not in log.splitlines()[-1]:
        raise SystemExit("Build log does not end in SUCCESS")
    every = [ln for ln in log.splitlines() if COMPILER.search(ln)]
    non_pp = [ln for ln in every if not re.search(r"\s-E\s", ln)]    # skip linker-script preprocessor (-E) lines
    compiles = [ln for ln in non_pp if re.search(r"\s-c\s", ln)]     # always check -c lines (mbed uses @includes.txt)
    link_all = [ln for ln in non_pp if not re.search(r"\s-c\s", ln)]
    links = [ln for ln in link_all if not re.search(r"\s@\S", ln)]   # skip @response-file link lines only
    lines = compiles + links
    if not compiles:
        raise SystemExit("No compiler lines in the log: rebuild with `pio run -v` after a clean")
    bad_soft = [ln for ln in lines if re.search(r"-mfloat-abi=soft(\s|\"|$)", ln)]
    missing = [ln for ln in lines if "-mfloat-abi=softfp" not in ln or "-mfpu=fpv4-sp-d16" not in ln]
    if bad_soft or missing:
        raise SystemExit(f"{len(bad_soft)} compiler/linker lines still carry -mfloat-abi=soft and {len(missing)} lack "
                         f"softfp/fpv4-sp-d16; first: {(bad_soft or missing)[0][:300]}")
    return {"compile_lines": len(compiles), "compiler_and_linker_lines_checked": len(lines),
            "preprocessor_lines_skipped": len(every) - len(non_pp),
            "response_file_lines_skipped": len(link_all) - len(links),
            "lines_with_mfloat_abi_soft": 0, "lines_lacking_softfp_fpu": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=sorted(ENVS))
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default outputs/firmware-ekan-fpu/<env>")
    args = parser.parse_args()
    out = (args.out or OUT_ROOT / args.env).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; archives are write-once")
    log = args.build_log.read_text(errors="replace")
    log_check = check_log(args.env, log)
    build = FW / ".pio/build" / args.env
    bin_path, elf_path = build / "firmware.bin", build / "firmware.elf"
    for p in (bin_path, elf_path):
        if not p.exists():
            raise SystemExit(f"{p} missing")
    if bin_path.stat().st_mtime < args.build_log.stat().st_mtime - 3600:
        raise SystemExit("firmware.bin is more than an hour older than the log; rebuild")
    base = ENVS[args.env]
    soft = SOFT_ROOT / base
    if sha256(bin_path) == sha256(soft / "firmware.bin"):
        raise SystemExit("FPU image is byte-identical to the soft-float archive: the flags did not take effect")
    tool = objdump()
    fpu_counts, soft_counts = disasm_counts(elf_path, tool), disasm_counts(soft / "firmware.elf", tool)
    if fpu_counts["project_vfp_f32_arith_instructions"] == 0:
        raise SystemExit(f"No VFP f32 arithmetic in project functions: {fpu_counts}")
    out.mkdir(parents=True)
    shutil.copy2(bin_path, out / "firmware.bin")
    shutil.copy2(elf_path, out / "firmware.elf")
    (out / "build.log").write_text(log, encoding="utf-8")
    sizes = [ln.strip() for ln in log.splitlines() if ln.strip().startswith(("RAM:", "Flash:"))]
    check = {"env": args.env, "soft_float_base_env": base, "created_utc": datetime.now(timezone.utc).isoformat(),
             "build_log_check": log_check, "size_lines": sizes,
             "image_sha256": sha256(out / "firmware.bin"), "image_bytes": (out / "firmware.bin").stat().st_size,
             "soft_float_image_sha256": sha256(soft / "firmware.bin"),
             "soft_float_image_bytes": (soft / "firmware.bin").stat().st_size,
             "disassembly": {"fpu": fpu_counts, "soft_float_archive": soft_counts},
             "objdump": str(tool), "plan": "docs/module8-review-fpu-plan.md"}
    (out / "fpu-check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        f"# {args.env} (edge KAN, hardware FPU)\n\n"
        f"Module 8 review M11 (docs/module8-review-fpu-plan.md). Same sources and flags as the archived soft-float\n"
        f"image outputs/firmware-ekan/{base}/, plus firmware/Arduino Nano 33/fpu_flags.py, which replaces PlatformIO's\n"
        f"appended -mfloat-abi=soft with -mfloat-abi=softfp -mfpu=fpv4-sp-d16 in every build environment.\n"
        f"Built from the working tree on HEAD {subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}"
        f" (fpu_flags.py and the *_fpu envs in platformio.ini uncommitted at build time).\n"
        f"Build-log and disassembly checks: fpu-check.json. Diagnostic only; not a shipped image.\n", encoding="utf-8")
    names = ["firmware.bin", "firmware.elf", "build.log", "README.md", "fpu-check.json"]
    (out / "SHA256SUMS").write_text("".join(f"{sha256(out / n)} *{n}\n" for n in names), encoding="utf-8")
    print(json.dumps({k: check[k] for k in ("env", "image_sha256", "image_bytes", "soft_float_image_bytes",
                                            "build_log_check", "disassembly")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
