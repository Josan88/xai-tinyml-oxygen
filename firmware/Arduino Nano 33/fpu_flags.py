"""PlatformIO post-script: build with the nRF52840 FPU (Module 8 review M11, 19 Nov 2026).

PlatformIO's mbed builder (platforms/nordicnrf52/builder/frameworks/arduino/mbed-core/arduino-core-mbed.py,
configure_fpu_flags) appends -mfloat-abi=soft after the framework's own flags (variants/ARDUINO_NANO33BLE/
cflags.txt: -mfloat-abi=softfp -mfpu=fpv4-sp-d16), so every image in the paper ran FP32 in software float.
This script replaces -mfloat-abi=soft with -mfloat-abi=softfp (FPU instructions, same calling convention as
soft, so the precompiled mbed libraries still link) in the main, project and library build environments.
Used only by the *_fpu envs in platformio.ini; no other image changes.
"""
Import("env")  # noqa: F821

try:
    Import("projenv")  # noqa: F821
except Exception:  # pragma: no cover - older cores
    projenv = None  # noqa: F841

SOFT = "-mfloat-abi=soft"
WANT = ["-mfloat-abi=softfp", "-mfpu=fpv4-sp-d16"]


def fix(e):
    for var in ("ASFLAGS", "CCFLAGS", "LINKFLAGS"):
        flags = [f for f in e.get(var, []) if f != SOFT and not (isinstance(f, str) and f.startswith("-mfpu="))]
        e.Replace(**{var: flags + WANT})


envs = [env]  # noqa: F821
if projenv is not None:  # noqa: F821
    envs.append(projenv)  # noqa: F821
for lb in env.GetLibBuilders():  # noqa: F821
    envs.append(lb.env)
for e in envs:
    fix(e)
print("fpu_flags.py: softfp + fpv4-sp-d16 applied to %d build environments" % len(envs))
