"""Read a Mean-Perturbation reference header written by scripts/make_xai_reference_header.py.

NumPy only (no torch/sklearn), so the board-session interpreter can import it. Returns the
reference tag and KAN_XAI_REF_Z as FP32, exactly the values the firmware holds.
"""
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER_V2 = ROOT / "firmware/Arduino Nano 33/include/kan_xai_reference_v2.h"
HEADER_P5 = ROOT / "firmware/Arduino Nano 33/include/kan_xai_reference_p5.h"   # job F2, 5-input P5 model
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_reference_header(path=HEADER_V2):
    text = re.sub(r"/\*.*?\*/", "", Path(path).read_text(encoding="utf-8"), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    tag = re.search(r'#define\s+KAN_XAI_REF_TAG\s+"([^"]+)"', text)
    body = re.search(r"const float KAN_XAI_REF_Z\[KAN_INPUT_DIM\]\s*=\s*\{(.*?)\};", text, flags=re.S)
    if not tag or not body:
        raise ValueError(f"{path} is not a KAN_XAI_REF_Z reference header")
    values = [np.float32(float(v)) for v in re.findall(NUMBER + r"(?=f)", body.group(1))]
    return tag.group(1), np.array(values, dtype=np.float32)
