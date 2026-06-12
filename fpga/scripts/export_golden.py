#!/usr/bin/env python3
"""Export golden vectors from Python to FPGA testbench directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "l2l_dct" / "python"))

import numpy as np
from dct.loeffler_8pt import loeffler_dct_8_float

GOLDEN = ROOT / "l2l_dct" / "tests" / "golden"
GOLDEN.mkdir(parents=True, exist_ok=True)

x = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float64)
y = loeffler_dct_8_float(x)
with open(GOLDEN / "dct_golden.txt", "w") as f:
    for v in y:
        f.write(f"{v:.17g}\n")
print(f"Exported to {GOLDEN / 'dct_golden.txt'}")
