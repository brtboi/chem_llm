#!/usr/bin/env python3
"""Keep only rows 11 and 21 of expBandStruct_0.par and kpoints_0.par,
overwriting each file with the 2-row version."""

import os

ROWS = (11, 21)  # 1-indexed rows to keep
FILES = ("expBandStruct_0.par", "kpoints_0.par")

here = os.path.dirname(os.path.abspath(__file__))

for fname in FILES:
    path = os.path.join(here, fname)
    with open(path) as f:
        lines = f.readlines()

    selected = [lines[i - 1] for i in ROWS]

    with open(path, "w") as f:
        f.writelines(selected)

    print(f"{fname}: kept rows {ROWS} -> {len(selected)} rows")
