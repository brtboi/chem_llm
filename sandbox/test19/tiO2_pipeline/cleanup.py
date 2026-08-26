import shutil
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALC_DIR = os.path.join(SCRIPT_DIR, "calculations")

if os.path.exists(CALC_DIR):
    shutil.rmtree(CALC_DIR)
    print(f"Deleted {CALC_DIR}")
else:
    print(f"{CALC_DIR} does not exist")
