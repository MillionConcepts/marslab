from pathlib import Path

import pandas as pd


def parse_m3_fn(fn):
    metadata = {}
    if "m3g" in fn.lower():
        metadata["mode"] = "global"
    elif "m3t" in fn.lower():
        metadata["mode"] = "target"
    if "l0" in fn.lower():
        metadata["level"] = "l0"
    elif "l1b" in fn.lower():
        metadata["level"] = "l1b"
    elif "l2" in fn.lower():
        metadata["level"] = "l2"
    return metadata


DATA_PATH = Path(Path(__file__).parent, "data")

TARGET_WAVELENGTHS = pd.read_csv(Path(DATA_PATH, "m3_target_wavelengths.csv"))
GLOBAL_WAVELENGTHS = pd.read_csv(Path(DATA_PATH, "m3_global_wavelengths.csv"))
