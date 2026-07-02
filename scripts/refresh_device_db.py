"""
scripts/refresh_device_db.py

Downloads the smartphone specifications dataset from Kaggle and normalizes it
into data/devices.csv for use by hardware_research.py.

Prerequisites (one-time):
    1. Create a Kaggle account and go to kaggle.com/settings -> API -> Create New Token
    2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json
    3. pip install kaggle

Usage:
    python scripts/refresh_device_db.py
"""
import io
import os
import re
import shutil
import zipfile

import pandas as pd

KAGGLE_DATASET = "sady36/mobile-phones-specs"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "devices.csv")
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "_kaggle_tmp")


def _download() -> pd.DataFrame:
    import subprocess
    import shutil as _shutil

    kaggle_bin = _shutil.which("kaggle")
    if kaggle_bin is None:
        raise SystemExit("kaggle CLI not found. Run: pip install kaggle")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} ...")
    result = subprocess.run(
        [kaggle_bin, "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", DOWNLOAD_DIR, "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"kaggle download failed (exit {result.returncode}).")

    # Find the CSV — dataset may have one or several files
    csvs = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {DOWNLOAD_DIR} after download.")

    # Prefer the largest CSV (usually the full dataset)
    csvs.sort(key=lambda f: os.path.getsize(os.path.join(DOWNLOAD_DIR, f)), reverse=True)
    chosen = os.path.join(DOWNLOAD_DIR, csvs[0])
    print(f"Loading {chosen} ...")
    return pd.read_csv(chosen, low_memory=False)


def _parse_mb(val) -> int | None:
    """Parse strings like '6GB', '6 GB', '128GB', '256 GB' -> int MB."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*GB", s)
    if m:
        return int(float(m.group(1)) * 1024)
    m = re.search(r"(\d+(?:\.\d+)?)\s*MB", s)
    if m:
        return int(float(m.group(1)))
    m = re.search(r"^(\d+)$", s)
    if m:
        # bare number — assume GB if >= 1, MB otherwise
        n = float(m.group(1))
        return int(n * 1024) if n <= 512 else int(n)
    return None


def _parse_gib(val) -> int | None:
    """Parse strings like '3 GiB RAM', '6 GiB RAM' -> int MB."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    m = re.search(r"(\d+(?:\.\d+)?)\s*GIB", s)
    if m:
        return int(float(m.group(1)) * 1024)
    # fall through to generic parser
    return _parse_mb(val)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # sady36 (PhoneDB) column layout:
    #   Brand, Model            -> device name (concatenated)
    #   RAM Capacity (converted) -> RAM as "N GiB RAM"
    #   Memory Capacity          -> storage in GB (numeric)
    #   CPU                      -> chipset string

    cols = df.columns.tolist()
    print(f"  Columns: {cols[:15]} ...")

    # Build device name from Brand + Model
    if "Brand" in cols and "Model" in cols:
        device = (df["Brand"].fillna("").str.strip()
                  + " "
                  + df["Model"].fillna("").str.strip()).str.strip()
    elif "Model" in cols:
        device = df["Model"].astype(str).str.strip()
    else:
        raise ValueError(f"Cannot find device name columns. Got: {cols}")

    # RAM
    ram_col = "RAM Capacity (converted)" if "RAM Capacity (converted)" in cols else None
    if ram_col is None:
        for c in cols:
            if "ram" in c.lower():
                ram_col = c
                break

    # Storage — "Memory Capacity" is in GB as a float in PhoneDB
    storage_col = "Memory Capacity" if "Memory Capacity" in cols else None
    if storage_col is None:
        for c in cols:
            if "memory" in c.lower() or "storage" in c.lower():
                storage_col = c
                break

    # Chipset — "CPU" column has the full chipset string
    chipset_col = "CPU" if "CPU" in cols else None

    print(f"  name=Brand+Model  ram={ram_col!r}  storage={storage_col!r}  chipset={chipset_col!r}")

    out = pd.DataFrame()
    out["device"] = device

    out["ram_mb"] = df[ram_col].apply(_parse_gib) if ram_col else None

    # Memory Capacity in PhoneDB is a numeric GB value (e.g. 32.0, 128.0)
    if storage_col:
        def _storage_to_mb(v):
            try:
                return int(float(v) * 1024)
            except (TypeError, ValueError):
                return _parse_mb(v)
        out["storage_mb"] = df[storage_col].apply(_storage_to_mb)
    else:
        out["storage_mb"] = None

    out["chipset"] = df[chipset_col].astype(str).str.strip() if chipset_col else ""

    # Drop rows where we have neither RAM nor storage
    out = out.dropna(subset=["ram_mb", "storage_mb"], how="all")
    out = out.reset_index(drop=True)

    print(f"  {len(out)} usable rows after normalization.")
    return out


def main():
    df_raw = _download()
    df = _normalize(df_raw)

    out_path = os.path.abspath(OUT_PATH)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} devices to {out_path}")

    # Clean up temp download dir
    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
