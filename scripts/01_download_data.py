#!/usr/bin/env python3
"""
01_download_data.py
-------------------
Downloads the required datasets for SimuGAN-Whisper-ATC:
  1. ATCOSIM (Clean speech, simulation)
  2. ATCO2 (Real noisy speech, labeled)
  3. TartanAviation (Real noisy speech, unlabeled)

Usage:
  python scripts/01_download_data.py --data_root ./data
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

# Constants
HF_ATCOSIM = "jlvdoorn/atcosim"
HF_ATCO2 = "jlvdoorn/atco2-asr"
TARTAN_REPO_URL = "https://github.com/castacks/TartanAviation.git"

def parse_args():
    # Default: RepoRoot/data
    default_root = Path(__file__).parents[1] / "data"
    
    p = argparse.ArgumentParser(description="Download ATC Datasets")
    p.add_argument("--data_root", type=Path, default=default_root,
                   help="Root directory for storing datasets.")
    p.add_argument("--skip_tartan", action="store_true", 
                   help="Skip downloading TartanAviation (it's large).")
    return p.parse_args()

def install_minio_if_needed():
    """TartanAviation script requires minio."""
    try:
        import minio
    except ImportError:
        print("[Installer] Installing 'minio' for TartanAviation download...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "minio"])

def download_hf_dataset(repo_id: str, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"[Skip] {repo_id} appears to be downloaded at {dest}")
        return
        
    print(f"[Download] Fetching {repo_id} -> {dest}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"[Done] {repo_id} ready.\n")

def download_tartan(data_root: Path):
    tartan_dest = data_root / "tartan_raw"
    tartan_dest.mkdir(parents=True, exist_ok=True)
    
    # Clone the tools repo to access their download script
    tools_dir = data_root / "tartan_tools"
    if not tools_dir.exists():
        print("[Git] Cloning TartanAviation repo for download tools...")
        subprocess.check_call(["git", "clone", "--depth", "1", TARTAN_REPO_URL, str(tools_dir)])
    
    # Locate download script
    # It moves around in their repo, so we check common spots
    possible_scripts = [
        tools_dir / "download.py",
        tools_dir / "audio" / "download_audio.py",
        tools_dir / "speech" / "download.py"
    ]
    script_path = next((p for p in possible_scripts if p.exists()), None)
    
    if not script_path:
        print("[Error] Could not find 'download.py' in TartanAviation repo.")
        return

    install_minio_if_needed()
    
    print("[Tartan] Launching download script (Subset: Date_Range 2021-05 to 2021-07)...")
    # Using the arguments from your original script history
    cmd = [
        sys.executable, str(script_path),
        "--option", "Date_Range",
        "--start_date", "2021-05",
        "--end_date", "2021-07",
        "--location", "Both",
        "--save_dir", str(tartan_dest)
    ]
    
    try:
        subprocess.check_call(cmd)
        print("[Done] TartanAviation downloaded.")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Tartan download failed: {e}")

def main():
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    print(f"--- SimuGAN Data Downloader ---\nTarget: {args.data_root}\n")

    # 1. ATCOSIM (Clean)
    download_hf_dataset(HF_ATCOSIM, args.data_root / "atcosim_raw")

    # 2. ATCO2 (Real Noisy + Labels)
    download_hf_dataset(HF_ATCO2, args.data_root / "atco2_raw")

    # 3. TartanAviation (Real Noisy Backgrounds)
    if not args.skip_tartan:
        download_tartan(args.data_root)

if __name__ == "__main__":
    main()