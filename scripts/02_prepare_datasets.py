#!/usr/bin/env python3
"""
02_prepare_datasets.py
----------------------
Prepares raw downloaded data for training.

Actions:
  1. ATCO2: Extracts WAVs from Parquet shards and creates JSONL manifests (train/dev).
  2. TartanAviation: Applies VAD to segment long recordings into active speech/noise clips 
     (discarding pure silence) for GAN discriminator training.

Usage:
  python scripts/02_prepare_datasets.py --data_root ./data
"""

import argparse
import json
import random
import sys
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
from datasets import load_dataset

# Add repo root to path to import src
sys.path.append(str(Path(__file__).parents[1]))
from src.data.cleaning import VADProcessor

def parse_args():
    default_root = Path(__file__).parents[1] / "data"
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=default_root)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vad_aggressiveness", type=int, default=2)
    return p.parse_args()

def process_atco2(data_root: Path, seed: int):
    """
    Extracts ATCO2 parquet -> WAV files + JSONL manifests.
    """
    raw_dir = data_root / "atco2_raw"
    out_dir = data_root / "atco2_prepared"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[ATCO2] Loading Parquet from {raw_dir}...")
    try:
        # Load streaming to avoid massive RAM usage
        ds = load_dataset("parquet", data_files=str(raw_dir / "**/*.parquet"), split="train", streaming=True)
    except Exception as e:
        print(f"[Error] Could not load ATCO2 parquet: {e}")
        return

    train_manifest = open(out_dir / "train.jsonl", "w", encoding="utf-8")
    dev_manifest = open(out_dir / "dev.jsonl", "w", encoding="utf-8")
    
    rng = random.Random(seed)
    count = 0
    
    print("[ATCO2] Extracting WAVs and splitting manifests...")
    for row in tqdm(ds, desc="ATCO2 Processing"):
        # Identify text column
        text = row.get("text") or row.get("sentence") or row.get("transcript")
        if not text:
            continue
            
        # Save WAV
        filename = f"atco2_{count:06d}.wav"
        filepath = audio_dir / filename
        
        # Audio is usually in 'audio' dict with 'array' and 'sampling_rate'
        audio_array = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]
        
        sf.write(str(filepath), audio_array, sr, subtype="PCM_16")
        
        # Create Metadata
        meta = {
            "audio_filepath": str(filepath.resolve()),
            "text": text,
            "duration": len(audio_array) / sr
        }
        
        # 80/20 Split
        if rng.random() < 0.8:
            train_manifest.write(json.dumps(meta) + "\n")
        else:
            dev_manifest.write(json.dumps(meta) + "\n")
            
        count += 1

    train_manifest.close()
    dev_manifest.close()
    print(f"[ATCO2] Done. Processed {count} files.")


def process_tartan(data_root: Path, vad_level: int):
    """
    Cleans TartanAviation: VAD Segmenting -> Keep active segments.
    """
    in_dir = data_root / "tartan_raw"
    out_dir = data_root / "tartan_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list(in_dir.rglob("*.wav"))
    if not wavs:
        print(f"[Tartan] No WAV files found in {in_dir}. Skipping.")
        return

    print(f"\n[Tartan] Found {len(wavs)} raw files. Starting VAD cleaning...")
    
    vad = VADProcessor(vad_mode=vad_level, frame_ms=30)
    total_segments = 0

    for wav_path in tqdm(wavs, desc="Tartan Cleaning"):
        try:
            # Load and conform to 16kHz Mono
            pcm, sr = vad.load_audio(wav_path)
            
            # Get active speech/noise segments (non-silence)
            timestamps = vad.get_speech_timestamps(pcm)
            
            # Save segments
            rel_path = wav_path.relative_to(in_dir)
            stem = rel_path.stem
            
            for i, (start, end) in enumerate(timestamps):
                # Filter tiny clips (< 1s)
                if (end - start) < 16000: 
                    continue
                    
                segment = pcm[start:end]
                
                # Save
                seg_name = f"{stem}_seg{i:03d}.wav"
                seg_out = out_dir / seg_name
                vad.save_wav(seg_out, segment, sr)
                total_segments += 1
                
        except Exception as e:
            print(f"Failed to process {wav_path.name}: {e}")

    print(f"[Tartan] Done. Created {total_segments} active segments in {out_dir}")

def main():
    args = parse_args()
    
    # 1. Prepare ATCO2 (Train/Dev Split)
    process_atco2(args.data_root, args.seed)
    
    # 2. Prepare Tartan (Cleaning for GAN Discriminator)
    process_tartan(args.data_root, args.vad_aggressiveness)

if __name__ == "__main__":
    main()