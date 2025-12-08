#!/usr/bin/env python3
"""
05_evaluate.py
--------------
Evaluates a fine-tuned Whisper model on the ATCO2 Dev set.
Reports WER (Word Error Rate) using ATC-specific normalization.

Usage:
  python scripts/05_evaluate.py --model_path checkpoints/whisper_finetune/final_model
"""

import argparse
import sys
import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.append(str(Path(__file__).parents[1]))
from src.data.loaders import ATCO2JsonlDataset
from src.utils.normalizer import filterAndNormalize

def parse_args():
    default_root = Path(__file__).parents[1] / "data"
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=default_root)
    p.add_argument("--model_path", type=Path, required=True, help="Path to fine-tuned model")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Eval] Loading model from {args.model_path}...")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()
    
    print("[Eval] Loading ATCO2 Dev set...")
    ds = ATCO2JsonlDataset(
        jsonl_path=args.data_root / "atco2_prepared" / "dev.jsonl",
        processor=processor
    )
    
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
    wer_metric = evaluate.load("wer")
    
    refs = []
    preds = []
    
    print("[Eval] Running Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"]
            
            # Generate
            generated_ids = model.generate(input_features, language="en")
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode Labels (handle padding -100)
            labels[labels == -100] = processor.tokenizer.pad_token_id
            references = processor.batch_decode(labels, skip_special_tokens=True)
            
            # Normalize
            preds.extend([filterAndNormalize(t) for t in transcriptions])
            refs.extend([filterAndNormalize(r) for r in references])
            
    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"\n-----------------------------")
    print(f"Final WER: {wer*100:.2f}%")
    print(f"-----------------------------")

if __name__ == "__main__":
    main()