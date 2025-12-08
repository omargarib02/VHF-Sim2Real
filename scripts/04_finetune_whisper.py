#!/usr/bin/env python3
"""
04_finetune_whisper.py
----------------------
Fine-tunes OpenAI Whisper (Large-v2) for ATC using a mix of:
  1. Real Noisy Data (ATCO2) - 67% of batch
  2. Synthetic Noisy Data (ATCOSIM + SimuGAN) - 33% of batch

Key Features:
  - Loads pre-trained SimuGAN Generator to inject noise on-the-fly.
  - Uses WeightedRandomSampler to enforce the 2:1 data ratio.
  - Evaluates WER on ATCO2 Dev set.

Usage:
  python scripts/04_finetune_whisper.py --gan_ckpt checkpoints/simugan_base/G_final.pth
"""

import argparse
import sys
import datetime
import torch
import evaluate
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# Add repo root to path for src imports
sys.path.append(str(Path(__file__).parents[1]))

from src.data.loaders import ATCOSIMParquetGAN, ATCO2JsonlDataset
from src.utils.normalizer import filterAndNormalize

# Constants
MODEL_NAME = "openai/whisper-large-v2"

def parse_args():
    default_root = Path(__file__).parents[1] / "data"
    p = argparse.ArgumentParser(description="Fine-tune Whisper with SimuGAN")
    
    # Paths
    p.add_argument("--data_root", type=Path, default=default_root)
    p.add_argument("--gan_ckpt", type=Path, required=True, help="Path to trained Generator checkpoint")
    p.add_argument("--output_dir", type=Path, default=Path("checkpoints/whisper_finetune"))
    p.add_argument("--run_name", type=str, default="whisper_simugan")
    
    # Hyperparams
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation", type=int, default=1)
    
    return p.parse_args()

class WeightedTrainer(Seq2SeqTrainer):
    """
    Custom Trainer to enforce 2:1 sampling ratio between Real (ATCO2) and Synthetic (ATCOSIM) data.
    """
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Identify sub-datasets from ConcatDataset
        # We assume dataset order: [ATCO2, ATCOSIM]
        ds_atco2 = self.train_dataset.datasets[0]
        ds_atcosim = self.train_dataset.datasets[1]
        
        count_atco2 = len(ds_atco2)
        count_atcosim = len(ds_atcosim)
        total_count = count_atco2 + count_atcosim
        
        # Calculate Weights for 2:1 Ratio
        # Target: P(ATCO2) = 0.66, P(ATCOSIM) = 0.33
        w_atco2 = (2/3) / count_atco2
        w_atcosim = (1/3) / count_atcosim
        
        # Assign weights to every sample
        weights = torch.zeros(total_count, dtype=torch.float)
        weights[:count_atco2] = w_atco2
        weights[count_atco2:] = w_atcosim
        
        print(f"[Trainer] Sampling Strategy: {count_atco2} Real, {count_atcosim} Synthetic.")
        print(f"[Trainer] Weights -> Real: {w_atco2:.2e}, Synth: {w_atcosim:.2e}")

        sampler = WeightedRandomSampler(weights, num_samples=total_count, replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )

def main():
    args = parse_args()
    
    # Setup Paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.run_name}_{timestamp}"
    
    # 1. Load Model & Processor
    print(f"[Model] Loading {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Optimization: Disable cache for training, enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # 2. Prepare Datasets
    # Main training device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # A. Real Data (ATCO2)
    print("[Data] Loading Real ATCO2...")
    ds_atco2 = ATCO2JsonlDataset(
        jsonl_path=args.data_root / "atco2_prepared" / "train.jsonl", 
        processor=processor
    )
    
    # B. Synthetic Data (ATCOSIM + SimuGAN)
    print(f"[Data] Loading ATCOSIM with Generator {args.gan_ckpt}...")
    
    # Explicitly use CPU for data loading generator to ensure multiprocessing safety
    loader_device = torch.device("cpu")
    
    ds_atcosim = ATCOSIMParquetGAN(
        parquet_dir=args.data_root / "atcosim_raw",
        processor=processor,
        generator_ckpt=args.gan_ckpt,
        device=loader_device
    )
    
    # Combine (Order matters for WeightedTrainer!)
    train_ds = ConcatDataset([ds_atco2, ds_atcosim])
    
    # C. Validation Data
    val_ds = ATCO2JsonlDataset(
        jsonl_path=args.data_root / "atco2_prepared" / "dev.jsonl",
        processor=processor
    )
    
    # 3. Metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # Replace -100 with pad_token_id to decode
        labels[labels == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Normalize text using our custom ATC normalizer
        pred_norm = [filterAndNormalize(s) for s in pred_str]
        label_norm = [filterAndNormalize(s) for s in label_str]
        
        wer = wer_metric.compute(predictions=pred_norm, references=label_norm)
        return {"wer": wer * 100}

    # 4. Data Collator
    def data_collator(features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Mask padding in labels so loss isn't computed there
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Create decoder_input_ids (shifted right) if model doesn't create them automatically
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False, 
        dataloader_num_workers=4,
        predict_with_generate=True,
        generation_max_length=225,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("[Train] Starting Fine-Tuning...")
    trainer.train()
    
    # Save Final
    trainer.save_model(str(run_dir / "final_model"))
    processor.save_pretrained(str(run_dir / "final_model"))
    print(f"[Done] Model saved to {run_dir / 'final_model'}")

if __name__ == "__main__":
    main()