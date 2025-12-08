#!/usr/bin/env python3
"""
03_train_simugan.py
-------------------
Trains the SimuGAN (Spectrogram-Domain GAN) to learn realistic noise patterns
from unpaired data (Clean ATCOSIM vs. Noisy Tartan/ATCO2).

Key Method:
  - Input: Clean Magnitude Spectrogram (Clean)
  - Output: Generated Noisy Magnitude Spectrogram (Fake)
  - Discriminator: Distinguishes Real Noisy vs. Fake Noisy
  - Losses: Adversarial + Feature Matching + Identity + MR-STFT

Usage:
  python scripts/03_train_simugan.py --data_root ./data --run_name experiment_v1
"""

import argparse
import csv
import os
import random
import sys
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import DataLoader

# Add repo root to path for src imports
sys.path.append(str(Path(__file__).parents[1]))

from src.models import GeneratorUNet, PatchDiscriminator
from src.data.loaders import (
    CleanATCOSIMParquet, 
    TartanAtco2NoisyWav, 
    UnpairedSTFTDataset, 
    collate_fn_gan
)
from src.data.transforms import AudioTransforms
from src.utils.metrics import mr_stft_loss

def parse_args():
    default_root = Path(__file__).parents[1] / "data"
    p = argparse.ArgumentParser(description="Train SimuGAN")
    
    # Paths
    p.add_argument("--data_root", type=Path, default=default_root)
    p.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run_name", type=str, default="simugan_base")
    
    # Training Hyperparams
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    
    # Loss Weights (from Paper Eq 2 & Table 2)
    p.add_argument("--lambda_fm", type=float, default=10.0, help="Feature Matching Weight")
    p.add_argument("--lambda_id", type=float, default=4.0, help="Identity Loss Weight")
    p.add_argument("--lambda_mr", type=float, default=1.0, help="Multi-Resolution STFT Weight")
    
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_sample_audio(G, clean_wav, epoch, save_dir, device):
    """Generates and saves a single audio sample for sanity checking."""
    G.eval()
    with torch.no_grad():
        mag, pha = AudioTransforms.stft_mag_phase(clean_wav.to(device))
        mag = AudioTransforms.pad_to_mult_of_8(mag)
        pha = AudioTransforms.pad_to_mult_of_8(pha)
        
        # Generate Fake
        fake_mag = G(mag.unsqueeze(0)).squeeze(0)
        
        # Reconstruct
        fake_wav = AudioTransforms.istft_from_mag_phase(fake_mag, pha)
        
    out_path = save_dir / f"epoch_{epoch:03d}_sample.wav"
    torchaudio.save(str(out_path), fake_wav.cpu(), 16000)
    G.train()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup Directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    
    # Logging
    csv_file = open(run_dir / "train_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "step", "D_loss", "G_adv", "G_fm", "G_id", "G_mr", "G_total"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SimuGAN Training ---\nDevice: {device}\nOutput: {run_dir}")

    # 1. Data Loading
    print("[Data] Initializing Datasets...")
    clean_ds = CleanATCOSIMParquet(args.data_root / "atcosim_raw")
    
    # Construct paths for noisy data
    noisy_paths = [
        str(args.data_root / "tartan_cleaned"),
        str(args.data_root / "atco2_prepared" / "audio") 
        # Note: We use prepared ATCO2 audio as noise source proxy or 
        # you can use specific noise-only folders if you ran extraction script 04
    ]
    # If explicit noise folders exist, use them preferrentially
    if (args.data_root / "atco2_noise").exists():
        noisy_paths.append(str(args.data_root / "atco2_noise"))

    noisy_ds = TartanAtco2NoisyWav(",".join(noisy_paths))
    
    dataset = UnpairedSTFTDataset(clean_ds, noisy_ds)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn_gan,
        drop_last=True
    )
    
    # 2. Models
    G = GeneratorUNet().to(device)
    D = PatchDiscriminator().to(device)
    
    # 3. Optimizers
    opt_G = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    
    sched_G = optim.lr_scheduler.StepLR(opt_G, step_size=5, gamma=0.5)
    sched_D = optim.lr_scheduler.StepLR(opt_D, step_size=5, gamma=0.5)
    
    # 4. Losses
    adv_criterion = nn.MSELoss() # Least Squares GAN (LSGAN) is stable
    l1_criterion = nn.L1Loss()
    
    # Training Loop
    print(f"[Train] Starting loop for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        for i, (c_mag, c_pha, n_mag, _, _) in enumerate(loader):
            c_mag = c_mag.to(device)
            n_mag = n_mag.to(device)
            
            # ==================================================================
            # Train Discriminator
            # ==================================================================
            opt_D.zero_grad()
            
            # Real
            real_logits, _ = D(n_mag)
            label_real = torch.full_like(real_logits, 0.9, device=device) # Label Smoothing
            errD_real = adv_criterion(real_logits, label_real)
            
            # Fake
            with torch.no_grad():
                fake_mag = G(c_mag)
            fake_logits, _ = D(fake_mag.detach())
            label_fake = torch.zeros_like(fake_logits, device=device)
            errD_fake = adv_criterion(fake_logits, label_fake)
            
            errD = 0.5 * (errD_real + errD_fake)
            errD.backward()
            opt_D.step()
            
            # ==================================================================
            # Train Generator
            # ==================================================================
            opt_G.zero_grad()
            
            # 1. Adversarial Loss (Fool D)
            # We re-run G to get the graph for backprop
            fake_mag_g = G(c_mag)
            pred_fake, pred_feats = D(fake_mag_g)
            label_real_g = torch.ones_like(pred_fake, device=device)
            loss_adv = adv_criterion(pred_fake, label_real_g)
            
            # 2. Feature Matching Loss
            # Compare D's internal features for Real vs Fake
            with torch.no_grad():
                _, target_feats = D(n_mag)
            
            loss_fm = 0.0
            for feat_f, feat_r in zip(pred_feats, target_feats):
                loss_fm += l1_criterion(feat_f, feat_r)
            loss_fm = loss_fm / len(pred_feats)
            
            # 3. Identity Loss (Don't destroy the speech)
            loss_id = l1_criterion(fake_mag_g, c_mag)
            
            # 4. Multi-Resolution STFT Loss (Spectral Consistency)
            # Reconstruct time-domain to calculate this
            # We assume c_pha is valid for reconstruction approx
            fake_wav = AudioTransforms.istft_from_mag_phase(fake_mag_g, c_pha.to(device))
            clean_wav = AudioTransforms.istft_from_mag_phase(c_mag, c_pha.to(device))
            loss_mr = mr_stft_loss(clean_wav, fake_wav, device=device)
            
            # Total G Loss
            loss_G = loss_adv + \
                     (args.lambda_fm * loss_fm) + \
                     (args.lambda_id * loss_id) + \
                     (args.lambda_mr * loss_mr)
                     
            loss_G.backward()
            opt_G.step()
            
            # Logging
            if i % 50 == 0:
                print(f"[Ep {epoch}/{args.epochs}][{i}] "
                      f"D: {errD.item():.4f} | G: {loss_G.item():.4f} "
                      f"(Adv: {loss_adv.item():.3f}, FM: {loss_fm.item():.3f}, ID: {loss_id.item():.3f})")
                
                csv_writer.writerow([
                    epoch, i, 
                    errD.item(), loss_adv.item(), loss_fm.item(), 
                    loss_id.item(), loss_mr, loss_G.item()
                ])
                csv_file.flush()

        # End of Epoch
        sched_G.step()
        sched_D.step()
        
        # Save Checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(G.state_dict(), run_dir / f"G_epoch{epoch}.pth")
            torch.save(D.state_dict(), run_dir / f"D_epoch{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch}")
            
        # Save Sample Audio
        sample_wav = clean_ds.clean_wavs[0] # Always check the same file
        save_sample_audio(G, sample_wav, epoch, run_dir / "samples", device)

    # Save Final
    torch.save(G.state_dict(), run_dir / "G_final.pth")
    print("Training Complete. Model saved.")
    csv_file.close()

if __name__ == "__main__":
    main()