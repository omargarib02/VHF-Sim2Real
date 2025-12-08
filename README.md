# VHF-Sim2Real: Generative Noise Injection for Air Traffic Control ASR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AIAA SciTech 2026](https://img.shields.io/badge/Accepted-AIAA%20SciTech%202026-blue)](https://www.aiaa.org/SciTech)

**Official PyTorch implementation of the paper:** *"SimuGAN-Whisper-ATC: Generative Noise Injection for Improved Automatic Speech Recognition in Air Traffic Control"* Accepted at **AIAA SciTech Forum 2026**.

**Authors:** Omar Garib, Mohamed Ghanem, Olivia J. Pinon Fischer, Dimitri N. Mavris  
**Affiliation:** Aerospace Systems Design Laboratory (ASDL), Georgia Institute of Technology

---

## 📖 Abstract

Automatic Speech Recognition (ASR) in Air Traffic Control (ATC) faces a severe data scarcity problem: while clean simulation data is abundant, labeled real-world data with realistic Very High Frequency (VHF) noise is rare.

**SimuGAN-Whisper-ATC** bridges this gap using a **Spectrogram-Domain Generative Adversarial Network (SimuGAN)**. We learn realistic VHF noise patterns (squelch, static, channel distortion) from unlabeled real-world audio (TartanAviation) and inject them into clean simulated speech (ATCOSIM).

When fine-tuned on this augmented data, **OpenAI Whisper-Large-v2** achieves a **Word Error Rate (WER) of 3.58%** on the real-world ATCO2 benchmark, representing a **75% relative improvement** over baselines trained without generative augmentation.

![SimuGAN Architecture Diagram](./assets/simugan_architecture.png)

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/SimuGAN-Whisper-ATC.git
cd SimuGAN-Whisper-ATC
```

### 2. Environment Setup

We recommend using a Conda environment with Python 3.10+.

```bash
conda create -n atc-asr python=3.10
conda activate atc-asr
pip install -r requirements.txt
```

**Key Dependencies:**

- `torch` (2.0+)
- `transformers` (Hugging Face)
- `webrtcvad` (Voice Activity Detection)
- `wandb` (Experiment Tracking)

-----

## 🚀 Reproduction Pipeline

The pipeline consists of 5 sequential steps, orchestrated by the `scripts/` directory.

### Step 1: Download Data

Downloads ATCOSIM (Clean), ATCO2 (Real/Labeled), and TartanAviation (Real/Unlabeled).

```bash
python scripts/01_download_data.py --data_root ./data
```

### Step 2: Data Preparation

Cleans TartanAviation using VAD to isolate noise profiles and extracts ATCO2 manifests.

```bash
python scripts/02_prepare_datasets.py --data_root ./data
```

### Step 3: Train SimuGAN

Trains the Generator ($G$) to map *Clean Spectrograms* → *Noisy Spectrograms* using unpaired data.

```bash
python scripts/03_train_simugan.py \
    --data_root ./data \
    --run_name "simugan_v1" \
    --epochs 15
```

*Outputs checkpoints to `checkpoints/simugan_v1/`.*

### Step 4: Fine-Tune Whisper

Fine-tunes Whisper using a **2:1 ratio** of Real (ATCO2) to Synthetic (SimuGAN-Augmented) data. Noise is injected on-the-fly during training.

```bash
python scripts/04_finetune_whisper.py \
    --gan_ckpt checkpoints/simugan_v1/G_final.pth \
    --output_dir checkpoints/whisper_finetune
```

### Step 5: Evaluation

Evaluates the fine-tuned model on the ATCO2 Dev set using ATC-specific text normalization (NATO alphabet expansion, callsign formatting).

```bash
python scripts/05_evaluate.py \
    --model_path checkpoints/whisper_finetune/final_model
```

**Expected Result:** WER ≈ 3.58%

-----

## 📂 Repository Structure

```text
SimuGAN-Whisper-ATC/
├── scripts/                # Driver scripts for the pipeline
│   ├── 01_download_data.py
│   ├── 02_prepare_datasets.py
│   ├── 03_train_simugan.py
│   ├── 04_finetune_whisper.py
│   └── 05_evaluate.py
├── src/                    # Core library
│   ├── data/               # VAD cleaning, Transforms, Dataset Loaders
│   ├── models/             # PyTorch definitions for Generator/Discriminator
│   └── utils/              # Metrics (LSD, MR-STFT) and Normalizer
├── requirements.txt
└── README.md
```

-----

## 📊 Results & Metrics

| Model                 | Training Data                     | WER (ATCO2 Dev) |
|----------------------|-----------------------------------|-----------------|
| Whisper (Zero-shot)  | Pre-trained                       | 29.05%          |
| Whisper Baseline     | ATCOSIM (Clean) + ATCO2           | 14.66%          |
| **SimuGAN-Whisper-ATC** | **SimuGAN-Augmented + ATCO2**   | **3.58%**       |

-----

## 📜 Citation

If you use this code or dataset strategy, please cite our AIAA SciTech 2026 paper:

```bibtex
@inproceedings{garib2026simugan,
  title={SimuGAN-Whisper-ATC: Generative Noise Injection for Improved Automatic Speech Recognition in Air Traffic Control},
  author={Garib, Omar and Ghanem, Mohamed and Pinon Fischer, Olivia J. and Mavris, Dimitri N.},
  booktitle={AIAA SciTech 2026 Forum},
  year={2026}
}
```

-----

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
