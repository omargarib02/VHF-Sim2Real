"""
Acoustic Evaluation Metrics for SimuGAN.
Includes LSD, MR-STFT, SI-SDR, and embedding-based distances.
"""
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from scipy.stats import entropy
from src.data.transforms import N_FFT, HOP_LENGTH, WIN_LENGTH, AudioTransforms

# Optional Dependencies
try:
    from pystoi import stoi as _stoi_fn
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

try:
    from frechetaudio import FrechetAudioDistance
    HAS_FAD = True
except ImportError:
    HAS_FAD = False

# ---------------------------------------------------------------------------
# Spectral Metrics
# ---------------------------------------------------------------------------

def lsd(ref_mag: torch.Tensor, est_mag: torch.Tensor) -> float:
    """Log-Spectral Distance (dB). Lower is better."""
    # Ensure same size
    minF = min(ref_mag.size(-2), est_mag.size(-2))
    minT = min(ref_mag.size(-1), est_mag.size(-1))
    
    ref = ref_mag[..., :minF, :minT]
    est = est_mag[..., :minF, :minT]
    
    ref_db = 20 * torch.log10(torch.clamp(ref, min=1e-5))
    est_db = 20 * torch.log10(torch.clamp(est, min=1e-5))
    
    return float((ref_db - est_db).abs().mean().item())

def mr_stft_loss(ref_wav: torch.Tensor, est_wav: torch.Tensor, device='cpu') -> float:
    """Multi-Resolution STFT Loss. Lower is better."""
    # Config: (n_fft, hop, win)
    configs = [(256, 64, 256), (128, 32, 128), (64, 16, 64)]
    
    tot_loss = 0.0
    ref_wav = ref_wav.to(device)
    est_wav = est_wav.to(device)
    
    for n_fft, hop, win in configs:
        window = torch.hann_window(win, device=device)
        
        # Helper inner function
        def get_mag(w):
            return torch.stft(w, n_fft, hop, win, window=window, 
                            return_complex=True, center=True).abs()
        
        R = get_mag(ref_wav)
        E = get_mag(est_wav)
        
        minF = min(R.size(-2), E.size(-2))
        minT = min(R.size(-1), E.size(-1))
        
        tot_loss += (R[..., :minF, :minT] - E[..., :minF, :minT]).abs().mean()
        
    return float(tot_loss / len(configs))

def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB). Higher is better."""
    # Remove mean
    ref = ref - ref.mean()
    est = est - est.mean()
    
    # Projection
    alpha = torch.dot(est.view(-1), ref.view(-1)) / (torch.dot(ref.view(-1), ref.view(-1)) + 1e-8)
    s_target = alpha * ref
    e_noise = est - s_target
    
    si_sdr_val = 10 * torch.log10(
        (s_target.pow(2).sum() + 1e-8) / (e_noise.pow(2).sum() + 1e-8)
    )
    return float(si_sdr_val.item())

# ---------------------------------------------------------------------------
# Perceptual & Embedding Metrics
# ---------------------------------------------------------------------------

_W2V_MODEL = None
_SPK_MODEL = None

def get_w2v_model():
    global _W2V_MODEL
    if _W2V_MODEL is None:
        _W2V_MODEL = torchaudio.pipelines.WAV2VEC2_BASE.get_model().eval()
    return _W2V_MODEL

def get_spk_model():
    """Tries to load SUPERB X-Vector, falls back to W2V."""
    global _SPK_MODEL
    if _SPK_MODEL is None:
        try:
            _SPK_MODEL = torchaudio.pipelines.SUPERB_XVECTOR.get_model().eval()
        except Exception:
            print("[Metrics] Warning: SUPERB XVector not found. Falling back to Wav2Vec2.")
            _SPK_MODEL = get_w2v_model()
    return _SPK_MODEL

@torch.inference_mode()
def embedding_distance(ref_wav: torch.Tensor, est_wav: torch.Tensor) -> float:
    """Cosine distance between Wav2Vec2 embeddings. Lower is better."""
    model = get_w2v_model().to(ref_wav.device)
    
    # Add batch dim if needed
    if ref_wav.dim() == 1: ref_wav = ref_wav.unsqueeze(0)
    if est_wav.dim() == 1: est_wav = est_wav.unsqueeze(0)
    
    # Extract
    r_emb = model(ref_wav)[0].mean(dim=1)
    e_emb = model(est_wav)[0].mean(dim=1)
    
    return float(1.0 - torch.nn.functional.cosine_similarity(r_emb, e_emb).item())

@torch.inference_mode()
def speaker_distance(ref_wav: torch.Tensor, est_wav: torch.Tensor) -> float:
    """Cosine distance between Speaker Embeddings. Lower is better."""
    model = get_spk_model().to(ref_wav.device)
    
    if ref_wav.dim() == 1: ref_wav = ref_wav.unsqueeze(0)
    if est_wav.dim() == 1: est_wav = est_wav.unsqueeze(0)
    
    # Extract (handle pipeline differences)
    try:
        r_emb = model(ref_wav)
        e_emb = model(est_wav)
        if isinstance(r_emb, torch.Tensor):
            # For XVector, output is typically [batch, 1, 512] or similar
            if r_emb.dim() > 2: r_emb = r_emb.mean(dim=1)
            if e_emb.dim() > 2: e_emb = e_emb.mean(dim=1)
    except Exception:
        # Fallback for W2V
        r_emb = model(ref_wav)[0].mean(dim=1)
        e_emb = model(est_wav)[0].mean(dim=1)

    return float(1.0 - torch.nn.functional.cosine_similarity(r_emb, e_emb).item())

def stoi_score(clean_np: np.ndarray, est_np: np.ndarray, sr=16000) -> float:
    if not HAS_STOI:
        return float('nan')
    return _stoi_fn(clean_np, est_np, sr, extended=False)

def compute_fad(real_files: list, fake_files: list) -> float:
    if not HAS_FAD:
        print("[Metrics] FrechetAudioDistance not installed.")
        return float('nan')
        
    fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_cuda=torch.cuda.is_available())
    return fad.calculate(fake_files, real_files)