import torch
import torch.nn as nn

class GeneratorUNet(nn.Module):
    """
    U-Net Generator for SimuGAN-Whisper-ATC.
    
    Architecture:
      - Encoder: 3 layers of Conv2d + InstanceNorm + LeakyReLU
      - Bottleneck: 2 layers of Conv2d + InstanceNorm + ReLU
      - Decoder: 3 layers of ConvTranspose2d + InstanceNorm + ReLU
      - Output: ConvTranspose2d + Softplus (forces non-negative magnitude spectrograms)
    
    Input:
      x (torch.Tensor): Clean magnitude spectrogram [Batch, 1, Freq, Time]
    
    Output:
      out (torch.Tensor): Synthetic noisy magnitude spectrogram [Batch, 1, Freq, Time]
    """
    def __init__(self, base_ch: int = 64):
        super().__init__()

        def conv_block(in_ch, out_ch, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, True),
            )

        def deconv_block(in_ch, out_ch, k, s, p):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, k, s, p),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(True),
            )

        # Encoder (Downsampling)
        # Input: [B, 1, F, T]
        self.enc1 = conv_block(1, base_ch, (3, 4), (1, 2), (1, 1))
        self.enc2 = conv_block(base_ch, base_ch * 2, (3, 4), (1, 2), (1, 1))
        self.enc3 = conv_block(base_ch * 2, base_ch * 4, (3, 4), (1, 2), (1, 1))

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(True),
        )

        # Decoder (Upsampling) with Skip Connections
        self.dec3 = deconv_block(base_ch * 4, base_ch * 2, (3, 4), (1, 2), (1, 1))
        self.dec2 = deconv_block(base_ch * 4, base_ch, (3, 4), (1, 2), (1, 1))
        
        # Final Output Layer
        # We use Softplus to ensure strictly positive spectrogram magnitudes
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2 + 1, 1, (3, 4), (1, 2), (1, 1)),
            nn.Softplus(beta=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        m = self.mid(e3)

        # Decoder + Skip Connections
        d3 = self.dec3(m)
        d3 = torch.cat([d3, e2], 1)

        d2 = self.dec2(d3)

        # Dynamic Cropping: Ensure all tensors match the smallest Time dimension
        # This handles edge cases where downsampling/upsampling introduces rounding diffs.
        minT = min(d2.size(-1), e1.size(-1), x.size(-1))
        
        if d2.size(-1) != minT: d2 = d2[..., :minT]
        if e1.size(-1) != minT: e1 = e1[..., :minT]
        if x.size(-1)  != minT: x  = x[...,  :minT]

        # Final Skip Connection (Identity)
        # We concatenate the original input 'x' to preserve speech structure
        d1_in = torch.cat([d2, e1, x], 1)
        
        return self.dec1(d1_in)