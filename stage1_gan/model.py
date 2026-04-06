"""
model.py — StyleGAN-inspired Generator + PatchGAN Discriminator.

Generator pipeline:
  z (noise) → MappingNetwork → w (style) → SynthesisNetwork → Image

Discriminator pipeline:
  Image → Progressive Conv blocks → Patch-level real/fake scores

Both networks are designed to fit in 8-12 GB VRAM at 256×256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  SHARED BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════

class PixelNorm(nn.Module):
    """Pixel-wise feature vector normalization (ProGAN)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True) + 1e-8).sqrt()


class EqualLinear(nn.Module):
    """
    Fully-connected layer with equalized learning rate.
    Scales weights at runtime instead of at init, keeping all
    layers in the same gradient magnitude range (StyleGAN trick).
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias   = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale  = (2 / in_dim) ** 0.5   # He init factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    Injects the style vector w into spatial features:
        AdaIN(x, w) = γ(w) · norm(x) + β(w)
    """
    def __init__(self, channels: int, w_dim: int):
        super().__init__()
        self.norm  = nn.InstanceNorm2d(channels)
        # One linear produces both scale (γ) and bias (β)
        self.style = EqualLinear(w_dim, channels * 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # w → [B, 2C]
        style = self.style(w)
        gamma, beta = style.chunk(2, dim=1)          # [B, C] each
        gamma = gamma[:, :, None, None]               # [B, C, 1, 1]
        beta  = beta [:, :, None, None]
        return (1 + gamma) * self.norm(x) + beta


class NoiseInjection(nn.Module):
    """
    Per-pixel learnable noise injection.
    Adds stochastic variation to texture — critical for lace details.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


# ══════════════════════════════════════════════════════════════════
#  GENERATOR
# ══════════════════════════════════════════════════════════════════

class MappingNetwork(nn.Module):
    """
    Maps latent z → disentangled style w.
    8 EqualLinear layers with LeakyReLU + pixel normalization.
    """
    def __init__(self, z_dim: int = 512, w_dim: int = 512, depth: int = 8):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(depth):
            layers += [
                EqualLinear(z_dim if i == 0 else w_dim, w_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SynthBlock(nn.Module):
    """
    Single synthesis block:
      (optional upsample) → Conv3×3 → NoiseInjection → AdaIN → LReLU
    """
    def __init__(self, in_ch: int, out_ch: int, w_dim: int,
                 upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.conv   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise  = NoiseInjection()
        self.adain  = AdaIN(out_ch, w_dim)
        self.act    = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2,
                              mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.noise(x)
        x = self.adain(x, w)
        return self.act(x)


class Generator(nn.Module):
    """
    StyleGAN-inspired Generator: 4×4 → 256×256 over 7 synthesis blocks.

    Channel schedule (ngf=64):
        4   → 8   → 16  → 32  → 64  → 128 → 256
        512   512   256   256   128    64    32

    Args:
        z_dim      : Dimension of input noise z.
        w_dim      : Dimension of style vector w.
        ngf        : Base channel multiplier.
        image_size : Output image size (must be 2^k, default 256).
    """
    def __init__(self, z_dim: int = 512, w_dim: int = 512,
                 ngf: int = 64, image_size: int = 256):
        super().__init__()
        self.z_dim = z_dim

        # Mapping z → w
        self.mapping = MappingNetwork(z_dim, w_dim)

        # Learnable 4×4 constant input
        self.const = nn.Parameter(torch.randn(1, ngf * 8, 4, 4))

        # Channel sizes at each resolution
        ch = {
            4:   ngf * 8,   # 512
            8:   ngf * 8,   # 512
            16:  ngf * 4,   # 256
            32:  ngf * 4,   # 256
            64:  ngf * 2,   # 128
            128: ngf,       # 64
            256: ngf // 2,  # 32
        }

        self.blocks = nn.ModuleList([
            SynthBlock(ch[4],   ch[8],   w_dim, upsample=False),   # 4→4
            SynthBlock(ch[8],   ch[16],  w_dim, upsample=True),    # 4→8
            SynthBlock(ch[16],  ch[32],  w_dim, upsample=True),    # 8→16
            SynthBlock(ch[32],  ch[64],  w_dim, upsample=True),    # 16→32
            SynthBlock(ch[64],  ch[128], w_dim, upsample=True),    # 32→64
            SynthBlock(ch[128], ch[256], w_dim, upsample=True),    # 64→128
            SynthBlock(ch[256], ch[256], w_dim, upsample=True),    # 128→256
        ])

        # 1×1 conv to RGB, output in [-1, 1]
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch[256], 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Random latent [B, z_dim]
        Returns:
            imgs: Generated images [B, 3, 256, 256]
        """
        w = self.mapping(z)                              # [B, w_dim]
        x = self.const.expand(z.size(0), -1, -1, -1)   # [B, 512, 4, 4]
        for block in self.blocks:
            x = block(x, w)
        return self.to_rgb(x)                           # [B, 3, 256, 256]


# ══════════════════════════════════════════════════════════════════
#  DISCRIMINATOR
# ══════════════════════════════════════════════════════════════════

class DiscBlock(nn.Module):
    """Conv → BN → LReLU with stride-2 downsampling."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator: classifies overlapping image patches as
    real or fake. Naturally sensitive to local texture — ideal for lace.

    Input:  [B, 3, 256, 256]
    Output: [B, 1, H', W'] patch predictions
    """
    def __init__(self, ndf: int = 64):
        super().__init__()

        # First layer: no BN (common practice)
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),    # 256→128
            nn.LeakyReLU(0.2, inplace=True),

            DiscBlock(ndf,     ndf * 2),   # 128→64
            DiscBlock(ndf * 2, ndf * 4),   # 64→32
            DiscBlock(ndf * 4, ndf * 8),   # 32→16
            DiscBlock(ndf * 8, ndf * 8),   # 16→8

            # Final patch prediction (no sigmoid — handled by loss)
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1, bias=False),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d,)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


# ══════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    G = Generator(z_dim=512, w_dim=512, ngf=64).to(device)
    D = Discriminator(ndf=64).to(device)

    z    = torch.randn(4, 512, device=device)
    imgs = G(z)
    pred = D(imgs)

    print(f"✅ Generator  output : {imgs.shape}")   # [4, 3, 256, 256]
    print(f"✅ Discriminator out : {pred.shape}")   # [4, 1, ?, ?]

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"📊 Generator  params : {g_params:.1f}M")
    print(f"📊 Discriminator params: {d_params:.1f}M")
