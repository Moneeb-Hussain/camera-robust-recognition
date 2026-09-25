"""
invariance_core.py  (starter, tested on CPU with a dummy encoder)

Three pieces:
  1. Distortion ops   : perspective, photometric, blur, noise, moire (all differentiable)
  2. AutoAugmentor    : learnable distortion settings, trained to HURT the head (adversary)
  3. InvariantHead    : small residual MLP on top of a FROZEN CLIP image feature

Images are float tensors in [0, 1], shape (B, 3, H, W).
The frozen encoder is any callable: enc(x) -> (B, D). Wrap open_clip so it does its own
normalization inside, and keep its parameters requires_grad=False.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as KT
import kornia.filters as KF

OPS = ["moire", "geo", "photo", "noise", "blur"]  # same order idea as the TIACam composition


# ----------------------------------------------------------------------------- ops
def op_perspective(x, offsets):
    """offsets: (B, 4, 2), each corner moved by a fraction of image size."""
    B, C, H, W = x.shape
    src = torch.tensor([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                       dtype=x.dtype, device=x.device).expand(B, 4, 2)
    scale = torch.tensor([W, H], dtype=x.dtype, device=x.device)
    dst = src + offsets * scale
    M = KT.get_perspective_transform(src, dst)
    return KT.warp_perspective(x, M, (H, W), padding_mode="border")


def op_photometric(x, alpha, beta, gamma):
    """alpha, beta, gamma: (B,) -> contrast, brightness, gamma."""
    a, b, g = (t.view(-1, 1, 1, 1) for t in (alpha, beta, gamma))
    return (a * x.clamp(1e-4, 1.0) ** g + b).clamp(0, 1)


def op_blur(x, sigma):
    """sigma: (B,) pixels at 224 px width. Scaled automatically for bigger images (e.g. full frames)."""
    s = (sigma * (x.shape[-1] / 224.0)).clamp(min=0.1)
    k = int(2 * math.ceil(3 * float(s.max().detach())) + 1)   # kernel covers about 3 sigma each side
    return KF.gaussian_blur2d(x, (k, k), s.view(-1, 1).expand(-1, 2))


def op_noise(x, sigma):
    return (x + sigma.view(-1, 1, 1, 1) * torch.randn_like(x)).clamp(0, 1)


def op_moire(x, amp, fx, fy, phase):
    """Stripe pattern added to the image. fx, fy in cycles per image."""
    B, C, H, W = x.shape
    v = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
    u = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W)
    fx, fy, ph, am = (t.view(-1, 1, 1, 1) for t in (fx, fy, phase, amp))
    return (x + am * torch.sin(2 * math.pi * (fx * u + fy * v) + ph)).clamp(0, 1)


# ----------------------------------------------------------------------------- learned augmentor
class AutoAugmentor(nn.Module):
    """
    Every distortion setting = lo + (hi - lo) * sigmoid(raw). The raw numbers are the
    learnable parameters (Theta). Each sample gets a little random jitter on raw, so one
    batch sees many different distortions around the current setting.
    Ranges are the physical limits (the "clamp" in the paper), so it cannot go crazy.
    """
    RANGES = {                       # (lo, hi) at full strength
        "corner":  (-0.18, 0.18),    # perspective corner shift, fraction of size
        "alpha":   (0.6, 1.4),       # contrast
        "beta":    (-0.2, 0.2),      # brightness
        "gamma":   (0.6, 1.6),
        "sigma_b": (0.1, 2.5),       # blur sigma in pixels
        "sigma_n": (0.0, 0.08),      # noise std
        "amp":     (0.0, 0.10),      # moire strength
        "fx":      (4.0, 40.0), "fy": (4.0, 40.0),
    }

    def __init__(self, ops=OPS, jitter=0.6):
        super().__init__()
        self.ops, self.jitter = list(ops), jitter
        shapes = {"corner": (4, 2)}
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.zeros(shapes.get(k, ()))) for k in self.RANGES})

    def _val(self, name, B, device):
        lo, hi = self.RANGES[name]
        raw = self.raw[name].to(device)
        noise = self.jitter * torch.randn(B, *raw.shape, device=device)
        return lo + (hi - lo) * torch.sigmoid(raw + noise)

    def forward(self, x):
        B, dev = x.shape[0], x.device
        for op in self.ops:
            if op == "moire":
                x = op_moire(x, self._val("amp", B, dev), self._val("fx", B, dev),
                             self._val("fy", B, dev), torch.rand(B, device=dev) * 2 * math.pi)
            elif op == "geo":
                x = op_perspective(x, self._val("corner", B, dev))
            elif op == "photo":
                x = op_photometric(x, self._val("alpha", B, dev), self._val("beta", B, dev),
                                   self._val("gamma", B, dev))
            elif op == "noise":
                x = op_noise(x, self._val("sigma_n", B, dev))
            elif op == "blur":
                x = op_blur(x, self._val("sigma_b", B, dev))
        return x


# ----------------------------------------------------------------------------- fixed distortions for evaluation
@torch.no_grad()
def distort(x, kind, severity, seed=0):
    """Deterministic test distortion. severity in [0, 1]. kind in OPS + ['combined']."""
    if severity <= 0:
        return x
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, dev = x.shape[0], x.device
    R = lambda *s: (torch.rand(*s, generator=g) * 2 - 1).to(dev)      # uniform in [-1, 1]
    kinds = OPS if kind == "combined" else [kind]
    for k in kinds:
        if k == "geo":
            x = op_perspective(x, R(B, 4, 2) * 0.18 * severity)
        elif k == "photo":
            x = op_photometric(x, 1 + R(B) * 0.4 * severity, R(B) * 0.2 * severity, 1 + R(B) * 0.6 * severity)
        elif k == "blur":
            x = op_blur(x, torch.full((B,), 0.1 + 2.4 * severity, device=dev))
        elif k == "noise":
            x = op_noise(x, torch.full((B,), 0.08 * severity, device=dev))
        elif k == "moire":
            x = op_moire(x, torch.full((B,), 0.10 * severity, device=dev),
                         (R(B) + 1) * 18 + 4, (R(B) + 1) * 18 + 4, (R(B) + 1) * math.pi)
    return x


# ----------------------------------------------------------------------------- head
class ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p),
                               nn.Linear(d, d), nn.BatchNorm1d(d))

    def forward(self, h):
        return F.relu(self.f(h) + h)


class InvariantHead(nn.Module):
    """3 residual blocks + projection into the CLIP text space (so we can score against text anchors)."""
    def __init__(self, d_in=512, d_out=512):
        super().__init__()
        self.blocks = nn.Sequential(ResBlock(d_in), ResBlock(d_in), ResBlock(d_in))
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, z):
        return F.normalize(self.proj(self.blocks(z)), dim=-1)


# ----------------------------------------------------------------------------- losses
def loss_inv(F_hat, F_clean):             # invariance: distorted feature should match clean feature
    return (1 - (F_hat * F_clean).sum(-1)).mean()

def loss_text(Fx, anchors, y, scale=30.0):  # text anchoring: feature should point at its class caption
    return F.cross_entropy(scale * Fx @ anchors.t(), y)

def loss_sem(z_hat, z_clean, tau=0.15):   # fidelity: keep the distorted image "still the same thing" in CLIP space
    drift = 1 - F.cosine_similarity(z_hat, z_clean, dim=-1)
    return F.relu(drift - tau).mean()


# ----------------------------------------------------------------------------- one training step
def train_step(x, y, enc, head, aug, anchors, opt_head, opt_aug, lam_text=1.0, lam_sem=5.0):
    """
    Step A: the augmentor climbs (finds distortions that break the head, but stay faithful).
    Step B: the head descends (learns to ignore those distortions and stay near its text anchor).
    enc is frozen, but gradients still flow THROUGH it into the augmentor in step A.
    """
    with torch.no_grad():
        z_clean = enc(x)
    # ---- Step A
    head.eval()   # BatchNorm in eval so step A does not disturb running stats
    with torch.no_grad():
        F_clean = head(z_clean)
    z_hat = enc(aug(x))
    F_hat = head(z_hat)
    lossA = -(loss_inv(F_hat, F_clean) - lam_sem * loss_sem(z_hat, z_clean))
    opt_aug.zero_grad(); head.zero_grad(); lossA.backward(); opt_aug.step()
    # ---- Step B
    head.train()
    with torch.no_grad():
        z_hat = enc(aug(x))
    F_clean, F_hat = head(z_clean), head(z_hat)
    lossB = loss_inv(F_hat, F_clean) + lam_text * (loss_text(F_clean, anchors, y) + loss_text(F_hat, anchors, y))
    opt_head.zero_grad(); lossB.backward(); opt_head.step()
    return {"A_inv": -lossA.item(), "B_total": lossB.item()}


# ----------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    # dummy frozen "encoder": conv net -> 512-d. Replace with wrapped open_clip image encoder.
    enc = nn.Sequential(nn.Conv2d(3, 8, 7, 4), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128, 512))
    for p in enc.parameters():
        p.requires_grad_(False)
    head, aug = InvariantHead(), AutoAugmentor()
    anchors = F.normalize(torch.randn(70, 512), dim=-1)
    opt_head = torch.optim.AdamW(head.parameters(), 1e-3)
    opt_aug = torch.optim.Adam(aug.parameters(), 5e-2)
    x = torch.rand(16, 3, 224, 224); y = torch.randint(0, 70, (16,))

    for kind in OPS + ["combined"]:
        for s in (0.0, 0.5, 1.0):
            d = distort(x, kind, s)
            assert d.shape == x.shape and torch.isfinite(d).all() and d.min() >= 0 and d.max() <= 1, (kind, s)
    print("distort(): all kinds finite, in [0,1], same shape")

    raw0 = {k: v.detach().clone() for k, v in aug.raw.items()}
    head0 = head.proj.weight.detach().clone()
    for i in range(5):
        info = train_step(x, y, enc, head, aug, anchors, opt_head, opt_aug)
    moved = sum((aug.raw[k] - raw0[k]).abs().sum().item() for k in raw0)
    assert moved > 0, "augmentor got no gradient"
    assert (head.proj.weight - head0).abs().sum().item() > 0, "head did not update"
    assert all(p.grad is None for p in enc.parameters()), "encoder should stay untouched"
    print("train_step(): augmentor moved by %.4f, head updated, encoder untouched" % moved, info)
