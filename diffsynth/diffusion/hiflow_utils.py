import math
import torch
import torch.nn.functional as F

# ---------- basic mapping ----------
def predict_x0_from_v(x: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # x0 = x - sigma*v
    return x - sigma * v

def v_from_x0(x: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # v = (x - x0)/sigma
    return (x - x0) / torch.clamp(sigma, min=eps)

# ---------- latent spatial upsample (phi) ----------
def upsample_latents_bcthw(x: torch.Tensor, target_h: int, target_w: int, mode: str = "bilinear") -> torch.Tensor:
    # x: [B,C,T,H,W] -> upsample on (H,W)
    b, c, t, h, w = x.shape
    xt = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    xt = F.interpolate(xt, size=(target_h, target_w), mode=mode, align_corners=False if mode in ["bilinear", "bicubic"] else None)
    xt = xt.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()
    return xt

# ---------- Butterworth low-pass on each frame (2D FFT) ----------
_filter_cache = {}

def _butterworth_lowpass_mask(H: int, W: int, cutoff: float, order: int, device, dtype):
    # cutoff: normalized radius in [0, 0.5] typically, we normalize by 0.5 -> [0,1]
    key = (H, W, float(cutoff), int(order), str(device), str(dtype))
    if key in _filter_cache:
        return _filter_cache[key]

    fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=torch.float32)  # [-0.5,0.5)
    fx = torch.fft.fftfreq(W, d=1.0, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(xx * xx + yy * yy)  # [0, ~0.707]
    rr = rr / 0.5  # normalize so Nyquist radius ~1
    # Butterworth LP: 1 / (1 + (r/c)^ (2n))
    c = max(cutoff / 0.5, 1e-6)  # normalize cutoff similarly
    mask = 1.0 / (1.0 + (rr / c) ** (2 * order))
    mask = mask.to(dtype=torch.float32)  # keep fp32 for stability
    _filter_cache[key] = mask
    return mask

def lowfreq_bcthw(x: torch.Tensor, cutoff: float = 0.08, order: int = 2) -> torch.Tensor:
    """
    x: [B,C,T,H,W], return low-frequency component with Butterworth LPF applied per-frame per-channel.
    """
    b, c, t, h, w = x.shape
    device = x.device
    mask = _butterworth_lowpass_mask(h, w, cutoff=cutoff, order=order, device=device, dtype=x.dtype)  # [H,W]

    # reshape to [N,H,W]
    xt = x.permute(0, 2, 1, 3, 4).reshape(b * t * c, h, w).to(torch.float32)
    X = torch.fft.fft2(xt)  # complex64
    X = X * mask[None, :, :]  # broadcast
    xt_lp = torch.fft.ifft2(X).real
    xt_lp = xt_lp.to(dtype=x.dtype)

    out = xt_lp.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    return out
