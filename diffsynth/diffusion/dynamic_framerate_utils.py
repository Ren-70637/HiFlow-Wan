"""
Dynamic Frame Rate Utilities for HiFlow acceleration.
Migrated and adapted from Dynamic-Res project.

Supports sparse temporal sampling during diffusion steps:
- Apply sparse sampling before model forward
- Restore to full frames after model forward
- Scheduler step is always on full frames for consistency
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


def parse_framerate_schedule(schedule_str: str, num_inference_steps: int) -> Dict[int, int]:
    """
    Parse framerate schedule string.
    
    Args:
        schedule_str: Format "start-end:stride,start-end:stride,..."
                     e.g. "0-10:2,10-20:1,20-30:4"
                     stride=1 means full frames, stride=2 means every other frame, etc.
        num_inference_steps: Total inference steps
    
    Returns:
        framerate_map: {step_index: stride} dict
    
    Examples:
        >>> parse_framerate_schedule("0-10:2,10-20:1", 30)
        {0:2, 1:2, ..., 9:2, 10:1, 11:1, ..., 19:1, 20:1, ..., 29:1}
    """
    if not schedule_str:
        # Default: all steps use full frames
        return {i: 1 for i in range(num_inference_steps)}
    
    framerate_map = {}
    
    # Fill all steps with default (stride=1, full frames)
    for i in range(num_inference_steps):
        framerate_map[i] = 1
    
    # Parse custom schedule
    segments = schedule_str.split(',')
    for segment in segments:
        try:
            range_part, stride_part = segment.split(':')
            stride = int(stride_part)
            
            if '-' in range_part:
                start, end = map(int, range_part.split('-'))
            else:
                # Single step
                start = end = int(range_part)
            
            # Ensure valid range
            start = max(0, min(start, num_inference_steps - 1))
            end = max(0, min(end, num_inference_steps - 1))
            
            # Set stride for this range
            for step_idx in range(start, end + 1):
                framerate_map[step_idx] = stride
        
        except Exception as e:
            print(f"[dynamic_framerate] Warning: Failed to parse segment '{segment}': {e}")
            continue
    
    return framerate_map


def apply_temporal_sparse_sampling(
    latent_input: torch.Tensor,
    stride: int,
    keep_boundary: bool = True
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Apply sparse sampling on temporal dimension.
    
    Args:
        latent_input: [B, C, T, H, W] input latent
        stride: temporal stride (1=full frames, 2=every other frame, etc.)
        keep_boundary: whether to keep first and last frames (ensures temporal continuity)
    
    Returns:
        latent_sparse: [B, C, T', H, W] sparse sampled latent
        frame_info: dict with original info for restoration
    """
    if stride <= 1:
        return latent_input, None
    
    B, C, T, H, W = latent_input.shape
    
    if T <= 2:
        # Too few frames, skip sparse sampling
        return latent_input, None
    
    # Build frame indices
    if keep_boundary:
        # Keep first/last frames + sample middle by stride
        middle_indices = list(range(1, T - 1, stride))
        frame_indices = [0] + middle_indices + [T - 1]
        frame_indices = sorted(set(frame_indices))  # Dedupe and sort
    else:
        # Simple uniform sampling
        frame_indices = list(range(0, T, stride))
    
    # Ensure all indices are in valid range [0, T-1]
    frame_indices = [idx for idx in frame_indices if 0 <= idx < T]
    
    if len(frame_indices) == 0:
        frame_indices = [0]
    
    frame_indices_tensor = torch.tensor(frame_indices, device=latent_input.device, dtype=torch.long)
    
    # Sample
    latent_sparse = latent_input[:, :, frame_indices_tensor, :, :]
    
    frame_info = {
        'original_shape': latent_input.shape,
        'original_T': T,
        'sampled_T': len(frame_indices),
        'frame_indices': frame_indices_tensor,
        'stride': stride
    }
    
    return latent_sparse, frame_info


def restore_temporal_full_frames(
    latent_sparse: torch.Tensor,
    frame_info: Optional[Dict],
    interpolation_mode: str = 'trilinear'
) -> torch.Tensor:
    """
    Restore sparse frames to full frame count via interpolation.
    
    Args:
        latent_sparse: [B, C, T', H, W] sparse sampled latent
        frame_info: info dict from apply_temporal_sparse_sampling
        interpolation_mode: interpolation mode for F.interpolate
    
    Returns:
        latent_full: [B, C, T, H, W] restored to original frame count
    """
    if frame_info is None:
        # No sparse sampling was done
        return latent_sparse
    
    original_T = frame_info['original_T']
    current_T = latent_sparse.shape[2]
    
    if current_T == original_T:
        # Already full frames
        return latent_sparse
    
    # Use interpolation to restore full frames
    B, C, _, H, W = latent_sparse.shape
    
    # F.interpolate for 3D needs 5D input: [B, C, D, H, W]
    latent_full = F.interpolate(
        latent_sparse,
        size=(original_T, H, W),
        mode=interpolation_mode,
        align_corners=False if interpolation_mode != 'nearest' else None
    )
    
    return latent_full


def get_framerate_for_step(step_index: int, framerate_map: Dict[int, int]) -> int:
    """
    Get stride for a specific step.
    
    Args:
        step_index: current inference step index
        framerate_map: step to stride mapping
    
    Returns:
        stride: temporal stride for this step
    """
    return framerate_map.get(step_index, 1)  # Default 1 (full frames)


def apply_dynamic_framerate(
    latent_model_input: torch.Tensor,
    step_index: int,
    framerate_map: Dict[int, int],
    keep_boundary: bool = True
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Apply dynamic framerate based on current step.
    
    Args:
        latent_model_input: [B, C, T, H, W]
        step_index: current inference step
        framerate_map: framerate schedule dict
        keep_boundary: whether to keep first/last frames
    
    Returns:
        latent_adjusted: adjusted latent (sparse or original)
        frame_info: frame info dict (None if no sparse sampling)
    """
    stride = get_framerate_for_step(step_index, framerate_map)
    
    if stride <= 1:
        return latent_model_input, None
    
    return apply_temporal_sparse_sampling(
        latent_model_input,
        stride,
        keep_boundary
    )


def log_framerate_schedule(framerate_map: Dict[int, int], num_inference_steps: int):
    """
    Print framerate schedule summary.
    """
    print("=" * 60)
    print("Dynamic Frame Rate Schedule:")
    print("-" * 60)
    
    # Count steps per stride
    stride_counts = {}
    for stride in framerate_map.values():
        stride_counts[stride] = stride_counts.get(stride, 0) + 1
    
    for stride in sorted(stride_counts.keys()):
        count = stride_counts[stride]
        percentage = count / num_inference_steps * 100
        frame_desc = "full frames" if stride == 1 else f"every {stride}th frame"
        print(f"  Stride {stride} ({frame_desc}): {count} steps ({percentage:.1f}%)")
    
    print("=" * 60)


# ============ Spatial resize utilities (for dynamic resolution) ============

def resize_latents_spatial(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "bilinear"
) -> torch.Tensor:
    """
    Resize latents spatially (H, W only, keep T unchanged).
    
    Args:
        x: [B, C, T, H, W]
        target_h: target height
        target_w: target width
        mode: interpolation mode
    
    Returns:
        Resized tensor [B, C, T, target_h, target_w]
    """
    B, C, T, H, W = x.shape
    
    if H == target_h and W == target_w:
        return x
    
    # Reshape to [B*T, C, H, W] for 2D interpolation
    x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    
    x_resized = F.interpolate(
        x_flat,
        size=(target_h, target_w),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None
    )
    
    # Reshape back to [B, C, T, H', W']
    x_resized = x_resized.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()
    
    return x_resized


def compute_stage_resolution(
    target_h: int,
    target_w: int,
    res_rate: float,
    patch_h: int = 2,
    patch_w: int = 2,
    min_h: int = 4,
    min_w: int = 4
) -> Tuple[int, int]:
    """
    Compute resolution for a stage, ensuring patch alignment.
    
    Args:
        target_h: final target latent height
        target_w: final target latent width
        res_rate: resolution rate (0.5 = half resolution)
        patch_h: patch height (must be divisible)
        patch_w: patch width (must be divisible)
        min_h: minimum height
        min_w: minimum width
    
    Returns:
        (stage_h, stage_w): aligned resolution for this stage
    """
    # Compute raw resolution
    h_raw = int(target_h * res_rate)
    w_raw = int(target_w * res_rate)
    
    # Align to patch size
    h_aligned = max(patch_h, (h_raw // patch_h) * patch_h)
    w_aligned = max(patch_w, (w_raw // patch_w) * patch_w)
    
    # Ensure even (for some VAE requirements)
    h_aligned = max(min_h, (h_aligned // 2) * 2)
    w_aligned = max(min_w, (w_aligned // 2) * 2)
    
    return h_aligned, w_aligned


def get_current_stage(
    progress_id: int,
    stage_start_idx: int,
    res_step_list: List[int],
    res_rate_list: List[float]
) -> Tuple[int, float]:
    """
    Determine current resolution stage based on progress.
    
    Args:
        progress_id: current global step index
        stage_start_idx: step index where dynamic res starts (tau_index for HiFlow Stage2)
        res_step_list: list of step offsets (relative to stage_start_idx) for each stage
        res_rate_list: list of resolution rates for each stage
    
    Returns:
        (stage_idx, res_rate): current stage index and resolution rate
    """
    relative_step = progress_id - stage_start_idx
    
    current_stage = 0
    for i, step_threshold in enumerate(res_step_list):
        if relative_step >= step_threshold:
            current_stage = i
    
    return current_stage, res_rate_list[current_stage]


def should_switch_resolution(
    progress_id: int,
    stage_start_idx: int,
    res_step_list: List[int]
) -> Tuple[bool, Optional[int]]:
    """
    Check if we should switch resolution at this step.
    
    Args:
        progress_id: current global step index
        stage_start_idx: step index where dynamic res starts
        res_step_list: list of step offsets for each stage
    
    Returns:
        (should_switch, next_stage_idx): whether to switch and next stage index
    """
    relative_step = progress_id - stage_start_idx
    
    for i, step_threshold in enumerate(res_step_list):
        if relative_step == step_threshold and i > 0:
            return True, i
    
    return False, None
