import torch
import time
from typing import Dict, List, Optional, Tuple

from ..diffusion.hiflow_utils import (
    predict_x0_from_v,
    v_from_x0,
    upsample_latents_bcthw,
    lowfreq_bcthw,
)

from ..diffusion.dynamic_framerate_utils import (
    parse_framerate_schedule,
    apply_dynamic_framerate,
    restore_temporal_full_frames,
    resize_latents_spatial,
    compute_stage_resolution,
    get_current_stage,
    should_switch_resolution,
)

# -------------------------
# helpers: cfg + dit switch
# -------------------------
def _maybe_switch_dit(pipe, models: Dict, timestep_tensor_1d: torch.Tensor, switch_DiT_boundary: float):
    # 你原来的逻辑：timestep.item() < switch_DiT_boundary * 1000
    if (
        timestep_tensor_1d.item() < switch_DiT_boundary * 1000
        and getattr(pipe, "dit2", None) is not None
        and (not models["dit"] is pipe.dit2)
    ):
        pipe.load_models_to_device(pipe.in_iteration_models_2)
        models["dit"] = pipe.dit2
        models["vace"] = pipe.vace2
    return models


def _perf_sample_begin(enable: bool):
    if not enable:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
    else:
        e0 = e1 = None
    t0 = time.perf_counter()
    return (t0, e0, e1)


def _perf_sample_end(perf, key: str, state):
    if state is None:
        return
    t0, e0, e1 = state
    wall_ms = (time.perf_counter() - t0) * 1000.0
    cuda_ms = None
    if e0 is not None:
        e1.record()
        torch.cuda.synchronize()
        cuda_ms = float(e0.elapsed_time(e1))
    if perf is None:
        return
    add = getattr(perf, "add_sample", None)
    if callable(add):
        add(key, wall_ms=wall_ms, cuda_ms=cuda_ms)


@torch.no_grad()
def _cfg_guided_v(
    pipe,
    models: Dict,
    inputs_shared: Dict,
    inputs_posi: Dict,
    inputs_nega: Dict,
    timestep: torch.Tensor,          # [1]
    cfg_scale: float,
    cfg_merge: bool,
):
    # 与你 wan_video.py 完全一致
    v_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
    if cfg_scale != 1.0:
        if cfg_merge:
            v_posi, v_nega = v_posi.chunk(2, dim=0)
        else:
            v_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
        v = v_nega + cfg_scale * (v_posi - v_nega)
    else:
        v = v_posi
    return v


# -----------------------------------------
# Stage-1: low-res denoise + record x0_pred
# -----------------------------------------
@torch.no_grad()
def _run_denoise_record_x0(
    pipe,
    models: Dict,
    inputs_shared: Dict,
    inputs_posi: Dict,
    inputs_nega: Dict,
    cfg_scale: float,
    cfg_merge: bool,
    switch_DiT_boundary: float,
    progress_bar_cmd,
    store_on_cpu: bool = True,
    # profiling
    perf=None,
    profile_sample_every: int = 0,
) -> List[torch.Tensor]:
    """
    返回：x0_low_pred_list[i]，长度 = num_inference_steps（与 scheduler.timesteps 对齐）
    每个元素 shape = [B,C,T,Hlow,Wlow]
    """
    x0_list: List[torch.Tensor] = []

    # 确保 scheduler 已经在外部 set_timesteps 过（wan_video.py 本来就会做）
    for progress_id, ts_cpu in enumerate(progress_bar_cmd(pipe.scheduler.timesteps)):
        sample_on = bool(profile_sample_every and profile_sample_every > 0 and (progress_id % int(profile_sample_every) == 0))
        st = _perf_sample_begin(sample_on)
        # Switch DiT if necessary（保持一致）
        models = _maybe_switch_dit(pipe, models, ts_cpu, switch_DiT_boundary)

        # timestep 给模型用：要 [1] 且在 GPU
        timestep = ts_cpu.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        # v = dx/dσ（对齐你的 scheduler）
        v = _cfg_guided_v(pipe, models, inputs_shared, inputs_posi, inputs_nega, timestep, cfg_scale, cfg_merge)

        # 取 σ（与 progress_id 对齐）
        sigma = pipe.scheduler.sigmas[progress_id].to(device=pipe.device, dtype=pipe.torch_dtype)
        sigma_view = sigma.view(1, 1, 1, 1, 1)

        # x0_pred = x - σ v
        x = inputs_shared["latents"]
        x0_pred = predict_x0_from_v(x, v, sigma_view)

        # 记录（建议 CPU，省显存）
        if store_on_cpu:
            x0_list.append(x0_pred.detach().to("cpu"))
        else:
            x0_list.append(x0_pred.detach())

        # 积分一步（与你原循环一致：step 里用的是 ts_cpu，而不是 GPU timestep）
        inputs_shared["latents"] = pipe.scheduler.step(v, pipe.scheduler.timesteps[progress_id], x)

        # 你原逻辑：锁首帧（虽然你说不用，但保留不影响）
        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        _perf_sample_end(perf, "hiflow_stage1_step", st)

    return x0_list


# -----------------------------------------
# Stage-2: HiFlow on high-res from tau_index
# -----------------------------------------
@torch.no_grad()
def _run_denoise_hiflow(
    pipe,
    models: Dict,
    inputs_shared: Dict,
    inputs_posi: Dict,
    inputs_nega: Dict,
    x0_low_list: List[torch.Tensor],
    high_latent_h: int,
    high_latent_w: int,
    tau_ratio: float,          # 0~1：从第 tau_ratio*N 步开始做 high-res
    cfg_scale: float,
    cfg_merge: bool,
    switch_DiT_boundary: float,
    progress_bar_cmd,
    alpha: float = 0.6,        # direction alignment strength
    beta: float = 0.4,         # acceleration alignment strength
    lp_cutoff: float = 0.08,
    lp_order: int = 2,
    upsample_mode: str = "bilinear",
    ntk_factor_h: float = 1.0,
    ntk_factor_w: float = 1.0,
    train_seq_len: Optional[int] = None,
    # profiling
    perf=None,
    profile_sample_every: int = 0,
    # ========== Dynamic Frame Rate (NEW) ==========
    enable_dynamic_framerate: bool = False,
    framerate_schedule: Optional[str] = None,
    framerate_interpolation_mode: str = "trilinear",
    framerate_keep_boundary: bool = True,
    # ========== Dynamic Resolution (NEW) ==========
    enable_dynamic_res: bool = False,
    res_rate_list: Optional[List[float]] = None,
    res_step_list: Optional[List[int]] = None,
    res_upsample_mode: str = "bilinear",
) -> torch.Tensor:
    """
    返回 high-res 最终 latents: [B,C,T,Hhigh,Whigh]
    
    New features:
    - Dynamic Frame Rate: sparse temporal sampling during model forward
    - Dynamic Resolution: progressive resolution scaling during Stage2
    """
    N = len(pipe.scheduler.timesteps)
    assert len(x0_low_list) == N, "x0_low_list length must match scheduler.timesteps"

    # tau_index：从这个步开始跑 high-res
    tau_ratio = float(max(0.0, min(0.999, tau_ratio)))
    tau_index = int(tau_ratio * N)

    # ========== Parse dynamic framerate schedule ==========
    framerate_map = None
    if enable_dynamic_framerate and framerate_schedule:
        framerate_map = parse_framerate_schedule(framerate_schedule, N)
    
    # ========== Parse dynamic resolution config ==========
    # Validate and default res config
    if enable_dynamic_res:
        if res_rate_list is None or res_step_list is None:
            enable_dynamic_res = False
            print("[HiFlow] Warning: enable_dynamic_res=True but res_rate_list or res_step_list is None. Disabled.")
        elif len(res_rate_list) != len(res_step_list):
            enable_dynamic_res = False
            print("[HiFlow] Warning: res_rate_list and res_step_list length mismatch. Disabled.")
        else:
            # Ensure res_step_list is sorted
            res_step_list = list(res_step_list)
            res_rate_list = list(res_rate_list)
            # Make sure first stage starts at 0
            if res_step_list[0] != 0:
                res_step_list = [0] + res_step_list
                res_rate_list = [res_rate_list[0]] + res_rate_list

    # Get patch size for resolution alignment
    pT, pH, pW = models["dit"].patch_size if hasattr(models["dit"], "patch_size") else (1, 2, 2)

    # ========== Prepare base noise (used for all stages) ==========
    B, C, T = inputs_shared["latents"].shape[:3]
    
    # Compute initial resolution for Stage2
    if enable_dynamic_res and res_rate_list:
        initial_rate = res_rate_list[0]
        current_h, current_w = compute_stage_resolution(
            high_latent_h, high_latent_w, initial_rate, pH, pW
        )
    else:
        current_h, current_w = high_latent_h, high_latent_w
    
    # Base noise at final resolution (will be resized for each stage)
    noise_base = torch.randn((B, C, T, high_latent_h, high_latent_w), device=pipe.device, dtype=pipe.torch_dtype)
    
    # Get noise for current resolution
    if current_h == high_latent_h and current_w == high_latent_w:
        noise_current = noise_base
    else:
        noise_current = resize_latents_spatial(noise_base, current_h, current_w, mode=res_upsample_mode)

    # 取 tau 时刻的 x0_ref(tau) = phi(x0_low_pred[tau])
    x0_low_tau = x0_low_list[tau_index].to(device=pipe.device, dtype=pipe.torch_dtype)
    x0_ref_tau = upsample_latents_bcthw(x0_low_tau, current_h, current_w, mode=upsample_mode)

    # 初始化：x_high(σ_tau) = add_noise(x0_ref_tau, noise_current, timestep_tau)
    ts_tau = pipe.scheduler.timesteps[tau_index].to(device=pipe.device, dtype=pipe.torch_dtype)
    inputs_shared["latents"] = pipe.scheduler.add_noise(x0_ref_tau, noise_current, ts_tau)

    v_prev: Optional[torch.Tensor] = None
    vref_prev: Optional[torch.Tensor] = None
    
    # Track current stage for dynamic resolution
    current_stage_idx = 0

    _sentinel = object()
    prev_train_seq_len = inputs_shared.get("train_seq_len", _sentinel)
    prev_ntk_h = inputs_shared.get("ntk_factor_h", _sentinel)
    prev_ntk_w = inputs_shared.get("ntk_factor_w", _sentinel)

    if train_seq_len is not None:
        inputs_shared["train_seq_len"] = train_seq_len
    else:
        inputs_shared.pop("train_seq_len", None)

    inputs_shared["ntk_factor_h"] = ntk_factor_h
    inputs_shared["ntk_factor_w"] = ntk_factor_w

    # 从 tau_index 开始积分到末尾
    try:
        for progress_id in range(tau_index, N):
            sample_on = bool(profile_sample_every and profile_sample_every > 0 and (((progress_id - tau_index) % int(profile_sample_every)) == 0))
            st = _perf_sample_begin(sample_on)
            ts_cpu = pipe.scheduler.timesteps[progress_id]
            models = _maybe_switch_dit(pipe, models, ts_cpu, switch_DiT_boundary)
            timestep = ts_cpu.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            
            # ========== Dynamic Frame Rate: sparse sampling before model forward ==========
            frame_info = None
            latents_full = inputs_shared["latents"]
            
            if framerate_map is not None:
                latents_sparse, frame_info = apply_dynamic_framerate(
                    latents_full, progress_id, framerate_map, keep_boundary=framerate_keep_boundary
                )
                if frame_info is not None:
                    # Temporarily use sparse latents for model forward
                    inputs_shared["latents"] = latents_sparse
            
            # Model forward
            v_model = _cfg_guided_v(pipe, models, inputs_shared, inputs_posi, inputs_nega, timestep, cfg_scale, cfg_merge)
            
            # ========== Dynamic Frame Rate: restore full frames after model forward ==========
            if frame_info is not None:
                # Restore latents to full
                inputs_shared["latents"] = latents_full
                # Restore v_model to full frames
                v_model = restore_temporal_full_frames(
                    v_model.unsqueeze(0) if v_model.dim() == 4 else v_model,
                    frame_info,
                    interpolation_mode=framerate_interpolation_mode
                )
                if v_model.dim() == 5 and v_model.shape[0] == 1:
                    v_model = v_model  # Keep [B,C,T,H,W]
            
            sigma = pipe.scheduler.sigmas[progress_id].to(device=pipe.device, dtype=pipe.torch_dtype)
            sigma_view = sigma.view(1, 1, 1, 1, 1)
            x = inputs_shared["latents"]
            x0_pred = predict_x0_from_v(x, v_model, sigma_view)
            
            # Upsample x0_low to current resolution (may differ from final high_latent_h/w)
            x0_low_i = x0_low_list[progress_id].to(device=pipe.device, dtype=pipe.torch_dtype)
            x0_ref = upsample_latents_bcthw(x0_low_i, current_h, current_w, mode=upsample_mode)

            lp_ref = lowfreq_bcthw(x0_ref, cutoff=lp_cutoff, order=lp_order)
            lp_pred = lowfreq_bcthw(x0_pred, cutoff=lp_cutoff, order=lp_order)
            x0_aligned = x0_pred + alpha * (lp_ref - lp_pred)

            v = v_from_x0(x, x0_aligned, sigma_view)

            # Reference flow with current resolution noise
            v_ref = noise_current - x0_ref
            if v_prev is not None and vref_prev is not None:
                dv = v - v_prev
                dv_ref = v_ref - vref_prev
                v = v + beta * (dv_ref - dv)

            v_prev = v
            vref_prev = v_ref

            inputs_shared["latents"] = pipe.scheduler.step(v, ts_cpu, x)
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
            
            # ========== Dynamic Resolution: check if we need to switch ==========
            if enable_dynamic_res and res_step_list and progress_id < N - 1:
                # Check next step
                next_progress_id = progress_id + 1
                relative_next = next_progress_id - tau_index
                
                # Find if we're at a resolution switch point
                should_switch = False
                next_stage = current_stage_idx
                for i, step_threshold in enumerate(res_step_list):
                    if relative_next == step_threshold and i > current_stage_idx:
                        should_switch = True
                        next_stage = i
                        break
                
                if should_switch and next_stage < len(res_rate_list):
                    # Compute new resolution
                    new_rate = res_rate_list[next_stage]
                    new_h, new_w = compute_stage_resolution(
                        high_latent_h, high_latent_w, new_rate, pH, pW
                    )
                    
                    if new_h != current_h or new_w != current_w:
                        # Use x0_aligned as clean estimate for transition
                        x0_switch = x0_aligned
                        
                        # Upsample to new resolution
                        x0_switch_up = resize_latents_spatial(x0_switch, new_h, new_w, mode=res_upsample_mode)
                        
                        # Get noise for new resolution
                        noise_new = resize_latents_spatial(noise_base, new_h, new_w, mode=res_upsample_mode)
                        
                        # Add noise at next timestep level
                        ts_next = pipe.scheduler.timesteps[next_progress_id].to(device=pipe.device, dtype=pipe.torch_dtype)
                        inputs_shared["latents"] = pipe.scheduler.add_noise(x0_switch_up, noise_new, ts_next)
                        
                        # Update current resolution tracking
                        current_h, current_w = new_h, new_w
                        noise_current = noise_new
                        current_stage_idx = next_stage
                        
                        # Reset v_prev/vref_prev (shape changed)
                        v_prev = None
                        vref_prev = None
            
            _perf_sample_end(perf, "hiflow_stage2_step", st)
            
    finally:
        for key, prev in (("train_seq_len", prev_train_seq_len),
                        ("ntk_factor_h", prev_ntk_h),
                        ("ntk_factor_w", prev_ntk_w)):
            if prev is _sentinel:
                inputs_shared.pop(key, None)
            else:
                inputs_shared[key] = prev
    
    # If we ended at a lower resolution, upsample final result to target
    if current_h != high_latent_h or current_w != high_latent_w:
        inputs_shared["latents"] = resize_latents_spatial(
            inputs_shared["latents"], high_latent_h, high_latent_w, mode=res_upsample_mode
        )
    
    return inputs_shared["latents"]
