import torch
from typing import Dict, List, Optional, Tuple

from ..diffusion.hiflow_utils import (
    predict_x0_from_v,
    v_from_x0,
    upsample_latents_bcthw,
    lowfreq_bcthw,
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
) -> List[torch.Tensor]:
    """
    返回：x0_low_pred_list[i]，长度 = num_inference_steps（与 scheduler.timesteps 对齐）
    每个元素 shape = [B,C,T,Hlow,Wlow]
    """
    x0_list: List[torch.Tensor] = []

    # 确保 scheduler 已经在外部 set_timesteps 过（wan_video.py 本来就会做）
    for progress_id, ts_cpu in enumerate(progress_bar_cmd(pipe.scheduler.timesteps)):
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
) -> torch.Tensor:
    """
    返回 high-res 最终 latents: [B,C,T,Hhigh,Whigh]
    """
    N = len(pipe.scheduler.timesteps)
    assert len(x0_low_list) == N, "x0_low_list length must match scheduler.timesteps"

    # tau_index：从这个步开始跑 high-res
    tau_ratio = float(max(0.0, min(0.999, tau_ratio)))
    tau_index = int(tau_ratio * N)

    # 准备 high-res 噪声端 x1_high（HiFlow 里 reference flow 用到）
    # 这里 batch/通道/帧数来自当前 inputs_shared["latents"] 的 B,C,T
    B, C, T = inputs_shared["latents"].shape[:3]
    noise_high = torch.randn((B, C, T, high_latent_h, high_latent_w), device=pipe.device, dtype=pipe.torch_dtype)

    # 取 tau 时刻的 x0_ref(tau) = phi(x0_low_pred[tau])
    x0_low_tau = x0_low_list[tau_index].to(device=pipe.device, dtype=pipe.torch_dtype)
    x0_ref_tau = upsample_latents_bcthw(x0_low_tau, high_latent_h, high_latent_w, mode=upsample_mode)

    # 初始化：x_high(σ_tau) = add_noise(x0_ref_tau, noise_high, timestep_tau)
    ts_tau = pipe.scheduler.timesteps[tau_index].to(device=pipe.device, dtype=pipe.torch_dtype)
    inputs_shared["latents"] = pipe.scheduler.add_noise(x0_ref_tau, noise_high, ts_tau)

    v_prev: Optional[torch.Tensor] = None
    vref_prev: Optional[torch.Tensor] = None

    # 从 tau_index 开始积分到末尾
    # 注意：progress_bar_cmd 只用于显示；我们仍然按 progress_id 精确取 sigma/timestep
    for progress_id in range(tau_index, N):
        ts_cpu = pipe.scheduler.timesteps[progress_id]   # CPU tensor（step 用）
        models = _maybe_switch_dit(pipe, models, ts_cpu, switch_DiT_boundary)

        timestep = ts_cpu.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # 这里注入 ntk_factor，供 model_fn_wan_video 通过 **kwargs 取到
        inputs_shared["ntk_factor_h"] = ntk_factor_h
        inputs_shared["ntk_factor_w"] = ntk_factor_w
        inputs_shared["train_seq_len"] = train_seq_len

        v_model = _cfg_guided_v(pipe, models, inputs_shared, inputs_posi, inputs_nega, timestep, cfg_scale, cfg_merge)

        # 当前 σ
        sigma = pipe.scheduler.sigmas[progress_id].to(device=pipe.device, dtype=pipe.torch_dtype)
        sigma_view = sigma.view(1, 1, 1, 1, 1)

        x = inputs_shared["latents"]

        # 预测 x0_high
        x0_pred = predict_x0_from_v(x, v_model, sigma_view)

        # reference x0_ref = phi(x0_low_pred[progress_id])
        x0_low_i = x0_low_list[progress_id].to(device=pipe.device, dtype=pipe.torch_dtype)
        x0_ref = upsample_latents_bcthw(x0_low_i, high_latent_h, high_latent_w, mode=upsample_mode)

        # ---- Direction alignment（低频替换）----
        # x0_aligned = x0_pred + alpha * (LP(x0_ref) - LP(x0_pred))
        lp_ref = lowfreq_bcthw(x0_ref, cutoff=lp_cutoff, order=lp_order)
        lp_pred = lowfreq_bcthw(x0_pred, cutoff=lp_cutoff, order=lp_order)
        x0_aligned = x0_pred + alpha * (lp_ref - lp_pred)

        # 转回 v（保持 scheduler 一致）
        v = v_from_x0(x, x0_aligned, sigma_view)

        # ---- Acceleration alignment（速度差分对齐）----
        # 在你的 scheduler 定义下：v_ref = x1_high - x0_ref = noise_high - x0_ref
        v_ref = noise_high - x0_ref

        if v_prev is not None and vref_prev is not None:
            dv = v - v_prev
            dv_ref = v_ref - vref_prev
            v = v + beta * (dv_ref - dv)

        v_prev = v
        vref_prev = v_ref

        # scheduler step
        inputs_shared["latents"] = pipe.scheduler.step(v, ts_cpu, x)

        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

    return inputs_shared["latents"]
