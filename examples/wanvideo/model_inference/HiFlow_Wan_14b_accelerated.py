"""
HiFlow 高分辨率视频生成 + 动态帧率/动态分辨率加速示例

此脚本展示如何在 HiFlow Stage2 中启用：
1. 动态帧率（Dynamic Frame Rate）：早期步骤使用稀疏帧采样
2. 动态分辨率（Dynamic Resolution）：逐步提升空间分辨率

注意：这些加速功能默认关闭，可以根据需要启用
"""

import os
import glob
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.base_pipeline import PipelineUnit

class HiFlowConfigUnit(PipelineUnit):
    def __init__(self, **cfg):
        super().__init__(input_params=())
        self.cfg = cfg

    def process(self, pipe, **kwargs):
        return dict(self.cfg)

# ============ 配置 ============
BASE = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/home/rentianhao-20251020/data2/hiflow/checkpoints")

# 14B DiT 是分片权重（6 个 safetensors），必须传入"文件列表"
dit_paths = sorted(glob.glob(f"{BASE}/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-*-of-*.safetensors"))
print(f"[HiFlow_Wan_14b_accelerated] Found {len(dit_paths)} DiT shards under: {BASE}/Wan-AI/Wan2.1-T2V-14B/")
for p in dit_paths:
    print(f"  - {p}")
assert len(dit_paths) == 6, (
    "Expected 6 Wan2.1-T2V-14B DiT shard files, but got "
    f"{len(dit_paths)}. Please check BASE (DIFFSYNTH_MODEL_BASE_PATH) and downloaded weights."
)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=dit_paths),
        ModelConfig(path=f"{BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path=f"{BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"),
    ],
    tokenizer_config=ModelConfig(path=f"{BASE}/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl"),
    redirect_common_files=False,
)

# ============ 分辨率设置 ============
low_w, low_h = 1280, 720      # 低分辨率先验（720p）
high_w, high_h = 2080, 1200   # 目标高分辨率
num_frames = 33
num_inference_steps = 50

# 计算训练基准 token 序列长度
up = pipe.vae.upsampling_factor  # 8
pT, pH, pW = pipe.dit.patch_size
train_lat_h = low_h // up   # 720 / 8 = 90
train_lat_w = low_w // up   # 1280 / 8 = 160
train_seq_len = (num_frames // pT) * (train_lat_h // pH) * (train_lat_w // pW)

# NTK 外推因子
scale_h = high_h / low_h
scale_w = high_w / low_w
ntk_factor = max(scale_h, scale_w)

# ============ HiFlow 参数 ============
pipe.units.insert(0, HiFlowConfigUnit(
    use_hiflow=True,
    hiflow_low_height=low_h,
    hiflow_low_width=low_w,
    hiflow_tau_ratio=0.25,
    hiflow_alpha=0.6,
    hiflow_beta=0.4,
    hiflow_lp_cutoff=0.08,
    hiflow_lp_order=2,
    ntk_factor_h=ntk_factor,
    ntk_factor_w=ntk_factor,
))

# ============ 加速配置 ============
# 动态帧率配置
# 格式："start-end:stride,..." 
# stride=1 表示全帧，stride=3 表示每3帧采样一次
# 示例："0-22:3,23-49:1" 表示前22步用稀疏帧，后面用全帧
ENABLE_DYNAMIC_FRAMERATE = True
FRAMERATE_SCHEDULE = "0-22:3,23-49:1"  # 根据 tau_ratio=0.25，Stage2 从约第12步开始

# 动态分辨率配置
# res_rate_list: 每个阶段的分辨率比例（相对最终高分辨率）
# res_step_list: 每个阶段开始的步数（相对 tau_index）
ENABLE_DYNAMIC_RES = True
RES_RATE_LIST = [0.5, 0.75, 1.0]  # 50% -> 75% -> 100%
RES_STEP_LIST = [0, 12, 24]       # 相对 tau_index 的步数

# ============ 生成 ============
video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。阳光洒在它身上，毛发柔软而闪亮。背景是开阔草地和蓝天白云，中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    height=high_h,
    width=high_w,
    num_frames=num_frames,
    target_height=high_h,
    target_width=high_w,
    text_duplication=True,
    train_latent_h=train_lat_h,
    train_latent_w=train_lat_w,
    train_seq_len=train_seq_len,
    seed=42,
    tiled=True,
    num_inference_steps=num_inference_steps,
    cfg_scale=5.0,
    # ==== profiling ====
    profile_timings=True,
    profile_sample_every=5,
    # ==== 动态帧率加速 ====
    hiflow_enable_dynamic_framerate=ENABLE_DYNAMIC_FRAMERATE,
    hiflow_framerate_schedule=FRAMERATE_SCHEDULE,
    hiflow_framerate_interpolation_mode="trilinear",
    hiflow_framerate_keep_boundary=True,
    # ==== 动态分辨率加速 ====
    hiflow_enable_dynamic_res=ENABLE_DYNAMIC_RES,
    hiflow_res_rate_list=RES_RATE_LIST,
    hiflow_res_step_list=RES_STEP_LIST,
    hiflow_res_upsample_mode="bilinear",
)

save_video(video, "video_14b_2080x1200_accelerated.mp4", fps=15, quality=5)

print("==== pipe.last_timings ====")
print(pipe.last_timings)

print("\n==== 加速配置摘要 ====")
print(f"动态帧率: {'启用' if ENABLE_DYNAMIC_FRAMERATE else '关闭'}")
if ENABLE_DYNAMIC_FRAMERATE:
    print(f"  - 调度: {FRAMERATE_SCHEDULE}")
print(f"动态分辨率: {'启用' if ENABLE_DYNAMIC_RES else '关闭'}")
if ENABLE_DYNAMIC_RES:
    print(f"  - 分辨率比例: {RES_RATE_LIST}")
    print(f"  - 切换步数: {RES_STEP_LIST}")
