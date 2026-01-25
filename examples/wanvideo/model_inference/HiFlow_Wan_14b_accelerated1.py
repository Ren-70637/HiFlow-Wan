'''
HiFlow 高分辨率视频生成 + 动态分辨率加速示例（两阶段版本）
- Stage 2-B (75%): 28 步
- Stage 2-C (100%): 10 步
'''

import os
import glob
from datetime import datetime
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

dit_paths = sorted(glob.glob(f"{BASE}/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-*-of-*.safetensors"))
print(f"[HiFlow] Found {len(dit_paths)} DiT shards")
assert len(dit_paths) == 6

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
low_w, low_h = 1280, 720
high_w, high_h = 2080, 1200
num_frames = 81
num_inference_steps = 50

up = pipe.vae.upsampling_factor
pT, pH, pW = pipe.dit.patch_size
train_lat_h = low_h // up
train_lat_w = low_w // up
train_seq_len = (num_frames // pT) * (train_lat_h // pH) * (train_lat_w // pW)

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

# ============ 两阶段动态分辨率配置 ============
ENABLE_DYNAMIC_RES = True
RES_RATE_LIST = [0.8, 1.0]  # 两阶段：75% -> 100%
RES_STEP_LIST = [0, 20]       # 相对 tau_index：0 开始 75%，28 开始 100%

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
    profile_timings=True,
    profile_sample_every=5,
    # ==== 两阶段动态分辨率 ====
    hiflow_enable_dynamic_res=ENABLE_DYNAMIC_RES,
    hiflow_res_rate_list=RES_RATE_LIST,
    hiflow_res_step_list=RES_STEP_LIST,
    hiflow_res_upsample_mode="bilinear",
)

output_dir = "/home/rentianhao-20251020/DiffSynth-Studio"
video_prefix = "video_14b_2080x1200_2stage_"
existing = glob.glob(os.path.join(output_dir, f"{video_prefix}*.mp4"))
indices = []
for path in existing:
    name = os.path.basename(path)
    suffix = name.replace(video_prefix, "").replace(".mp4", "")
    if suffix.isdigit():
        indices.append(int(suffix))
next_idx = max(indices) + 1 if indices else 1
video_path = os.path.join(output_dir, f"{video_prefix}{next_idx}.mp4")

save_video(video, video_path, fps=15, quality=5)

print("==== pipe.last_timings ====")
print(pipe.last_timings)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
profile_path = os.path.join(output_dir, f"profile_14b_2080x1200_{timestamp}.log")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write("==== pipe.last_timings ====\n")
    f.write(f"{pipe.last_timings}\n")
    f.write("\n==== 两阶段动态分辨率配置 ====\n")
    f.write("Stage 2-B (75%): progress_id 12-39, 共 28 步\n")
    f.write("Stage 2-C (100%): progress_id 40-49, 共 10 步\n")
    f.write(f"video_path={video_path}\n")

print("\n==== 两阶段动态分辨率配置 ====")
print(f"Stage 2-B (75%): progress_id 12-39, 共 28 步")
print(f"Stage 2-C (100%): progress_id 40-49, 共 10 步")
print(f"\nSaved video: {video_path}")
print(f"Saved profile: {profile_path}")