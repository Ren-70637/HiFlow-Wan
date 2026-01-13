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
'''
# 加载 14B 模型
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/umt5-xxl/"),
)'''
BASE = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/home/rentianhao-20251020/data2/hiflow/checkpoints")

# 14B DiT 是分片权重（6 个 safetensors），必须传入“文件列表”，不能把目录直接传给 ModelConfig(path=...)
dit_paths = sorted(glob.glob(f"{BASE}/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-*-of-*.safetensors"))
print(f"[HiFlow_Wan_14b] Found {len(dit_paths)} DiT shards under: {BASE}/Wan-AI/Wan2.1-T2V-14B/")
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
        # DiT (must be file list)
        ModelConfig(path=dit_paths),
        # T5 + VAE (single file)
        ModelConfig(path=f"{BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path=f"{BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"),
    ],
    tokenizer_config=ModelConfig(path=f"{BASE}/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl"),
    redirect_common_files=False,  # 可选：禁用重定向
)

# 低分辨率先验（720p 原生）
low_w, low_h = 1280, 720
# 目标高分辨率
high_w, high_h = 2080, 1200
num_frames = 33

# 计算训练基准 token 序列长度
up = pipe.vae.upsampling_factor  # 8
pT, pH, pW = pipe.dit.patch_size
train_lat_h = low_h // up   # 720 / 8 = 90
train_lat_w = low_w // up   # 1280 / 8 = 160
train_seq_len = (num_frames // pT) * (train_lat_h // pH) * (train_lat_w // pW)

# NTK 外推因子（根据放大倍数）
scale_h = high_h / low_h  # 1200/720 ≈ 1.67
scale_w = high_w / low_w  # 2080/1280 ≈ 1.625
ntk_factor = max(scale_h, scale_w)  # ≈ 1.67

# 注入 HiFlow 参数
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

video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。阳光洒在它身上，毛发柔软而闪亮。背景是开阔草地和蓝天白云，中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    height=high_h,          # 1200
    width=high_w,           # 2080
    num_frames=num_frames,  # 33
    target_height=high_h,
    target_width=high_w,
    text_duplication=True,
    train_latent_h=train_lat_h,
    train_latent_w=train_lat_w,
    train_seq_len=train_seq_len,
    seed=42,
    tiled=True,
    num_inference_steps=50,
    cfg_scale=5.0,
    # ==== profiling ====
    profile_timings=True,        # 开启分段计时（self.last_timings）
    profile_sample_every=5,      # 每5步采样一次 step 级耗时；想更细改成1
)

save_video(video, "video_14b_2080x1200_42.mp4", fps=15, quality=5)

print("==== pipe.last_timings ====")
print(pipe.last_timings)