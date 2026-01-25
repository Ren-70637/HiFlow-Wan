import os
import glob
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
BASE = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/home/rentianhao-20251020/data2/hiflow/checkpoints")

# 14B DiT 是分片权重（6 个 safetensors），必须传入“文件列表”
dit_paths = sorted(glob.glob(f"{BASE}/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-*-of-*.safetensors"))
print(f"[Wan2.1-T2V-14B] Found {len(dit_paths)} DiT shards under: {BASE}/Wan-AI/Wan2.1-T2V-14B/")
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
    redirect_common_files=False,
)

# Text-to-video
video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。阳光洒在它身上，毛发柔软而闪亮。背景是开阔草地和蓝天白云，中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    height=2176,
    width=3840,
    num_frames=33,
    seed=0,
    tiled=True,
)
save_video(video, "video_Wan2.1-T2V-14B_3840x2176_33f.mp4", fps=15, quality=5)
