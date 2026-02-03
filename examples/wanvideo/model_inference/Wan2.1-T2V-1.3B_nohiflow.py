import os
import glob
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# 使用命令行指定 GPU，例如：
# CUDA_VISIBLE_DEVICES=5 python Wan2.1-T2V-1.3B_nohiflow.py
BASE = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/home/rentianhao-20251020/data2/hiflow/checkpoints")

# 1.3B DiT 可能是单文件或多分片，统一用 glob 获取
dit_paths = sorted(glob.glob(f"{BASE}/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model*.safetensors"))
print(f"[Wan2.1-T2V-1.3B] Found {len(dit_paths)} DiT shard(s) under: {BASE}/Wan-AI/Wan2.1-T2V-1.3B/")
for p in dit_paths:
    print(f"  - {p}")
assert len(dit_paths) >= 1, (
    "Expected Wan2.1-T2V-1.3B DiT weight files, but got 0. "
    "Please check BASE (DIFFSYNTH_MODEL_BASE_PATH) and downloaded weights."
)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # DiT (file list)
        ModelConfig(path=dit_paths),
        # T5 + VAE (single file)
        ModelConfig(path=f"{BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path=f"{BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"),
    ],
    tokenizer_config=ModelConfig(path=f"{BASE}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
    redirect_common_files=False,
)

# Text-to-video (no HiFlow)
# 1280x720
# 1920x1088
# 2080x1200
# 3380x1920
video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    height=1920,
    width=3380,
    num_frames=81,
    seed=0,
    tiled=True,
)
output_path = "/home/rentianhao-20251020/DiffSynth-Studio/video_Wan2.1-T2V-1.3B_3380x1920_81f.mp4"
save_video(video, output_path, fps=15, quality=5)

