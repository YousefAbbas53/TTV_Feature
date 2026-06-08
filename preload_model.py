import os
import sys
from pathlib import Path

# Find the Wan2GP repository directory
root_dir = Path(__file__).resolve().parent
wan_dir = root_dir / "Wan2GP"
if not wan_dir.exists():
    wan_dir = root_dir / "Wan2GP-main"

if not wan_dir.exists():
    print("Error: Could not find Wan2GP directory.")
    sys.exit(1)

# Change working directory and add to system path
os.chdir(wan_dir)
sys.path.insert(0, str(wan_dir))

# Mock command-line arguments for wgp parser
sys.argv = ["wgp.py"]

# Mock CUDA checks to allow preloading in CPU-only build environments
import torch
torch.cuda.is_available = lambda: False

import wgp

print(f"Preloading Wan 1.3B model and shared dependencies inside: {wan_dir}...")

# 1. Download model weights for wan_t2v_1_3b
model_url = "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/wan2.1_text2video_1.3B_mbf16.safetensors"
wgp.download_models(
    model_filename=model_url,
    model_type="t2v_1.3B",
    module_type=None,
    submodel_no=1
)

# 2. Download dependencies (Text Encoder, VAE, open-clip, open_clip_config, etc.)
wgp.download_models(
    model_filename="",
    model_type="t2v_1.3B",
    module_type="",
    submodel_no=-1
)

print("Preloading complete!")
