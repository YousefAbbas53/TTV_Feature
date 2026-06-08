#!/usr/bin/env bash
set -euo pipefail

PRESET="wan_t2v_1_3b"
PROJECT_DIR="${PROJECT_DIR:-/workspace/LITVISION2}"
WAN_DIR="${WAN_DIR:-/workspace/Wan2GP}"
API_PORT="${API_PORT:-8000}"
API_KEY="${VIDEO_API_KEY:-change-me}"

if [ "${1:-wan_t2v_1_3b}" != "wan_t2v_1_3b" ]; then
  echo "[WARN] Only wan_t2v_1_3b is supported. Ignoring requested preset: ${1:-}"
fi

echo "[1/6] System packages"
apt-get update -qq
apt-get install -y -qq git git-lfs ffmpeg aria2 >/dev/null

echo "[2/6] Python project requirements"
cd "$PROJECT_DIR"
python -m pip install --upgrade pip wheel >/dev/null
python -m pip install "setuptools<82" >/dev/null
python -m pip install -r requirements.txt

echo "[3/6] Wan2GP checkout"
if [ -d "$WAN_DIR/.git" ]; then
  git -C "$WAN_DIR" pull --ff-only
else
  git clone https://github.com/deepbeepmeep/Wan2GP.git "$WAN_DIR"
fi
cp "$PROJECT_DIR/wgp_headless_t2v_fix.py" "$WAN_DIR/wgp_headless_t2v_fix.py"

echo "[4/6] Wan2GP requirements"
WAN_REQ="/tmp/wan2gp_requirements_compat.txt"
awk '
  # rembg is only needed for background-removal tools, not headless T2V.
  # Recent rembg releases require numpy>=2.3, while Wan2GP pins numpy==2.1.2.
  !/^rembg(\[[^]]+\])?==/ { print }
' "$WAN_DIR/requirements.txt" > "$WAN_REQ"
python -m pip install -r "$WAN_REQ"

echo "[5/6] Runtime environment"
mkdir -p "$PROJECT_DIR/outputs" "$PROJECT_DIR/tmp" "$PROJECT_DIR/books" "$WAN_DIR/ckpts" /workspace/hf-cache

export VIDEO_API_KEY="$API_KEY"
export VIDEO_MODEL_PRESET="$PRESET"
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache
export WAN_REPO_DIR="$WAN_DIR"
export WAN_HEADLESS_SCRIPT="$WAN_DIR/wgp_headless_t2v_fix.py"
export WAN_TEMPLATE_PATH="$WAN_DIR/defaults/t2v.json"
export VIDEO_OUTPUT_DIR="$PROJECT_DIR/outputs"
export VIDEO_TEMP_DIR="$PROJECT_DIR/tmp"
export MISTRAL_ENABLED="${MISTRAL_ENABLED:-0}"
export VIDEO_MODEL_TYPE=t2v_1.3B
export VIDEO_WIDTH=768
export VIDEO_HEIGHT=432
export VIDEO_NUM_FRAMES=80
export VIDEO_FPS=16
export VIDEO_STEPS=40
export VIDEO_CFG=4.8

echo "[6/6] Health preflight"
python - <<'PY'
from config import settings
print("backend: wan")
print("preset:", settings.video_model_preset)
print("model_type:", settings.default_model_type)
print("wan_repo:", settings.wan_repo_dir, settings.wan_repo_dir.exists())
print("headless:", settings.wan_headless_script, settings.wan_headless_script.exists())
print("template:", settings.wan_template_path, settings.wan_template_path.exists())
PY

echo "[OK] Starting API on port $API_PORT"
exec uvicorn app:app --host 0.0.0.0 --port "$API_PORT"
