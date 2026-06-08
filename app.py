from pathlib import Path
import json
import os
import tempfile
import time
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from analyzer import analyze_book, analyze_raw_text, run_analysis
from config import settings
from model_profiles import MODEL_PRESETS
from prompt_enhancer import ADAPTER_DIR, BASE_MODEL, MISTRAL_ENABLED, _load_model_bundle, mistral_available
from text_pipeline import build_final_video_prompts
from video_generator import generate_video_from_prompt_items, generate_video_from_prompts
from video_service import VideoGenerationError, VideoGenerator


app = FastAPI(
    title="Video Generation API",
    version="1.0.0",
    description="Generate and return MP4 files using switchable local video backends.",
)

generator = VideoGenerator(settings)
previews_dir = settings.temp_dir / "previews"
PREVIEW_TTL_SECONDS = 24 * 60 * 60


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _ensure_preview_dir() -> None:
    previews_dir.mkdir(parents=True, exist_ok=True)


def _cleanup_old_previews() -> None:
    _ensure_preview_dir()
    cutoff = time.time() - PREVIEW_TTL_SECONDS
    for preview_path in previews_dir.glob("*.json"):
        try:
            if preview_path.stat().st_mtime < cutoff:
                preview_path.unlink()
        except OSError:
            continue


def _save_preview(items: list[dict], source_name: str) -> dict[str, object]:
    _cleanup_old_previews()
    _ensure_preview_dir()
    preview_id = uuid.uuid4().hex
    preview_path = previews_dir / f"{preview_id}.json"
    payload = {
        "preview_id": preview_id,
        "source_name": source_name,
        "scene_count": len(items),
        "scenes": items,
    }
    with preview_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def _load_preview(preview_id: str) -> dict[str, object]:
    _cleanup_old_previews()
    _ensure_preview_dir()
    preview_path = previews_dir / f"{preview_id}.json"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found.")
    with preview_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _public_video_payload(request: Request, video_path: str | Path) -> dict[str, str]:
    resolved_path = Path(video_path).resolve()
    filename = resolved_path.name
    return {
        "filename": filename,
        "video_path": str(resolved_path),
        "download_url": str(request.url_for("download_output_file", filename=filename)),
    }


def _runtime_status(load_mistral: bool = False) -> dict[str, object]:
    current_preset = MODEL_PRESETS.get(settings.video_model_preset)
    wan_status = {
        "repo_dir": str(settings.wan_repo_dir),
        "repo_exists": settings.wan_repo_dir.exists(),
        "headless_script": str(settings.wan_headless_script),
        "headless_script_exists": settings.wan_headless_script.exists(),
        "template_path": str(settings.wan_template_path),
        "template_exists": settings.wan_template_path.exists(),
    }

    mistral_status: dict[str, object] = {
        "enabled": MISTRAL_ENABLED,
        "base_model": BASE_MODEL,
        "adapter_dir": str(ADAPTER_DIR),
        "adapter_exists": ADAPTER_DIR.exists(),
        "runtime_ready": mistral_available(),
    }

    try:
        import torch
        mistral_status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            mistral_status["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        mistral_status["cuda_available"] = False
        mistral_status["torch_error"] = str(exc)

    if load_mistral:
        try:
            _load_model_bundle()
            mistral_status["load_check"] = "ok"
        except Exception as exc:
            mistral_status["load_check"] = "failed"
            mistral_status["load_error"] = str(exc)

    backend_ok = (
        wan_status["repo_exists"]
        and wan_status["headless_script_exists"]
        and wan_status["template_exists"]
    )

    overall_ok = backend_ok and (not MISTRAL_ENABLED or mistral_status["runtime_ready"])

    return {
        "ok": bool(overall_ok),
        "video_backend": "wan",
        "video_model_preset": settings.video_model_preset,
        "model_defaults": {
            "width": settings.default_width,
            "height": settings.default_height,
            "num_frames": settings.default_num_frames,
            "fps": settings.default_fps,
            "steps": settings.default_steps,
            "cfg": settings.default_cfg,
            "model_type": settings.default_model_type,
            "description": current_preset.description if current_preset else None,
        },
        "wan2gp": wan_status,
        "mistral": mistral_status,
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="The text prompt to render.")
    negative_prompt: str | None = Field(default=None, description="Optional negative prompt passed into the renderer.")
    width: int | None = Field(default=None, ge=256, le=1920)
    height: int | None = Field(default=None, ge=256, le=1920)
    num_frames: int | None = Field(default=None, ge=8, le=256)
    fps: int | None = Field(default=None, ge=1, le=60)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=1.0, le=20.0)
    seed: int | None = Field(default=None)


class GenerateFromTextRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Book excerpt or story text to split into scenes.")
    max_scenes: int = Field(default=3, ge=1, le=5, description="Safety limit for generated scenes per request.")
    scene_window: int = Field(default=3, ge=1, le=8, description="Number of paragraphs grouped into one scene.")
    width: int | None = Field(default=None, ge=256, le=1920)
    height: int | None = Field(default=None, ge=256, le=1920)
    num_frames: int | None = Field(default=None, ge=8, le=256)
    fps: int | None = Field(default=None, ge=1, le=60)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=1.0, le=20.0)
    seed: int | None = Field(default=None)


class PreviewFromTextRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Book excerpt or story text to split into scenes.")
    max_scenes: int | None = Field(default=None, ge=1, le=20, description="Optional cap on extracted scenes.")
    scene_window: int = Field(default=3, ge=1, le=8, description="Number of paragraphs grouped into one scene.")


class GenerateSelectedScenesRequest(BaseModel):
    preview_id: str = Field(..., min_length=8)
    scene_ids: list[str] = Field(..., min_length=1, description="Scene IDs chosen by the user from preview output.")
    width: int | None = Field(default=None, ge=256, le=1920)
    height: int | None = Field(default=None, ge=256, le=1920)
    num_frames: int | None = Field(default=None, ge=8, le=256)
    fps: int | None = Field(default=None, ge=1, le=60)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=1.0, le=20.0)
    seed: int | None = Field(default=None)


class GenerateSelectedScenesJsonRequest(GenerateSelectedScenesRequest):
    output_name: str | None = Field(default=None, description="Optional output MP4 filename without path traversal.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/full")
def health_full(load_mistral: bool = Query(default=False)) -> dict[str, object]:
    return _runtime_status(load_mistral=load_mistral)


@app.get("/model-info")
def model_info() -> dict[str, object]:
    current = MODEL_PRESETS.get(settings.video_model_preset)
    return {
        "current_backend": "wan",
        "current_preset": settings.video_model_preset,
        "current_description": current.description if current else None,
        "current_defaults": {
            "width": settings.default_width,
            "height": settings.default_height,
            "num_frames": settings.default_num_frames,
            "fps": settings.default_fps,
            "steps": settings.default_steps,
            "cfg": settings.default_cfg,
            "model_type": settings.default_model_type,
        },
        "available_presets": {
            name: {
                "backend": "wan",
                "description": preset.description,
                "model_type": preset.model_type,
                "defaults": {
                    "width": preset.width,
                    "height": preset.height,
                    "num_frames": preset.num_frames,
                    "fps": preset.fps,
                    "steps": preset.steps,
                    "cfg": preset.cfg,
                },
            }
            for name, preset in MODEL_PRESETS.items()
        },
    }


@app.get("/files/{filename}", name="download_output_file")
def download_output_file(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = (settings.outputs_dir / safe_name).resolve()
    outputs_root = settings.outputs_dir.resolve()
    if file_path.parent != outputs_root or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=safe_name,
    )


@app.on_event("startup")
def startup_log() -> None:
    status = _runtime_status(load_mistral=False)
    print(
        "[startup] wan2gp_ok={wan_ok} mistral_enabled={m_enabled} mistral_ready={m_ready}".format(
            wan_ok=(
                status["wan2gp"]["repo_exists"]
                and status["wan2gp"]["headless_script_exists"]
                and status["wan2gp"]["template_exists"]
            ),
            m_enabled=status["mistral"]["enabled"],
            m_ready=status["mistral"]["runtime_ready"],
        )
    )


@app.post("/generate", dependencies=[Depends(require_api_key)])
def generate_video(request: GenerateRequest) -> FileResponse:
    try:
        result = generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            fps=request.fps,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
        )
    except VideoGenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=result.video_path,
        media_type="video/mp4",
        filename=result.filename,
    )


@app.post("/generate-from-text", dependencies=[Depends(require_api_key)])
def generate_from_text(request: GenerateFromTextRequest) -> dict[str, object]:
    scene_prompts = build_final_video_prompts(
        request.text,
        max_scenes=request.max_scenes,
        scene_window=request.scene_window,
    )
    if not scene_prompts:
        raise HTTPException(status_code=400, detail="No valid scenes could be extracted from the text.")

    videos: list[dict[str, str]] = []
    for index, scene in enumerate(scene_prompts):
        try:
            result = generator.generate(
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                steps=request.steps,
                cfg=request.cfg,
                seed=(request.seed + index) if request.seed is not None else None,
            )
        except VideoGenerationError as exc:
            raise HTTPException(status_code=500, detail=f"Scene {scene.scene_id} failed: {exc}") from exc

        videos.append(
            {
                "scene_id": scene.scene_id,
                "scene_excerpt": scene.scene_excerpt,
                "prompt": scene.prompt,
                "video_path": result.video_path,
                "filename": result.filename,
            }
        )

    return {
        "scene_count": len(scene_prompts),
        "videos": videos,
    }


@app.post("/preview-from-text", dependencies=[Depends(require_api_key)])
def preview_from_text(request: PreviewFromTextRequest) -> dict[str, object]:
    scene_prompts = analyze_raw_text(
        request.text,
        max_scenes=request.max_scenes,
        scene_window=request.scene_window,
    )
    if not scene_prompts:
        raise HTTPException(status_code=400, detail="No valid scenes could be extracted from the text.")
    return _save_preview(scene_prompts, source_name="text")


@app.post("/generate-from-file", dependencies=[Depends(require_api_key)])
async def generate_from_file(
    file: UploadFile = File(...),
    max_scenes: int = 3,
    scene_window: int = 3,
) -> FileResponse:
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.temp_dir) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    prompts_path = settings.temp_dir / f"{tmp_path.stem}_prompts.json"
    video_path = settings.outputs_dir / f"{tmp_path.stem}.mp4"

    try:
        prompts_json = run_analysis(
            tmp_path,
            output_json_path=prompts_path,
            max_scenes=max_scenes,
            scene_window=scene_window,
        )
        final_video = generate_video_from_prompts(prompts_json, video_path)
        return FileResponse(
            path=final_video,
            media_type="video/mp4",
            filename=f"generated_{Path(file.filename or 'video').stem}.mp4",
        )
    except (VideoGenerationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


@app.post("/preview-from-file", dependencies=[Depends(require_api_key)])
async def preview_from_file(
    file: UploadFile = File(...),
    max_scenes: int | None = None,
    scene_window: int = 3,
) -> dict[str, object]:
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.temp_dir) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        scene_prompts = analyze_book(
            tmp_path,
            max_scenes=max_scenes,
            scene_window=scene_window,
        )
        if not scene_prompts:
            raise HTTPException(status_code=400, detail="No valid scenes could be extracted from the file.")
        return _save_preview(scene_prompts, source_name=file.filename or "upload")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


@app.post("/generate-selected-scenes", dependencies=[Depends(require_api_key)])
def generate_selected_scenes(request: GenerateSelectedScenesRequest) -> dict[str, object]:
    preview = _load_preview(request.preview_id)
    all_scenes = preview.get("scenes", [])
    if not isinstance(all_scenes, list) or not all_scenes:
        raise HTTPException(status_code=400, detail="Preview contains no scenes.")

    selected_ids = set(request.scene_ids)
    selected_scenes = [
        scene for scene in all_scenes
        if isinstance(scene, dict) and scene.get("scene_id") in selected_ids
    ]
    if not selected_scenes:
        raise HTTPException(status_code=400, detail="No matching scenes found for the provided scene_ids.")

    found_ids = {str(scene.get("scene_id")) for scene in selected_scenes}
    missing_ids = [scene_id for scene_id in request.scene_ids if scene_id not in found_ids]
    if missing_ids:
        raise HTTPException(status_code=400, detail=f"Unknown scene_ids: {', '.join(missing_ids)}")

    videos: list[dict[str, str]] = []
    for index, scene in enumerate(selected_scenes):
        try:
            result = generator.generate(
                prompt=str(scene.get("prompt", "")),
                negative_prompt=scene.get("negative_prompt"),
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                steps=request.steps,
                cfg=request.cfg,
                seed=(request.seed + index) if request.seed is not None else None,
            )
        except VideoGenerationError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Scene {scene.get('scene_id', 'unknown')} failed: {exc}",
            ) from exc

        videos.append(
            {
                "scene_id": str(scene.get("scene_id", "")),
                "scene_excerpt": str(scene.get("scene_excerpt", "")),
                "prompt": str(scene.get("prompt", "")),
                "video_path": result.video_path,
                "filename": result.filename,
            }
        )

    return {
        "preview_id": request.preview_id,
        "selected_scene_count": len(selected_scenes),
        "videos": videos,
    }


@app.post("/generate-selected-scenes-json", dependencies=[Depends(require_api_key)])
def generate_selected_scenes_json(
    payload: GenerateSelectedScenesJsonRequest,
    request: Request,
) -> dict[str, object]:
    preview = _load_preview(payload.preview_id)
    all_scenes = preview.get("scenes", [])
    if not isinstance(all_scenes, list) or not all_scenes:
        raise HTTPException(status_code=400, detail="Preview contains no scenes.")

    selected_ids = set(payload.scene_ids)
    selected_scenes = [
        scene for scene in all_scenes
        if isinstance(scene, dict) and scene.get("scene_id") in selected_ids
    ]
    if not selected_scenes:
        raise HTTPException(status_code=400, detail="No matching scenes found for the provided scene_ids.")

    found_ids = {str(scene.get("scene_id")) for scene in selected_scenes}
    missing_ids = [scene_id for scene_id in payload.scene_ids if scene_id not in found_ids]
    if missing_ids:
        raise HTTPException(status_code=400, detail=f"Unknown scene_ids: {', '.join(missing_ids)}")

    merged_items: list[dict[str, object]] = []
    for index, scene in enumerate(selected_scenes):
        item = dict(scene)
        if payload.width is not None:
            item["width"] = payload.width
        if payload.height is not None:
            item["height"] = payload.height
        if payload.num_frames is not None:
            item["num_frames"] = payload.num_frames
        if payload.fps is not None:
            item["fps"] = payload.fps
        if payload.steps is not None:
            item["steps"] = payload.steps
        if payload.cfg is not None:
            item["cfg"] = payload.cfg
        if payload.seed is not None:
            item["seed"] = payload.seed + index
        merged_items.append(item)

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    requested_name = (payload.output_name or f"selected_{payload.preview_id[:8]}.mp4").strip()
    safe_name = Path(requested_name).name
    if not safe_name.lower().endswith(".mp4"):
        safe_name = f"{safe_name}.mp4"
    output_path = settings.outputs_dir / safe_name

    try:
        final_video_path = generate_video_from_prompt_items(merged_items, output_path)
    except (VideoGenerationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ok": True,
        "preview_id": payload.preview_id,
        "selected_scene_count": len(selected_scenes),
        "selected_scene_ids": [str(scene.get("scene_id", "")) for scene in selected_scenes],
        "video": _public_video_payload(request, final_video_path),
    }
