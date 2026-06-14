import io
import json
import os
import sys
import shutil
import time
import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import runpod
import requests

# Ensure workspace project directory is in system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import settings
from video_generator import generate_video_from_prompt_items


# ---------------------------------------------------------------------------
# S3 / R2 helpers
# ---------------------------------------------------------------------------

def _get_s3_client_and_bucket():
    """Return (boto3_s3_client, bucket_name, public_url_prefix) from env vars."""
    bucket_name = (
        os.getenv("S3_BUCKET_NAME")
        or os.getenv("AWS_BUCKET_NAME")
        or os.getenv("BUCKET_NAME")
    )
    if not bucket_name:
        raise ValueError("Cloud storage bucket name (S3_BUCKET_NAME) is not configured.")

    access_key = (
        os.getenv("S3_ACCESS_KEY_ID")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("R2_ACCESS_KEY_ID")
    )
    secret_key = (
        os.getenv("S3_SECRET_ACCESS_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or os.getenv("R2_SECRET_ACCESS_KEY")
    )
    endpoint_url = (
        os.getenv("S3_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL")
        or os.getenv("ENDPOINT_URL")
    )
    region_name = (
        os.getenv("S3_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_REGION")
    )
    public_url_prefix = (
        os.getenv("S3_PUBLIC_URL_PREFIX")
        or os.getenv("PUBLIC_URL_PREFIX")
    )

    client_kwargs = {}
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if region_name:
        client_kwargs["region_name"] = region_name

    s3 = boto3.client("s3", **client_kwargs)
    return s3, bucket_name, public_url_prefix, endpoint_url, region_name


def _build_public_url(object_name: str, public_url_prefix, endpoint_url, bucket_name, region_name):
    """Construct the public URL for an uploaded object."""
    if public_url_prefix:
        return f"{public_url_prefix.rstrip('/')}/{object_name}"
    elif endpoint_url:
        return f"{endpoint_url.rstrip('/')}/{bucket_name}/{object_name}"
    else:
        region = region_name or "us-east-1"
        return f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"


# ---------------------------------------------------------------------------
# Preview cloud storage (stateless — stored in R2/S3)
# ---------------------------------------------------------------------------

def upload_preview_to_cloud(preview_id: str, preview_data: dict) -> None:
    """Serialize preview metadata and upload to previews/{preview_id}.json in R2/S3."""
    s3, bucket_name, *_ = _get_s3_client_and_bucket()
    object_key = f"previews/{preview_id}.json"
    body = json.dumps(preview_data, ensure_ascii=False, indent=2).encode("utf-8")

    print(f"Uploading preview metadata to s3://{bucket_name}/{object_key} ...")
    s3.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Preview {preview_id} saved to cloud storage.")


def download_preview_from_cloud(preview_id: str) -> dict:
    """Download and deserialize preview metadata from previews/{preview_id}.json.

    Raises ValueError with a clear message if the preview_id does not exist.
    """
    s3, bucket_name, *_ = _get_s3_client_and_bucket()
    object_key = f"previews/{preview_id}.json"

    print(f"Downloading preview metadata from s3://{bucket_name}/{object_key} ...")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404", "AccessDenied"):
            raise ValueError(
                f"Preview not found: '{preview_id}' does not exist in cloud storage. "
                f"It may have expired or the ID is incorrect."
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Video upload
# ---------------------------------------------------------------------------

def upload_to_cloud_storage(local_file_path: Path) -> str:
    """Upload final video to Cloudflare R2 or AWS S3 and return the public URL."""
    s3, bucket_name, public_url_prefix, endpoint_url, region_name = _get_s3_client_and_bucket()

    object_name = f"ttv/{uuid.uuid4().hex}_{local_file_path.name}"
    print(f"Uploading {local_file_path} to s3://{bucket_name}/{object_name}...")

    extra_args = {
        "ContentType": "video/mp4",
    }
    s3_acl = os.getenv("S3_ACL", "public-read")
    if s3_acl:
        extra_args["ACL"] = s3_acl

    s3.upload_file(
        Filename=str(local_file_path),
        Bucket=bucket_name,
        Key=object_name,
        ExtraArgs=extra_args,
    )

    final_url = _build_public_url(object_name, public_url_prefix, endpoint_url, bucket_name, region_name)
    print(f"Upload successful. URL: {final_url}")
    return final_url


# ---------------------------------------------------------------------------
# File download & input safeguards (unchanged)
# ---------------------------------------------------------------------------

def download_file_to_temp(url: str, suffix: str = "") -> str:
    """Download a remote HTTP/HTTPS file to the local temp directory for processing."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return url

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = settings.temp_dir / f"downloaded_{uuid.uuid4().hex}{suffix}"

    print(f"Downloading {url} to temporary path: {temp_file_path}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(temp_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return str(temp_file_path)


def apply_input_safeguards(d: dict):
    """Detect URLs sent in identification fields (safeguards) and download URLs locally."""
    if not isinstance(d, dict):
        return

    def is_url(v):
        return isinstance(v, str) and (v.startswith("http://") or v.startswith("https://"))

    # Map of ID/misplaced keys to actual target keys
    safeguard_redirects = {
        "video_id": ["video_url", "video_source"],
        "image_id": ["image_url", "image_start"],
        "prompt_id": ["prompt"],
    }

    # Apply redirects if backend sent a URL inside an ID field
    for bad_key, good_keys in safeguard_redirects.items():
        val = d.get(bad_key)
        if is_url(val):
            for good_key in good_keys:
                if not d.get(good_key):
                    print(f"Safeguard redirect: Found URL in '{bad_key}', copying to '{good_key}'")
                    d[good_key] = val

    # Automatically download any URLs to local disk so the Wan2GP backend can load them
    url_mappings = [
        ("video_url", "video_source"),
        ("image_url", "image_start"),
        ("video_source", "video_source"),
        ("image_start", "image_start"),
    ]
    for url_key, target_key in url_mappings:
        val = d.get(url_key)
        if is_url(val):
            suffix = ""
            # Extract suffix if available
            parsed_filename = val.split("/")[-1].split("?")[0]
            if "." in parsed_filename:
                suffix = "." + parsed_filename.split(".")[-1]

            try:
                local_path = download_file_to_temp(val, suffix)
                d[target_key] = local_path
                if url_key != target_key:
                    d[url_key] = local_path
            except Exception as e:
                print(f"Failed to download remote file from {url_key}: {e}")


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_input(job_input):
    """Parse RunPod job input and normalize it into a list of prompt scenes."""
    if isinstance(job_input, list):
        # Already a list of scene items
        for scene in job_input:
            apply_input_safeguards(scene)
        return job_input

    # Clone to avoid mutating original dictionary in-place
    job_input = dict(job_input)
    apply_input_safeguards(job_input)

    # 0. Check for raw book/story input (either as text or a remote URL)
    text_content = job_input.get("text") or job_input.get("book_text")
    book_url = job_input.get("book_url")

    if text_content or book_url:
        from analyzer import analyze_book, analyze_raw_text
        max_scenes = job_input.get("max_scenes", None)
        scene_window = job_input.get("scene_window", 3)

        # Resolve other parameters to pass to all scenes
        global_opts = {k: v for k, v in job_input.items() if k not in ("text", "book_text", "book_url", "max_scenes", "scene_window")}

        if book_url:
            suffix = ""
            parsed_filename = book_url.split("/")[-1].split("?")[0]
            if "." in parsed_filename:
                suffix = "." + parsed_filename.split(".")[-1]
            else:
                suffix = ".txt"

            print(f"Downloading book from URL: {book_url}")
            local_book_path = download_file_to_temp(book_url, suffix)
            try:
                scenes = analyze_book(local_book_path, max_scenes=max_scenes, scene_window=scene_window)
            finally:
                try:
                    os.unlink(local_book_path)
                except Exception:
                    pass
        else:
            print("Analyzing raw book text...")
            scenes = analyze_raw_text(text_content, max_scenes=max_scenes, scene_window=scene_window)

        if not scenes:
            raise ValueError("No valid scenes could be extracted from the book or text.")

        # Apply global options and safeguards
        for scene in scenes:
            for k, v in global_opts.items():
                if k not in scene:
                    scene[k] = v
            apply_input_safeguards(scene)
        return scenes

    # 1. Check for nested list of scenes (e.g. story mode / multiscene)
    if "scenes" in job_input and isinstance(job_input["scenes"], list):
        scenes = job_input["scenes"]
        global_opts = {k: v for k, v in job_input.items() if k != "scenes"}
        for scene in scenes:
            if isinstance(scene, dict):
                # Apply global defaults to scenes if not overridden locally
                for k, v in global_opts.items():
                    if k not in scene:
                        scene[k] = v
                apply_input_safeguards(scene)
        return scenes

    # 2. Check for alternative keys containing lists
    for key in ["prompt_items", "prompts"]:
        if key in job_input and isinstance(job_input[key], list):
            items = job_input[key]
            global_opts = {k: v for k, v in job_input.items() if k != key}
            for item in items:
                if isinstance(item, dict):
                    for k, v in global_opts.items():
                        if k not in item:
                            item[k] = v
                    apply_input_safeguards(item)
            return items

    # 3. Single scene/prompt job
    return [job_input]


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_local_files():
    """Wipe output and temp directories to conserve disk space on serverless worker."""
    print("Cleaning up local files and temporary folders...")
    for directory in [settings.outputs_dir, settings.temp_dir]:
        if directory.exists():
            for item in directory.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Error deleting local resource {item}: {e}")


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _normalize_scene_for_preview(scene: dict) -> dict:
    """Rename 'scene_id' → 'id' and keep only the fields the backend needs."""
    return {
        "id": str(scene.get("scene_id", scene.get("id", ""))),
        "scene_excerpt": str(scene.get("scene_excerpt", "")),
        "prompt": str(scene.get("prompt", "")),
    }


def _handle_preview(job_input: dict) -> dict:
    """action = 'preview': analyze the book/text and return scene metadata.

    The preview JSON is also uploaded to R2 so any future worker can retrieve it.
    """
    # Remove action key before passing to parse_input
    input_for_parsing = {k: v for k, v in job_input.items() if k != "action"}
    scenes = parse_input(input_for_parsing)

    preview_id = uuid.uuid4().hex
    normalized_scenes = [_normalize_scene_for_preview(s) for s in scenes]

    preview_payload = {
        "preview_id": preview_id,
        "scene_count": len(normalized_scenes),
        "scenes": normalized_scenes,
    }

    # Also store the *full* scene data (with prompts, negative_prompts, etc.)
    # in cloud so generate can use it later.
    cloud_data = {
        "preview_id": preview_id,
        "scene_count": len(scenes),
        "scenes": scenes,  # full scene dicts for rendering
    }
    upload_preview_to_cloud(preview_id, cloud_data)

    return preview_payload


def _handle_generate(job_input: dict) -> dict:
    """action = 'generate': render selected scenes and return the video URL.

    Reads optional overrides (steps, duration_seconds, fps, cfg, etc.)
    directly from job_input and merges them into each selected scene.
    """
    preview_id = job_input.get("preview_id")
    scene_ids = job_input.get("scene_ids")

    if not preview_id:
        raise ValueError("Missing required field: 'preview_id'.")
    if not scene_ids or not isinstance(scene_ids, list):
        raise ValueError("Missing or invalid field: 'scene_ids' (must be a non-empty list).")

    # Download the full preview metadata from R2
    preview_data = download_preview_from_cloud(preview_id)
    all_scenes = preview_data.get("scenes", [])
    if not all_scenes:
        raise ValueError("Preview contains no scenes.")

    # Filter to only the requested scene IDs
    requested_ids = set(str(sid) for sid in scene_ids)
    selected_scenes = []
    for scene in all_scenes:
        scene_id = str(scene.get("scene_id", scene.get("id", "")))
        if scene_id in requested_ids:
            selected_scenes.append(dict(scene))  # copy

    if not selected_scenes:
        raise ValueError(
            f"No matching scenes found for scene_ids: {list(requested_ids)}. "
            f"Available IDs: {[str(s.get('scene_id', s.get('id', ''))) for s in all_scenes]}"
        )

    # Apply overrides from job_input directly into each scene
    override_keys = ["steps", "duration_seconds", "fps", "cfg", "width", "height", "num_frames", "seed"]
    for scene in selected_scenes:
        for key in override_keys:
            if key in job_input:
                scene[key] = job_input[key]

    # Ensure working paths exist
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    # Render the video
    output_filename = f"final_{uuid.uuid4().hex}.mp4"
    local_output_path = settings.outputs_dir / output_filename

    print(f"Starting video generation for {len(selected_scenes)} selected scenes ...")
    final_video_path = generate_video_from_prompt_items(selected_scenes, local_output_path)
    print(f"Video generated successfully at: {final_video_path}")

    # Upload to R2 / S3
    video_url = upload_to_cloud_storage(Path(final_video_path))

    # Cleanup local copies
    cleanup_local_files()

    return {
        "video_url": video_url,
        "download_url": video_url,
    }


def _handle_legacy(job_input: dict) -> dict:
    """Legacy behavior (no action field): full pipeline in one shot."""
    prompt_items = parse_input(job_input)

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"final_{uuid.uuid4().hex}.mp4"
    local_output_path = settings.outputs_dir / output_filename

    print(f"Starting video generation for {len(prompt_items)} prompts...")
    final_video_path = generate_video_from_prompt_items(prompt_items, local_output_path)
    print(f"Video generated successfully at: {final_video_path}")

    video_url = upload_to_cloud_storage(Path(final_video_path))
    cleanup_local_files()

    return {
        "status": "success",
        "download_url": video_url,
        "video_url": video_url,
    }


# ---------------------------------------------------------------------------
# RunPod entry point
# ---------------------------------------------------------------------------

def handler(job):
    """RunPod Serverless Job Entry Point.

    Supports three modes via ``input.action``:
      - ``"preview"``  → analyze book/text, return scene metadata, store in R2.
      - ``"generate"`` → load preview from R2, render selected scenes, upload video.
      - *(omitted)*    → legacy full-pipeline behavior (backward compatible).
    """
    start_time = time.time()
    job_input = job.get("input", {})
    action = job_input.get("action", "").strip().lower()

    try:
        if action == "preview":
            result = _handle_preview(job_input)
        elif action == "generate":
            result = _handle_generate(job_input)
        else:
            result = _handle_legacy(job_input)

        elapsed_time = time.time() - start_time
        result["execution_time_seconds"] = round(elapsed_time, 2)
        if "status" not in result:
            result["status"] = "success"
        return result

    except Exception as e:
        print(f"Error during job execution: {str(e)}")
        cleanup_local_files()
        return {
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
