import os
import sys
import shutil
import time
import uuid
from pathlib import Path
import boto3
import runpod
import requests

# Ensure workspace project directory is in system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import settings
from video_generator import generate_video_from_prompt_items

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
        ("image_start", "image_start")
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
        max_scenes = job_input.get("max_scenes", 3)
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

def upload_to_cloud_storage(local_file_path: Path) -> str:
    """Upload final video to Cloudflare R2 or AWS S3 and return the public URL."""
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
        ExtraArgs=extra_args
    )

    if public_url_prefix:
        final_url = f"{public_url_prefix.rstrip('/')}/{object_name}"
    elif endpoint_url:
        final_url = f"{endpoint_url.rstrip('/')}/{bucket_name}/{object_name}"
    else:
        region = region_name or "us-east-1"
        final_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"

    print(f"Upload successful. URL: {final_url}")
    return final_url

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

def handler(job):
    """RunPod Serverless Job Entry Point."""
    start_time = time.time()
    job_input = job.get("input", {})
    
    try:
        # 1. Parse inputs and download any remote reference files
        prompt_items = parse_input(job_input)
        
        # Ensure working paths exist
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Render the Text-to-Video outputs
        output_filename = f"final_{uuid.uuid4().hex}.mp4"
        local_output_path = settings.outputs_dir / output_filename
        
        print(f"Starting video generation for {len(prompt_items)} prompts...")
        final_video_path = generate_video_from_prompt_items(prompt_items, local_output_path)
        print(f"Video generated successfully at: {final_video_path}")
        
        # 3. Upload to R2 / AWS S3
        video_url = upload_to_cloud_storage(Path(final_video_path))
        
        # 4. Cleanup local copies immediately
        cleanup_local_files()
        
        elapsed_time = time.time() - start_time
        return {
            "status": "success",
            "download_url": video_url,
            "video_url": video_url,
            "execution_time_seconds": round(elapsed_time, 2)
        }
        
    except Exception as e:
        print(f"Error during job execution: {str(e)}")
        cleanup_local_files()
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
