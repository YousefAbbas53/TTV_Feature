# LITVISION

Book/text to video API using the original Kaggle-style Wan2GP baseline.

This branch intentionally supports one video model only:

- `Wan T2V 1.3B`
- `model_type=t2v_1.3B`
- `768x432`
- `80 frames`
- `16 fps`
- `40 steps`
- `cfg=4.8`

The 14B and Hunyuan experiments were removed because the deployed quantized/offloaded runs produced worse motion and blur than the original notebook baseline.

## RunPod / Vast Setup

```bash
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/basmala-19/LITVISION2.git
cd /workspace/LITVISION2
bash vastai_setup.sh
```

The setup script always starts the API with the Wan 1.3B baseline settings.

Check runtime:

```bash
curl http://127.0.0.1:8000/health/full
curl http://127.0.0.1:8000/model-info
```

Expected model info:

```json
{
  "current_preset": "wan_t2v_1_3b",
  "model_type": "t2v_1.3B",
  "width": 768,
  "height": 432,
  "num_frames": 80,
  "steps": 40,
  "cfg": 4.8
}
```

## Quick Test

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{"prompt":"cinematic live-action, photorealistic, natural skin texture, realistic shadows, soft film grain. A middle-aged British office worker with a trim dark mustache sits at a cluttered 1990s office desk near a large window, natural morning light, subtle head movement, small hand movement, single continuous shot, coherent motion, stable camera, no cuts.","seed":1234}' \
  --output wan13_baseline_test.mp4
```

## Preview Book

Put books in:

```bash
/workspace/LITVISION2/books
```

Create preview:

```bash
curl -X POST "http://127.0.0.1:8000/preview-from-file?max_scenes=6&scene_window=2" \
  -H "x-api-key: change-me" \
  -F "file=@/workspace/LITVISION2/books/your_book.pdf" \
  -o preview.json
```

Inspect scene IDs:

```bash
python -m json.tool preview.json
```

Render one selected scene first:

```bash
curl -X POST "http://127.0.0.1:8000/generate-selected-scenes-json" \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{"preview_id":"PUT_PREVIEW_ID_HERE","scene_ids":["scene_0"],"output_name":"scene0.mp4"}'
```

Download:

```bash
curl -L "http://127.0.0.1:8000/files/scene0.mp4" --output scene0.mp4
```

## Notes

- Do not use old preview IDs after changing prompt code.
- Test one scene before rendering many scenes.
- Keep the same prompt, seed, and settings when comparing local vs deployment.
