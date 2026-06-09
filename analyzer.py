from __future__ import annotations

import json
from pathlib import Path

from text_pipeline import build_final_video_prompts, scene_prompt_to_dict


def _read_book_file(book_path: str | Path) -> str:
    path = Path(book_path)
    if not path.exists():
        raise FileNotFoundError(f"Book file not found: {path}")

    # Normalize: check the full filename for known extensions
    # e.g. Gutenberg URLs end in .txt.utf-8 so path.suffix == '.utf-8'
    name_lower = path.name.lower()

    if name_lower.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise RuntimeError("PyPDF2 is required to read PDF files.") from exc

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    # Treat .txt, .txt.utf-8, .utf-8, .text, or any unknown extension as plain text
    return path.read_text(encoding="utf-8", errors="replace")


def analyze_raw_text(
    raw_text: str,
    *,
    max_scenes: int | None = None,
    scene_window: int = 3,
) -> list[dict]:
    final_video_prompts = build_final_video_prompts(
        raw_text,
        max_scenes=max_scenes,
        scene_window=scene_window,
    )
    return [scene_prompt_to_dict(item) for item in final_video_prompts]


def analyze_book(
    book_path: str | Path,
    *,
    max_scenes: int | None = None,
    scene_window: int = 3,
) -> list[dict]:
    raw_text = _read_book_file(book_path)
    return analyze_raw_text(
        raw_text,
        max_scenes=max_scenes,
        scene_window=scene_window,
    )


def run_analysis(
    book_path: str | Path,
    output_json_path: str | Path = "analysis_output.json",
    *,
    max_scenes: int = 3,
    scene_window: int = 3,
) -> str:
    final_video_prompts = analyze_book(
        book_path,
        max_scenes=max_scenes,
        scene_window=scene_window,
    )

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            final_video_prompts,
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return str(output_path.resolve())
