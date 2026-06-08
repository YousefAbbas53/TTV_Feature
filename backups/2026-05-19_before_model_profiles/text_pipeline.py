from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from prompt_enhancer import analyze_scene_with_mistral, infer_book_profile_with_mistral, mistral_available

# ---------------------------------------------------------------------------
# Global visual style defaults
# ---------------------------------------------------------------------------

GLOBAL_VISUAL_STYLE = {
    "lighting": "soft cinematic lighting, consistent exposure",
    "color": "muted natural colors, low saturation",
    "framing": "cinematic framing, shallow depth of field",
    "motion": "natural realistic motion",
}

BOOK_PROFILE = {
    "audience": "general",
    "visual_style": "cinematic realism",
    "realism_level": "realistic",
    "color_palette": "natural tones",
    "scene_seconds": 5,          # FIX: was 10 → caused 160+ frames & sliding-window artifacts
    "allow_animation": False,
}

GENRE_PRESETS = {
    "gothic":    {"lighting": "low-key lighting, deep shadows",               "color": "cold muted tones",                          "motion": "slow deliberate camera movement"},
    "horror":    {"lighting": "harsh contrast lighting",                       "color": "desaturated cold palette",                  "motion": "uneasy handheld framing"},
    "romance":   {"lighting": "soft warm lighting",                            "color": "warm pastel tones",                         "motion": "gentle smooth camera movement"},
    "fantasy":   {"lighting": "magical soft glow, practical motivated light",  "color": "rich saturated colors, subtle bloom",       "motion": "smooth cinematic movement"},
    "mystery":   {"lighting": "motivated low light, pockets of shadow",        "color": "neutral-cool palette, restrained saturation","motion": "controlled slow moves, investigative framing"},
    "thriller":  {"lighting": "high contrast, sharp practicals",               "color": "cool tones, slightly desaturated",           "motion": "tense handheld or tight dolly, urgent framing"},
    "adventure": {"lighting": "bright naturalistic light, high clarity",       "color": "vibrant but grounded colors",               "motion": "dynamic movement, wider lenses, energetic pacing"},
    "drama":     {"lighting": "soft naturalistic light, gentle contrast",      "color": "earthy tones, realistic skin tones",        "motion": "steady intimate coverage, character-focused framing"},
}

CANDIDATE_LABELS = ["fantasy", "adventure", "mystery", "horror", "gothic", "romance", "thriller", "drama"]
GENRE_DEFS = {
    "fantasy":   "magic, wizards, spells, mythical creatures, enchanted places, quests",
    "adventure": "journey, quest, danger, action, exploration, challenges",
    "mystery":   "investigation, secrets, clues, uncovering truth, suspense",
    "horror":    "fear, terror, threat, monsters, violence, dread, survival",
    "gothic":    "old castles, gloom, haunting, melancholy, ominous atmosphere",
    "romance":   "love story, relationships, longing, affection, courtship",
    "thriller":  "high stakes, suspense, danger, pursuit, tension, twists",
    "drama":     "character conflict, emotions, relationships, life struggles",
}

MOOD_KEYWORDS = {
    "joyful":    ["smiled", "laugh", "laughed", "happy", "joy", "bright", "warm"],
    "sad":       ["cried", "tears", "sorrow", "grief", "lonely", "silent"],
    "fearful":   ["fear", "afraid", "terror", "panic", "scream", "tremble"],
    "tense":     ["followed", "chase", "watching", "suspicious", "nervous", "waited"],
    "mysterious":["strange", "unknown", "shadow", "secret", "whisper", "mysterious"],
    "ominous":   ["dark", "storm", "threat", "ominous", "danger", "evil"],
    "calm":      ["quiet", "gentle", "peaceful", "soft", "still", "calm"],
    "angry":     ["angry", "shouted", "rage", "furious", "slam"],
}

TITLES    = {"Mr", "Mrs", "Ms", "Miss", "Dr", "Professor", "Uncle", "Aunt"}
NAME_STOP = {
    "Do", "It", "He", "She", "They", "His", "Her", "Their", "The", "A", "An",
    "And", "But", "With", "No", "If", "Everything", "Someone", "Somebody", "Anyone",
} | TITLES | {"Mrs", "Mr", "Ms", "Miss"}

# FIX: hard cap so we never exceed 80 frames (5 s × 16 fps = 80)
MAX_SCENE_SECONDS = 5


@dataclass(frozen=True)
class ScenePrompt:
    scene_id: str
    shot_id: int
    scene_excerpt: str
    prompt: str
    negative_prompt: str
    duration_seconds: int
    camera: str


# ---------------------------------------------------------------------------
# Text cleaning & segmentation
# ---------------------------------------------------------------------------

def clean_book_text(raw_text):
    """
    General-purpose text cleaner for PDF / TXT / OCR / Project Gutenberg.
    Guarantees: never returns 0 paragraphs if text exists.
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.replace("\t", " ").replace("\r", "\n")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    paragraphs = []
    buffer = []
    word_count = 0

    TARGET_WORDS = 120
    MIN_WORDS    = 40

    for line in lines:
        if re.fullmatch(r"\d{1,4}", line):
            continue
        words = line.split()
        buffer.append(line)
        word_count += len(words)

        if word_count >= TARGET_WORDS and re.search(r'[.!?؟!"\u201d]$', line):
            paragraphs.append(" ".join(buffer).strip())
            buffer = []
            word_count = 0

    if buffer:
        paragraph = " ".join(buffer).strip()
        if len(paragraph.split()) >= MIN_WORDS:
            paragraphs.append(paragraph)

    if not paragraphs:
        fallback = " ".join(lines)
        if fallback:
            paragraphs = [fallback]

    return paragraphs


def segment_scenes(paragraphs, window=3):
    scenes = []
    for i in range(0, len(paragraphs), window):
        chunk = " ".join(paragraphs[i:i + window])
        if len(chunk) > 200:
            scenes.append(chunk)
    return scenes


# ---------------------------------------------------------------------------
# Character / mood / setting helpers
# ---------------------------------------------------------------------------

def extract_characters_regex(text, max_names=6):
    titled  = re.findall(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Professor|Uncle|Aunt)\.?\s+[A-Z][a-z]+\b", text)
    families = re.findall(r"\b[Tt]he\s+[A-Z][a-z]+s\b", text)
    singles  = re.findall(r"\b[A-Z][a-z]{2,}\b", text)

    out: list[str] = []

    def add(value: str) -> None:
        if value and value not in out and len(out) < max_names:
            out.append(value)

    for v in titled:   add(v)
    for v in families: add(v)
    for v in singles:
        if v in NAME_STOP:
            continue
        add(v)

    pruned: list[str] = []
    for i, v in enumerate(out):
        is_sub = any(
            j != i and re.search(rf"\b{re.escape(v.lower())}\b", other.lower())
            for j, other in enumerate(out)
        )
        if not is_sub:
            pruned.append(v)

    text_lower = text.lower()
    final: list[str] = []
    for v in pruned:
        if re.fullmatch(r"[A-Z][a-z]+s", v) and f"the {v.lower()}" in text_lower:
            v = "the " + v
        if v.startswith("The "):
            v = "the " + v[4:]
        if v not in final:
            final.append(v)
    return final[:max_names]


def _guess_mood(scene_text):
    lowered = scene_text.lower()
    for mood, keywords in MOOD_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            return mood
    return "neutral"


def _guess_setting(scene_text):
    # Try to find a proper-noun location phrase
    match = re.search(
        r"\b(in|at|inside|outside|near|on)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,4})",
        scene_text,
    )
    if match:
        candidate = match.group(0).strip()
        # Reject overly short or stop-word-only results
        if len(candidate.split()) >= 2:
            return candidate
    return "a scene"


def _extract_characters(scene_text, max_names=4):
    return extract_characters_regex(scene_text, max_names=max_names)


def _extract_beats(scene_text, max_beats=3):
    sentences = re.split(r"(?<=[.!?؟])\s+", scene_text.strip())
    beats = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            beats.append(sentence)
        if len(beats) >= max_beats:
            break
    return beats


def build_dataset_from_scenes(scenes):
    dataset = []
    for i, scene_text in enumerate(scenes):
        llm_analysis = analyze_scene_with_mistral(scene_text) if mistral_available() else None
        dataset.append(
            {
                "scene_id": f"scene_{i}",
                "scene_text": scene_text,
                "analysis": {
                    "setting":    (llm_analysis or {}).get("setting",    _guess_setting(scene_text)),
                    "mood":       (llm_analysis or {}).get("mood",       _guess_mood(scene_text)),
                    "characters": _extract_characters(scene_text),
                    "beats":      _extract_beats(scene_text),
                },
            }
        )
    return dataset


# ---------------------------------------------------------------------------
# Scene → video item
# ---------------------------------------------------------------------------

def scene_to_video_item(scene, scene_id, book_profile):
    a      = scene.get("analysis", {}) or {}
    setting = a.get("setting", "a scene")
    mood    = a.get("mood",    "neutral")
    chars   = a.get("characters", []) or []
    beats   = a.get("beats",   []) or []

    # FIX: use only the FIRST beat so the model focuses on ONE clear action
    # instead of being confused by multiple "Then ..." clauses.
    action = beats[0].strip().rstrip(".") if beats else "A key moment unfolds"

    # Trim overly long action strings so the prompt stays focused
    if len(action) > 120:
        action = action[:117] + "..."

    main_char = chars[0] if chars else "a character"

    # FIX: enforce MAX_SCENE_SECONDS so frames stay ≤ 80 (no sliding-window)
    duration = min(int(book_profile.get("scene_seconds", 5)), MAX_SCENE_SECONDS)

    if mood in {"mysterious", "ominous", "tense", "fearful"}:
        cam = "slow creeping dolly in, restrained handheld micro-movement, tense framing"
    elif mood in {"joyful", "calm"}:
        cam = "gentle smooth pan, wider lens, airy framing"
    else:
        cam = "steady medium shot, slow controlled dolly"

    return {
        "scene_id": scene_id,
        "shot_id":  1,
        "duration": duration,
        "camera":   cam,
        # FIX: concise, single-action visuals line — no multi-clause "Then" chains
        "visuals":  f"Setting: {setting}. Character: {main_char}. Action: {action}. Mood: {mood}.",
    }


# ---------------------------------------------------------------------------
# Negative prompt
# ---------------------------------------------------------------------------

def build_negative_prompt(book_profile):
    base = [
        "low quality", "blurry", "flicker", "jitter",
        "watermark", "logo", "subtitles", "text",
        "morphing", "ghosting", "warped motion", "deformed",
    ]
    if not book_profile.get("allow_animation", False):
        base += ["cartoon", "anime", "3d render", "CGI"]
    return ", ".join(base)


# ---------------------------------------------------------------------------
# Book profile helpers
# ---------------------------------------------------------------------------

def heuristic_book_profile(scenes: Iterable[str]) -> dict:
    text   = " ".join([s for s in list(scenes)[:120] if isinstance(s, str)]).lower()
    fantasy  = ["magic", "wizard", "spell", "wand", "enchanted", "dragon", "fairy", "mermaid", "curse"]
    childlike = ["nursery", "bedtime", "toy", "storybook", "mother", "father", "schoolboy"]
    fscore = sum(word in text for word in fantasy)
    cscore = sum(word in text for word in childlike)

    if fscore >= 2:
        return {
            "audience":      "general",
            "visual_style":  "storybook fantasy",
            "realism_level": "stylized",
            "color_palette": "rich magical colors",
            "scene_seconds": MAX_SCENE_SECONDS,   # FIX: was 10
            "allow_animation": True,
        }
    if cscore >= 2:
        return {
            "audience":      "children",
            "visual_style":  "storybook fantasy",
            "realism_level": "stylized",
            "color_palette": "bright warm pastels",
            "scene_seconds": MAX_SCENE_SECONDS,   # FIX: was 9
            "allow_animation": True,
        }
    return {}


def _detect_genre_fallback(dataset: list[dict]) -> str:
    text = " ".join(item.get("scene_text", "") for item in dataset[:40]).lower()
    scores = {
        "fantasy":   sum(w in text for w in ["magic", "wizard", "spell", "dragon", "enchanted", "curse"]),
        "adventure": sum(w in text for w in ["journey", "quest", "danger", "travel", "explore", "battle"]),
        "mystery":   sum(w in text for w in ["secret", "clue", "mystery", "strange", "unknown", "hidden"]),
        "horror":    sum(w in text for w in ["blood", "monster", "fear", "terror", "scream", "grave"]),
        "gothic":    sum(w in text for w in ["castle", "gloom", "haunted", "shadow", "ominous", "melancholy"]),
        "romance":   sum(w in text for w in ["love", "kiss", "heart", "beloved", "longing", "romance"]),
        "thriller":  sum(w in text for w in ["chase", "pursuit", "threat", "danger", "escape", "suspense"]),
        "drama":     sum(w in text for w in ["family", "conflict", "emotion", "relationship", "struggle", "life"]),
    }
    best = max(scores.items(), key=lambda item: item[1])[0]
    return best if scores[best] > 0 else "drama"


def detect_genre(dataset: list[dict], max_scenes: int = 40) -> str:
    texts = [item.get("scene_text", "") for item in dataset[:max_scenes] if isinstance(item, dict)]
    if not texts:
        return "drama"

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return _detect_genre_fallback(dataset)

    try:
        embedder    = SentenceTransformer("all-MiniLM-L6-v2")
        label_vecs  = embedder.encode([GENRE_DEFS[l] for l in CANDIDATE_LABELS], normalize_embeddings=True)
        text_vecs   = embedder.encode(texts, normalize_embeddings=True)

        avg_scores: dict[str, float] = {}
        for li, label in enumerate(CANDIDATE_LABELS):
            avg_scores[label] = sum(float(v @ label_vecs[li]) for v in text_vecs) / max(1, len(text_vecs))

        ranked  = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        primary, score = ranked[0]
        return primary if score >= 0.35 else "drama"
    except Exception:
        return _detect_genre_fallback(dataset)


def apply_genre_preset(base_style: dict, active_genre: str) -> dict:
    style  = dict(base_style)
    preset = GENRE_PRESETS.get(active_genre, {})
    style.update({
        "lighting": preset.get("lighting", style["lighting"]),
        "color":    preset.get("color",    style["color"]),
        "motion":   preset.get("motion",   style["motion"]),
    })
    return style


# ---------------------------------------------------------------------------
# Prompt compilation
# ---------------------------------------------------------------------------

def wan_adapter(base_prompt, book_profile):
    style   = book_profile.get("visual_style",  "cinematic realism")
    realism = book_profile.get("realism_level", "realistic")
    palette = book_profile.get("color_palette", "natural tones")
    return f"{style}, {realism}, coherent motion, stable camera, color palette: {palette}. {base_prompt}"


def compile_video_prompt_scene(item, book_profile, visual_style):
    visuals = item.get("visuals", "")
    camera  = item.get("camera",  "steady shot")

    # FIX: keep the prompt SHORT and FOCUSED — video models degrade with long prompts
    base_prompt = (
        f"{visuals} "
        f"Camera: {camera}. "
        f"Lighting: {visual_style['lighting']}. "
        f"Color: {visual_style['color']}. "
        f"Motion: {visual_style['motion']}. "
        f"Single cohesive scene, no cuts, no time jumps."
    )
    prompt = wan_adapter(base_prompt, book_profile)

    return {
        "prompt":           prompt.strip(),
        "negative_prompt":  build_negative_prompt(book_profile),
        "duration_seconds": int(item.get("duration", book_profile.get("scene_seconds", MAX_SCENE_SECONDS))),
        "camera":           camera,
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def build_final_video_prompts(raw_text, max_scenes=3, scene_window=3):
    paragraphs = clean_book_text(raw_text)
    scenes     = segment_scenes(paragraphs, window=scene_window)
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    book_profile = dict(BOOK_PROFILE)

    if mistral_available():
        inferred_profile = infer_book_profile_with_mistral(scenes)
        if isinstance(inferred_profile, dict):
            book_profile.update({k: v for k, v in inferred_profile.items() if v not in (None, "")})

    heuristic_profile = heuristic_book_profile(scenes)
    for key, value in heuristic_profile.items():
        if key not in book_profile or book_profile.get(key) in [
            None, "", "cinematic realism", "realistic", "natural tones", "general", False
        ]:
            book_profile[key] = value

    # FIX: always enforce the hard cap after merging all profile sources
    book_profile["scene_seconds"] = min(
        int(book_profile.get("scene_seconds", MAX_SCENE_SECONDS)),
        MAX_SCENE_SECONDS,
    )

    dataset      = build_dataset_from_scenes(scenes)
    active_genre = detect_genre(dataset, max_scenes=40)
    visual_style = apply_genre_preset(GLOBAL_VISUAL_STYLE, active_genre)
    video_items  = [scene_to_video_item(scene, scene["scene_id"], book_profile) for scene in dataset]
    scene_text_map = {scene["scene_id"]: scene["scene_text"] for scene in dataset}

    final_video_prompts = []
    for item in video_items:
        sid      = item["scene_id"]
        compiled = compile_video_prompt_scene(item, book_profile, visual_style)
        final_video_prompts.append(
            ScenePrompt(
                scene_id=sid,
                shot_id=1,
                scene_excerpt=scene_text_map[sid][:240],
                prompt=compiled["prompt"],
                negative_prompt=compiled["negative_prompt"],
                duration_seconds=compiled["duration_seconds"],
                camera=compiled["camera"],
            )
        )

    return final_video_prompts


def scene_prompt_to_dict(scene_prompt: ScenePrompt) -> dict:
    return asdict(scene_prompt)
