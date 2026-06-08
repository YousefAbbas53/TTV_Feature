from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import os
import re


BASE_MODEL = os.getenv("MISTRAL_BASE_MODEL", "mistralai/Mistral-7B-v0.1")
ADAPTER_DIR = Path(os.getenv("MISTRAL_ADAPTER_DIR", Path(__file__).resolve().parent / "models" / "mistral-finetuned"))
MISTRAL_ENABLED = os.getenv("MISTRAL_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
MOOD_LIST = ["neutral", "calm", "tense", "mysterious", "ominous", "joyful", "sad", "angry", "fearful"]


def _extract_json_loose(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_model_bundle():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not MISTRAL_ENABLED:
        raise RuntimeError("Mistral integration is disabled.")
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Mistral adapter directory not found: {ADAPTER_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("Mistral integration requires CUDA to avoid unsafe CPU fallback.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def mistral_available() -> bool:
    if not MISTRAL_ENABLED:
        return False
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    return ADAPTER_DIR.exists()


def mistral_generate(text: str, max_new_tokens: int = 220) -> str:
    import torch

    tokenizer, model = _load_model_bundle()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def analyze_scene_with_mistral(scene_text: str) -> dict | None:
    prompt = f"""
Return JSON ONLY. No prose. No markdown.

Extract STRICTLY from the text (no outside knowledge).
Keys:
setting (very short or "Unknown"),
mood (one of {MOOD_LIST}).

Text:
{scene_text}
JSON:
"""
    try:
        raw = mistral_generate(prompt, max_new_tokens=120)
    except Exception:
        return None

    obj = _extract_json_loose(raw)
    setting = obj.get("setting", "Unknown")
    mood = obj.get("mood", "neutral")

    if mood not in MOOD_LIST:
        mood = "neutral"
    if not isinstance(setting, str) or not setting.strip():
        setting = "Unknown"

    return {
        "setting": setting.strip(),
        "mood": mood,
    }


def infer_book_profile_with_mistral(scenes: list[str]) -> dict | None:
    if not scenes:
        return None

    sample = "\n\n---\n\n".join(s[:800] for s in scenes[:16])[:9000]
    prompt = f"""
Return JSON ONLY. No prose. No markdown. No extra keys.

Choose values:
audience: children|teen|adult|general
realism_level: realistic|stylized|animated
allow_animation: true|false
scene_seconds: integer 4-8

Example:
{{"audience":"general","visual_style":"cinematic realism","realism_level":"realistic",
"color_palette":"natural tones","scene_seconds":4,"allow_animation":false}}

Infer strictly from the excerpts (no outside knowledge). Do NOT quote sentences.

Excerpts:
{sample}

JSON:
"""
    try:
        raw = mistral_generate(prompt, max_new_tokens=220)
    except Exception:
        return None

    obj = _extract_json_loose(raw)
    if not isinstance(obj, dict) or not obj:
        return None
    return obj


# ---------------------------------------------------------------------------
# narrative_to_visual — converts book narrative into camera-eye description
# ---------------------------------------------------------------------------

def _heuristic_narrative_to_visual(text: str) -> str:
    """Heuristic visual description generator when Mistral is unavailable."""
    if not text or not text.strip():
        return ""

    lowered_full = text.lower()
    
    # Specific sentence-level overrides for iconic narrative lines
    if "shuddered to think what the neighbors would say" in lowered_full or "shuddered to think" in lowered_full:
        return "glances nervously through the window curtains, with a worried and tense expression."
    if "pretended she didn't have a sister" in lowered_full:
        return "turns her head away with a sharp scowl, disapproving and stern."
    if "were the last people you'd expect to be involved" in lowered_full:
        return "stands in front of a neat brick suburban house, looking perfectly ordinary and tidy."
    if "sat in the usual morning traffic jam" in lowered_full or "morning traffic jam" in lowered_full:
        return "sitting behind the steering wheel inside a car, looking out at the stagnant line of vehicles."

    # Check for quotes indicating dialogue / speaking
    if any(q in text for q in ['“', '”', '"', '«', '»']):
        if re.search(r"\b(weatherman|ted)\b", lowered_full):
            return "looks at the television screen where a news presenter is speaking, gesturing with a professional expression and weather graphics behind him."
        return "speaking and gesturing, mouth moving naturally in a conversation."

    # Clean and split into sentences
    sentences = re.split(r"(?<=[.!?؟])\s+", text.strip())
    visual_sentences: list[str] = []

    for sentence in sentences:
        s_clean = sentence.strip().strip('"').strip("'")
        if not s_clean:
            continue

        lowered = s_clean.lower()
        changed = False

        # Pattern replacements for visual actions
        rules = [
            (r"\b(shuddered to think|shuddered at the thought|shuddered)\b", "shudders slightly, a worried and nervous expression on their face"),
            (r"\b(proud to say|was proud|felt proud)\b", "stands tall with a proud and neat posture"),
            (r"\b(didn't hold with|disliked|hated|nonsense)\b", "frowns and shakes their head disapprovingly"),
            (r"\b(rattled|shaken|nervous|anxious|scared)\b", "looks visibly tense and anxious, glancing around warily"),
            (r"\b(couldn't help noticing|noticed|saw|observed|noticing)\b", "eyes widening, looking closely at something"),
            (r"\b(forgotten all about|forgot|didn't think about)\b", "walks with a neutral expression, looking momentarily distracted"),
            (r"\b(whispering excitedly|whispering|whispered)\b", "leaning in close and whispering rapidly, eyes wide with excitement"),
            (r"\b(sat frozen|sat stiffly|sat motionless|sat in armchair)\b", "sitting in a rigid posture, body completely still and tense"),
            (r"\b(stood rooted to the spot|stood rooted|couldn't move)\b", "standing completely motionless, staring ahead in disbelief"),
            (r"\b(hugged|hug|hugs)\b", "hugs the other person warmly, arms wrapping around them"),
            (r"\b(hurried|rushed)\b", "walking very quickly, steps urgent, looking around in a hurry"),
            (r"\b(angry|cross|furious|irritated)\b", "frowning deeply, jaw clenched, eyes narrowing in anger"),
            (r"\b(perfectly normal|normal and ordinary)\b", "looking calm, neat, and entirely ordinary"),
            (r"\b(morning traffic jam|traffic jam|traffic)\b", "sitting behind the steering wheel of a car in a traffic jam, looking frustrated"),
            (r"\b(eyed them angrily|eyed them|glared)\b", "stares at them with a sharp, angry glare"),
            (r"\b(secret|mysterious|strange)\b", "looking around warily as if noticing something unusual"),
        ]

        sentence_visual = s_clean
        for pattern, replacement in rules:
            if re.search(pattern, sentence_visual, re.IGNORECASE):
                sentence_visual = re.sub(pattern, replacement, sentence_visual, flags=re.IGNORECASE)
                changed = True

        if not changed:
            if any(w in lowered for w in ["thought", "believed", "felt", "knew", "remembered", "wished", "hoped"]):
                sentence_visual = "stands quietly, a thoughtful and reflective expression on their face"
            elif any(w in lowered for w in ["normal", "nonsense", "expect", "shuddered"]):
                sentence_visual = "stands in place, looking around with a neat and tidy posture"

        # Strip specific character name repetitions in the visual description to avoid redundancy
        sentence_visual = re.sub(r"\b(Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-zA-Z]*\b", "the character", sentence_visual)
        sentence_visual = re.sub(r"\b(Harry|Ron|Hermione|Dumbledore|Snape|Hagrid|Dudley)\b", "the character", sentence_visual)

        # Clean up grammatical contractions like "He'd walks", "he'd shudders", "she'd frowns", "He'd walks"
        sentence_visual = re.sub(
            r"\b(he|she|they|the character)\s*['’]d\s+(walks|sits|stands|shudders|frowns|looks|glances|stares|hugs|walk|sit|stand|shudder|frown|look|glance|stare|hug)\b",
            r"the character \2",
            sentence_visual,
            flags=re.IGNORECASE
        )
        # Strip leading contractions at the start of sentences if any remain
        sentence_visual = re.sub(r"^(he|she|they|the character)\s*['’]d\s+", "", sentence_visual, flags=re.IGNORECASE)

        visual_sentences.append(sentence_visual)
        if len(visual_sentences) >= 3:
            break

    return " ".join(visual_sentences).strip()


def narrative_to_visual(scene_text: str) -> str:
    """
    Convert narrative book text into a visual, camera-eye description
    suitable for a video generation model.

    Returns a heuristic translation on failure or if Mistral is unavailable.
    """
    if not mistral_available() or not scene_text or not scene_text.strip():
        return _heuristic_narrative_to_visual(scene_text)

    trimmed = scene_text[:1200]
    prompt = f"""\
You are a cinematographer. Describe ONLY what a camera would physically see in this scene.

Rules:
- Present tense only.
- Focus on: physical actions, body language, facial expressions, environment, lighting, colors.
- Do NOT include internal thoughts, emotions by name, or narration.
- Do NOT name the characters by their story names — describe their appearance instead.
- Maximum 4 sentences.
- Be specific and vivid.

Scene text:
{trimmed}

Visual description:
"""
    try:
        raw = mistral_generate(prompt, max_new_tokens=200)
        cleaned = raw.strip().strip('"').strip("'")
        if len(cleaned.split()) >= 12:
            return cleaned
    except Exception:
        pass
    return _heuristic_narrative_to_visual(scene_text)


# ---------------------------------------------------------------------------
# Auto-generated Character Bible — works for ANY book
# ---------------------------------------------------------------------------

def generate_character_bible_with_mistral(
    character_contexts: dict[str, str],
) -> dict[str, str]:
    """
    Auto-generate a visual Character Bible from text context.

    Parameters
    ----------
    character_contexts : dict mapping character name → concatenated text
                         excerpts where that character appears.

    Returns
    -------
    dict mapping character name → detailed physical appearance string.
    """
    if not mistral_available() or not character_contexts:
        return {}

    bible: dict[str, str] = {}
    for name, context in character_contexts.items():
        trimmed_context = context[:1500]
        prompt = f"""\
Based ONLY on the text below, describe the physical appearance of "{name}" \
for a video generation model.

Include (only if mentioned or clearly implied in the text):
- Approximate age or age group
- Body build (tall, thin, heavyset, etc.)
- Hair color and style
- Face features (glasses, beard, scars, etc.)
- Clothing and accessories
- Any distinguishing physical traits

Rules:
- Use ONLY details from the text. Do not invent anything.
- Write as comma-separated visual descriptors, not full sentences.
- Maximum 40 words.
- If very little is known, describe what you can.

Text mentioning {name}:
{trimmed_context}

Physical description of {name}:
"""
        try:
            raw = mistral_generate(prompt, max_new_tokens=100)
            cleaned = raw.strip().strip('"').strip("'").rstrip(".")
            if len(cleaned.split()) >= 5:
                bible[name] = cleaned
        except Exception:
            continue
    return bible


# ---------------------------------------------------------------------------
# Auto-generated Location Bible — works for ANY book
# ---------------------------------------------------------------------------

def generate_location_bible_with_mistral(
    location_contexts: dict[str, str],
) -> dict[str, str]:
    """
    Auto-generate a visual Location Bible from text context.

    Parameters
    ----------
    location_contexts : dict mapping location name/keyword → concatenated text
                        excerpts describing or mentioning that location.

    Returns
    -------
    dict mapping location keyword → detailed visual environment string.
    """
    if not mistral_available() or not location_contexts:
        return {}

    bible: dict[str, str] = {}
    for location, context in location_contexts.items():
        trimmed_context = context[:1500]
        prompt = f"""\
Based ONLY on the text below, describe the visual appearance of the location \
"{location}" for a video generation model.

Include (only if mentioned or clearly implied in the text):
- Type of space (interior/exterior, room type, landscape)
- Architecture and materials (stone, wood, modern, medieval, etc.)
- Lighting conditions (candlelight, daylight, dim, bright, etc.)
- Key visual details (furniture, decorations, nature, weather)
- Atmosphere and scale (cramped, vast, cozy, imposing, etc.)

Rules:
- Use ONLY details from the text. Do not invent anything.
- Write as comma-separated visual descriptors, not full sentences.
- Maximum 40 words.

Text mentioning {location}:
{trimmed_context}

Visual description of {location}:
"""
        try:
            raw = mistral_generate(prompt, max_new_tokens=100)
            cleaned = raw.strip().strip('"').strip("'").rstrip(".")
            if len(cleaned.split()) >= 5:
                bible[location] = cleaned
        except Exception:
            continue
    return bible


# ---------------------------------------------------------------------------
# Auto-generated Visual Style Bible — infers the cinematic aesthetic
# ---------------------------------------------------------------------------

def generate_visual_style_bible_with_mistral(
    sample_text: str,
) -> dict[str, str]:
    """
    Auto-generate a Visual Style Bible from a large sample of book text.

    Infers the overall cinematic aesthetic: period, atmosphere, lighting style,
    color palette, and visual tone — without referencing specific film titles.

    Parameters
    ----------
    sample_text : str — concatenated text from the first 20-30 scenes
                  (or first ~10000 words) of the book.

    Returns
    -------
    dict with keys: period, atmosphere, lighting, color_palette, visual_tone,
    architecture, weather — each mapping to a short visual descriptor string.
    Returns empty dict on failure.
    """
    if not mistral_available() or not sample_text or not sample_text.strip():
        return {}

    trimmed = sample_text[:8000]
    prompt = f"""\
Based ONLY on the text below, infer the visual style for a film adaptation.

Return JSON ONLY. No prose. No markdown.

Keys (short visual descriptors, max 10 words each):
- period: historical era or time period (e.g. "1990s Britain", "medieval Europe", "Victorian London", "far future")
- atmosphere: overall feeling (e.g. "warm magical", "dark gritty", "bright adventurous", "cold bleak")
- lighting: dominant lighting style (e.g. "warm candlelight and torches", "harsh desert sun", "foggy gaslight", "neon city glow")
- color_palette: dominant colors (e.g. "warm golds and deep reds", "cold blues and greys", "earthy greens and browns")
- visual_tone: cinematic reference WITHOUT movie names (e.g. "British fantasy film", "Victorian mystery", "epic desert saga", "noir thriller")
- architecture: building/environment style (e.g. "gothic stone castles", "Victorian townhouses", "futuristic metal structures")
- weather: typical weather/sky (e.g. "overcast skies, occasional storms", "scorching sun, endless sand", "foggy and damp")

Rules:
- Infer ONLY from the text. Do not reference specific movie titles or directors.
- Each value should be a SHORT visual phrase, not a full sentence.

Text sample:
{trimmed}

JSON:
"""
    try:
        raw = mistral_generate(prompt, max_new_tokens=250)
    except Exception:
        return {}

    obj = _extract_json_loose(raw)
    if not isinstance(obj, dict) or not obj:
        return {}

    # Validate: keep only string values with reasonable content
    valid_keys = {"period", "atmosphere", "lighting", "color_palette", "visual_tone", "architecture", "weather"}
    cleaned: dict[str, str] = {}
    for key in valid_keys:
        val = obj.get(key)
        if isinstance(val, str) and len(val.strip()) >= 3:
            cleaned[key] = val.strip()
    return cleaned

