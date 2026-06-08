"""
prompt_enhancer_addon.py  –  LLM Refinement (optional toggle)
==============================================================
Refines a structured shot dict into fluid cinematic prose
that Wan 14B / HunyuanVideo handles better than template text.

Usage (in text_pipeline.py):
    from prompt_enhancer_addon import refine_shot_prompt
    if use_llm_refinement:
        prompt = refine_shot_prompt(shot_dict, book_profile)

Toggle via env var:  LLM_REFINE_PROMPTS=1
"""
from __future__ import annotations

import os
import re

# ── Feature flag ──────────────────────────────────────────────
LLM_REFINE_ENABLED: bool = os.getenv("LLM_REFINE_PROMPTS", "0").strip() in {"1", "true", "yes", "on"}

# Wan 14B hard limits we must not exceed even after LLM rewrite
_MAX_WORDS        = 82
_FORBIDDEN_PHRASES = [
    "cut to",
    "then",
    "after that",
    "meanwhile",
    "as he",
    "as she",
    "as they",
    "suddenly",
    "and then",
    "next,",
    "we see",
    "the camera",
]


# ─────────────────────────────────────────────────────────────
#  System prompt (sent once per call)
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a professional cinematographer writing video generation prompts.
Your job: rewrite a structured shot description into a single flowing \
cinematic sentence that a video diffusion model (Wan 14B) will follow precisely.

Rules (MUST follow every one):
1. Present tense only. Never past tense.
2. ONE subject, ONE physical action, ONE camera move.
3. Maximum {max_words} words total.
4. No cuts, no "and then", no "suddenly", no "we see", no narration.
5. No internal mental states — only what a camera physically sees.
6. Start with the visual style label, then subject+action+location.
7. End with: "Single continuous shot, no cuts."
8. Keep lighting and color description brief (≤ 8 words).
9. Keep camera directive to one short phrase.
10. Do NOT add characters, locations, or props not in the input.
Return ONLY the rewritten prompt. No explanation. No quotes. No markdown.
""".format(max_words=_MAX_WORDS)

_USER_TEMPLATE = """\
Rewrite this shot as cinematic prose (max {max_words} words):

Subject: {subject}
Action: {action}
Location: {setting}
Visible motion: {motion}
Camera: {camera}
Lighting: {lighting}
Color: {color}
Mood: {mood}
Continuity: {continuity}
Style: {style}
""".format


# ─────────────────────────────────────────────────────────────
#  Validation helpers
# ─────────────────────────────────────────────────────────────

def _too_long(text: str) -> bool:
    return len(text.split()) > _MAX_WORDS


def _has_forbidden(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _FORBIDDEN_PHRASES)


def _extract_text_from_response(data: dict) -> str:
    """Extract plain text from Anthropic /v1/messages response."""
    content = data.get("content", [])
    parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return " ".join(parts).strip()


def _validate_and_clean(raw: str, fallback: str) -> str:
    """
    Return the LLM-rewritten prompt if it passes quality gates,
    otherwise return the original fallback prompt unchanged.
    """
    cleaned = raw.strip().strip('"').strip("'")
    # Remove markdown artifacts
    cleaned = re.sub(r"```[a-z]*", "", cleaned).strip("`").strip()
    # Reject if too long
    if _too_long(cleaned):
        words = cleaned.split()
        cleaned = " ".join(words[:_MAX_WORDS])
        if not cleaned.endswith("."):
            cleaned = cleaned.rstrip(",;:") + "."
    # Reject if forbidden multi-action phrases still present
    if _has_forbidden(cleaned):
        return fallback
    # Reject if suspiciously short (< 20 words = LLM probably failed)
    if len(cleaned.split()) < 20:
        return fallback
    return cleaned


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def refine_shot_prompt(
    shot_dict:    dict,
    book_profile: dict | None = None,
) -> str:
    """
    Takes a shot dict (as produced by _decompose_scene_to_shots) and
    returns a refined cinematic prompt string.

    Falls back to shot_dict["prompt"] silently on any error.

    Parameters
    ----------
    shot_dict    : dict with keys: prompt, subject (optional), camera,
                   continuity_context, and the analysis fields.
    book_profile : optional dict with visual_style, realism_level, allow_animation.
    """
    fallback = shot_dict.get("prompt", "")
    if not LLM_REFINE_ENABLED:
        return fallback

    try:
        return _call_mistral_refine(shot_dict, book_profile or {}, fallback)
    except Exception:
        return fallback


def _style_label(book_profile: dict) -> str:
    if book_profile.get("allow_animation"):
        return "Cinematic animated scene"
    if book_profile.get("realism_level") == "stylized":
        return "Cinematic stylized live-action"
    return "Cinematic live-action"


def _call_mistral_refine(
    shot_dict:    dict,
    book_profile: dict,
    fallback:     str,
) -> str:
    """Calls the local Mistral adapter via prompt_enhancer.mistral_generate."""
    try:
        from prompt_enhancer import mistral_available, mistral_generate
    except ImportError:
        return fallback

    if not mistral_available():
        return fallback

    # Extract fields from the shot dict
    prompt_text  = shot_dict.get("prompt", fallback)
    camera       = shot_dict.get("camera") or shot_dict.get("camera", "steady medium shot")
    continuity   = shot_dict.get("continuity") or shot_dict.get("continuity_context", "") or "opening shot"
    excerpt      = shot_dict.get("scene_excerpt", "")[:200]

    subject      = shot_dict.get("subject")
    if not subject:
        subject_m  = re.search(r'(?:Cinematic [^.]+\.\s*)([A-Z][A-Za-z .]+?)\s+(?:sits|stands|walks|drives|spots|keeps|pauses|rushes|enters|leaves|turns|looks|reaches|grabs|holds|points|stops|moves|steps|rises)', prompt_text)
        subject    = subject_m.group(1).strip() if subject_m else "the character"

    action       = shot_dict.get("action") or (excerpt[:150] if excerpt else prompt_text[:120])
    setting      = shot_dict.get("setting") or _extract_setting(prompt_text)
    motion       = shot_dict.get("motion") or _extract_motion(prompt_text)
    lighting     = shot_dict.get("lighting") or _extract_lighting(prompt_text)
    color        = shot_dict.get("color") or _extract_color(prompt_text)
    mood         = shot_dict.get("mood") or _extract_mood(prompt_text)

    # Build the user message
    user_msg = _USER_TEMPLATE(
        max_words  = _MAX_WORDS,
        subject    = subject,
        action     = action,
        setting    = setting,
        motion     = motion,
        camera     = camera,
        lighting   = lighting,
        color      = color,
        mood       = mood,
        continuity = continuity,
        style      = _style_label(book_profile),
    )

    full_prompt = _SYSTEM_PROMPT + "\n\n" + user_msg

    try:
        raw = mistral_generate(full_prompt, max_new_tokens=160)
    except Exception:
        return fallback

    return _validate_and_clean(raw, fallback)


# ── Lightweight field extractors (regex on the already-assembled prompt) ──

def _extract_setting(prompt: str) -> str:
    m = re.search(r',\s*(a [a-z ]+|an [a-z ]+)\.', prompt)
    return m.group(1) if m else "the scene location"

def _extract_motion(prompt: str) -> str:
    # Motion is the second sentence in the prompt (after the core action sentence)
    sentences = prompt.split(". ")
    return sentences[2].strip() if len(sentences) > 2 else "natural movement"

def _extract_lighting(prompt: str) -> str:
    m = re.search(r'(soft [a-z ]+lighting|low-key lighting[^,]*|harsh [a-z ]+lighting|bright [a-z ]+light[^,]*)', prompt, re.I)
    return m.group(1) if m else "soft cinematic lighting"

def _extract_color(prompt: str) -> str:
    m = re.search(r'(muted [a-z ]+|warm [a-z ]+|cool [a-z ]+|neutral-cool [a-z ]+|rich [a-z ]+|desaturated [a-z ]+)', prompt, re.I)
    return m.group(1).rstrip(". ") if m else "natural tones"

def _extract_mood(prompt: str) -> str:
    moods = ["tense and uneasy", "quietly mysterious", "dark and threatening",
             "somber and intimate", "subtle and observational", "light but grounded",
             "frustrated and sharp", "quiet and restrained"]
    lowered = prompt.lower()
    for mood in moods:
        if mood in lowered:
            return mood
    return "subtle and observational"
