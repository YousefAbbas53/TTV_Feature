from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from prompt_enhancer import (
    analyze_scene_with_mistral,
    generate_character_bible_with_mistral,
    generate_location_bible_with_mistral,
    generate_visual_style_bible_with_mistral,
    infer_book_profile_with_mistral,
    mistral_available,
    narrative_to_visual,
)


# ---------------------------------------------------------------------------
# Global visual style defaults
# ---------------------------------------------------------------------------

GLOBAL_VISUAL_STYLE = {
    "lighting": "natural cinematic lighting, consistent exposure, realistic shadows",
    "color": "natural muted colors, realistic skin tones, low saturation",
    "framing": "cinematic framing, shallow depth of field",
    "motion": "natural realistic motion, visible physical movement",
}

BOOK_PROFILE = {
    "audience": "general",
    "visual_style": "cinematic realism",
    "realism_level": "realistic",
    "color_palette": "natural tones",
    "scene_seconds": 5,
    "allow_animation": False,
}

GENRE_PRESETS = {
    "gothic":    {"lighting": "low-key lighting, deep shadows",                "color": "cold muted tones",                           "motion": "slow deliberate camera movement"},
    "horror":    {"lighting": "harsh contrast lighting",                       "color": "desaturated cold palette",                    "motion": "uneasy handheld framing"},
    "romance":   {"lighting": "soft warm lighting",                            "color": "warm pastel tones",                           "motion": "gentle smooth camera movement"},
    "fantasy":   {"lighting": "magical soft glow, practical motivated light",  "color": "rich saturated colors, subtle warm bloom",    "motion": "smooth cinematic movement, flowing fabric"},
    "mystery":   {"lighting": "motivated low light, pockets of shadow",        "color": "neutral-cool palette, restrained saturation", "motion": "controlled slow moves, investigative framing"},
    "thriller":  {"lighting": "high contrast, sharp practicals",               "color": "cool tones, slightly desaturated",            "motion": "tense handheld or tight dolly, urgent framing"},
    "adventure": {"lighting": "bright naturalistic light, high clarity",        "color": "vibrant but grounded colors",                 "motion": "dynamic movement, wider lenses, energetic pacing"},
    "drama":     {"lighting": "soft naturalistic light, gentle contrast",       "color": "earthy tones, realistic skin tones",          "motion": "steady intimate coverage, character-focused framing"},
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
    "joyful":    ["smiled", "laugh", "laughed", "happy", "joy", "bright", "warm", "grinned", "cheerful"],
    "sad":       ["cried", "tears", "sorrow", "grief", "lonely", "silent", "wept", "mourned"],
    "fearful":   ["fear", "afraid", "terror", "panic", "scream", "tremble", "shaking", "dread"],
    "tense":     ["followed", "chase", "watching", "suspicious", "nervous", "waited", "gripped", "clutched"],
    "mysterious": ["strange", "unknown", "shadow", "secret", "whisper", "mysterious", "hidden", "puzzle"],
    "ominous":   ["dark", "storm", "threat", "ominous", "danger", "evil", "foreboding", "sinister"],
    "calm":      ["quiet", "gentle", "peaceful", "soft", "still", "calm", "serene", "resting"],
    "angry":     ["angry", "shouted", "rage", "furious", "slam", "roared", "fury", "clenched"],
}

TITLES = {"Mr", "Mrs", "Ms", "Miss", "Dr", "Professor", "Uncle", "Aunt"}
NAME_STOP = {
    "Do", "It", "He", "She", "They", "His", "Her", "Their", "The", "A", "An",
    "And", "But", "With", "No", "If", "Everything", "Someone", "Somebody", "Anyone",
    "Yes", "Now", "After", "What", "Which", "Who", "When", "How", "There", "Only",
    "All", "Look", "Nearly", "Right", "Well", "Scene", "Setting", "Character", "Action",
    "Mood", "That", "This", "Then", "Just", "Even", "Still", "Before", "Into",
    "About", "Over", "Very", "Could", "Would", "Should", "Said", "Been", "Were",
} | TITLES

MAX_SCENE_SECONDS = 5
MIN_SCENE_WORDS = 35
MAX_ACTION_CHARS = 200

# Expanded action keywords for better action detection
ACTION_KEYWORDS = (
    "sat", "sits", "sitting", "stood", "stands", "standing", "looked", "looks",
    "stared", "turned", "turns", "walked", "walks", "hurried", "dashed", "opened",
    "held", "picked", "placed", "threw", "ran", "leaned", "raised", "lowered",
    "watched", "rushed", "grabbed", "pointed", "shouted", "whispered", "pushed",
    "pulled", "waved", "caught", "dodged", "climbed", "flew", "jumped", "fell",
    "reached", "touched", "moved", "stepped", "entered", "left", "crossed",
    "knelt", "crouched", "stretched", "slammed", "kicked", "fighting", "fought",
    "swung", "ducked", "sprinted", "crawled", "trembled", "shivered",
)

DESCRIPTION_ONLY_PATTERNS = (
    " was a ", " is a ", " were a ", " are a ", " had ", " have ",
)

BAD_SCENE_MARKERS = (
    "contents", "chapter one", "chapter two", "chapter three",
    "chapter four", "chapter five", "chapter six", "chapter seven",
    "chapter eight", "chapter nine", "chapter ten", "table of contents",
    "copyright", "all rights reserved", "published by", "isbn",
)

# Dynamic camera presets based on action type
CAMERA_PRESETS = {
    "walking": "tracking shot following the character, handheld follow camera, medium-wide framing",
    "running": "fast tracking shot, urgent handheld camera, dynamic pursuit framing",
    "talking": "medium shot with gentle push-in, shallow depth of field, eye-level framing",
    "sitting": "slow orbit around subject, intimate close-up coverage, steady framing",
    "fighting": "dynamic handheld camera, rapid angle shifts, intense close framing",
    "looking": "slow dolly-in toward subject's face, focus pull, contemplative framing",
    "entering": "wide establishing shot pulling back, revealing environment, smooth dolly",
    "emotional": "slow push-in to close-up, shallow focus, intimate framing",
    "default": "dynamic medium tracking shot, camera following subject with natural movement",
}

# Dynamic motion presets based on action type
MOTION_PRESETS = {
    "walking": "full walking cycle with arm swing, weight shifting between steps, clothing swaying with movement, hair bouncing",
    "running": "urgent running motion, arms pumping, clothing billowing, rapid leg movement, environment blurring past",
    "talking": "natural gesturing while speaking, head movements, facial expression changes, subtle body shifts",
    "sitting": "natural fidgeting, shifting weight, hand gestures, slight body turns, breathing movement",
    "fighting": "rapid combat movement, dodging, striking, physical impact reactions, intense body motion",
    "looking": "visible head turn with body following, eyes tracking, slight body lean, natural reaction movement",
    "entering": "confident stride through doorway, body adjusting to new space, head turning to look around",
    "emotional": "visible physical emotion, trembling or stiffening, hand movements, facial shifts",
    "default": "continuous body movement, natural motion cycle, visible physical action, environment reacting to character",
}


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

def _normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\t", " ").replace("\r", "\n")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _is_noise_line(line: str) -> bool:
    lowered = line.lower().strip(" .:-")
    if not lowered:
        return True
    if lowered in {"contents", "table of contents"}:
        return True
    if re.fullmatch(r"\d{1,4}", lowered):
        return True
    if re.fullmatch(r"chapter\s+[ivxlcdm0-9]+", lowered):
        return True
    if re.fullmatch(r"[ivxlcdm]+", lowered):
        return True
    if lowered.startswith("chapter ") and len(lowered.split()) <= 4:
        return True
    if len(lowered.split()) <= 3 and lowered.upper() == lowered and lowered.isascii():
        return True
    return False


def is_toc_block(text: str) -> bool:
    score = 0
    if text.count("...") > 3:
        score += 2
    if re.search(r"\b(chapter|contents|page)\b", text.lower()):
        score += 2
    if len(re.findall(r"\b[A-Z][a-z]+\b", text)) > 20:
        score += 1
    return score >= 3


def _is_toc_or_front_matter(text: str) -> bool:
    """Detect and skip Table of Contents, copyright pages, and front-matter."""
    if len(text) > 4000:
        return False
    if is_toc_block(text):
        return True
        
    upper = text.upper()
    if "CONTENTS" in upper and len(text) < 2500:
        return True
    if "TABLE OF CONTENTS" in upper:
        return True
    if any(k in upper for k in ["COPYRIGHT", "ALL RIGHTS RESERVED", "PUBLISHED BY", "ISBN", "PRINTED IN"]):
        return True
    
    # Generic dedication detector
    if len(text) < 500:
        cleaned_ded = re.sub(r'[^A-Z\s]', '', upper).strip()
        if cleaned_ded.startswith("FOR ") or "DEDICATED TO" in cleaned_ded or cleaned_ded.startswith("TO "):
            return True
            
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        # Check if lines have trailing numbers (page numbers)
        has_page_nums = sum(1 for l in lines if re.search(r'\d+$', l) or "..." in l)
        if has_page_nums / len(lines) > 0.5:
            return True
        # Check if short lines
        short = sum(1 for line in lines if len(line.split()) <= 7)
        if short / len(lines) > 0.8:
            return True
        # Check if list of Harry Potter books, chapters, or generic titles
        if any(b in upper for b in ["HARRY POTTER", "SORCERER'S STONE", "CHAMBER OF SECRETS", "PRISONER OF AZKABAN", "GOBLET OF FIRE", "ORDER OF THE PHOENIX", "HALF-BLOOD PRINCE", "DEATHLY HALLOWS", "THE BOY WHO LIVED", "THE VANISHING GLASS", "THE LETTERS FROM NO ONE"]) and short / len(lines) > 0.5:
            return True
        # List of chapters detector (mostly title-cased short lines without trailing punctuation)
        no_punctuation_lines = sum(1 for l in lines if not re.search(r'[.!?؟]$', l))
        title_cased_lines = sum(1 for l in lines if l and l[0].isupper())
        if no_punctuation_lines / len(lines) > 0.8 and title_cased_lines / len(lines) > 0.8 and len(text) < 1500:
            return True
            
    return False


def sanitize_scene_text(text: str) -> str:
    if not text:
        return ""

    # Remove TOC patterns
    text = re.sub(r"\bCONTENTS\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCHAPTER\s+\w+\b.*", "", text, flags=re.IGNORECASE)

    # Remove dedication / front matter
    text = re.sub(r"^FOR\s+.+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"DEDICATED\s+TO.+$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove dotted leader TOC lines
    text = re.sub(r"\.{3,}", " ", text)

    # Remove repeated headers (all caps short lines)
    text = re.sub(r"^[A-Z\s]{10,}$", "", text, flags=re.MULTILINE)

    return text.strip()


def find_story_start(text: str) -> int:
    candidates = []
    # Search in the first 40000 characters for common start markers
    for match in re.finditer(r"CHAPTER\s+(ONE|1|I)\b|THE\s+BOY\s+WHO\s+LIVED|PROLOGUE|BOOK\s+(ONE|1|I)", text[:40000], re.IGNORECASE):
        start = match.start()
        window = text[start:start + 2000]
        # TOC detector heuristic
        if window.count("CHAPTER") > 3 or "....." in window or "..." in window[:100]:
            continue
        candidates.append(start)
    return candidates[0] if candidates else 0


def clean_book_text(raw_text: str) -> list[str]:
    if not raw_text or not raw_text.strip():
        return []

    normalized = _normalize_text(raw_text)

    # 1. Truncate everything before the first actual story start marker using smart TOC detection
    start_idx = find_story_start(normalized)
    if start_idx > 0:
        normalized = normalized[start_idx:]
    else:
        # Generic fallback: drop early blocks until we hit the first long narrative paragraph
        blocks_split = normalized.split("\n\n")
        fallback_start = 0
        for idx, block in enumerate(blocks_split[:15]):
            words = block.split()
            if len(words) >= 15 and re.search(r'[.!?؟""\u201d]$', block.strip()):
                fallback_start = idx
                break
        if fallback_start > 0:
            normalized = "\n\n".join(blocks_split[fallback_start:])

    # Split text into paragraphs first to detect and prune TOC/front matter at block level
    blocks = [b.strip() for b in normalized.split("\n\n") if b.strip()]
    cleaned_blocks = []
    
    for idx, block in enumerate(blocks):
        # Scan blocks for TOC and front matter without arbitrary index limits
        if _is_toc_or_front_matter(block):
            continue
        sanitized = sanitize_scene_text(block)
        if sanitized:
            cleaned_blocks.append(sanitized)
        
    text = "\n\n".join(cleaned_blocks)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    cleaned_lines: list[str] = []
    for line in lines:
        if _is_noise_line(line):
            continue
        if any(marker in line.lower() for marker in BAD_SCENE_MARKERS) and len(line.split()) < 16:
            continue
        lowered = line.lower()
        if len(line.split()) <= 7 and "harry potter" in lowered and any(sub in lowered for sub in ["stone", "chamber", "prisoner", "goblet", "order", "prince", "hallows"]):
            continue
        # Skip lines that are just chapter titles or structural noise
        # e.g., less than 6 words, all uppercase or title-cased, and no ending punctuation
        if len(line.split()) < 6 and not re.search(r'[.!?؟]$', line):
            if line.isupper() or all(w[0].isupper() for w in line.split() if w.isalpha()):
                continue
        cleaned_lines.append(line)

    paragraphs: list[str] = []
    buffer: list[str] = []
    word_count = 0
    target_words = 120

    for line in cleaned_lines:
        words = line.split()
        buffer.append(line)
        word_count += len(words)

        if word_count >= target_words and re.search(r'[.!?؟""\u201d]$', line):
            paragraph = " ".join(buffer).strip()
            paragraph = sanitize_scene_text(paragraph)
            if len(paragraph.split()) >= MIN_SCENE_WORDS and not _is_toc_or_front_matter(paragraph):
                paragraphs.append(paragraph)
            buffer = []
            word_count = 0

    if buffer:
        paragraph = " ".join(buffer).strip()
        paragraph = sanitize_scene_text(paragraph)
        if len(paragraph.split()) >= MIN_SCENE_WORDS and not _is_toc_or_front_matter(paragraph):
            paragraphs.append(paragraph)

    if not paragraphs and cleaned_lines:
        fallback = " ".join(cleaned_lines).strip()
        fallback = sanitize_scene_text(fallback)
        if fallback and not _is_toc_or_front_matter(fallback):
            paragraphs = [fallback]

    return paragraphs


def segment_scenes(paragraphs: list[str], window: int = 3) -> list[str]:
    """Window=3 gives richer context per scene for better visual descriptions."""
    scenes: list[str] = []
    for index in range(0, len(paragraphs), window):
        chunk = " ".join(paragraphs[index:index + window]).strip()
        if len(chunk.split()) >= MIN_SCENE_WORDS:
            scenes.append(chunk)
    return scenes


# ---------------------------------------------------------------------------
# Character extraction
# ---------------------------------------------------------------------------

def extract_characters_regex(text: str, max_names: int = 6) -> list[str]:
    titled = re.findall(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Professor|Uncle|Aunt)\.?\s+[A-Z][a-z]+\b", text)
    families = re.findall(r"\b[Tt]he\s+[A-Z][a-z]+s\b", text)
    singles = re.findall(r"\b[A-Z][a-z]{2,}\b", text)

    out: list[str] = []

    def add(value: str) -> None:
        if value and value not in out and len(out) < max_names:
            out.append(value)

    for value in titled:
        add(value)
    for value in families:
        add(value)
    for value in singles:
        if value in NAME_STOP:
            continue
        add(value)

    final: list[str] = []
    seen_lower: set[str] = set()
    text_lower = text.lower()
    for value in out:
        normalized = value
        if re.fullmatch(r"[A-Z][a-z]+s", value) and f"the {value.lower()}" in text_lower:
            normalized = f"the {value}"
        lowered = normalized.lower()
        if lowered not in seen_lower:
            final.append(normalized)
            seen_lower.add(lowered)
    return final[:max_names]


def _extract_characters(scene_text: str, max_names: int = 4) -> list[str]:
    return extract_characters_regex(scene_text, max_names=max_names)


# ---------------------------------------------------------------------------
# Mood & setting helpers
# ---------------------------------------------------------------------------

def _guess_mood(scene_text: str) -> str:
    lowered = scene_text.lower()
    for mood, keywords in MOOD_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            return mood
    return "neutral"


def _normalize_mood_label(mood: str) -> str:
    lowered = (mood or "neutral").strip().lower()
    mood_map = {
        "fearful": "tense and uneasy",
        "ominous": "dark and threatening",
        "mysterious": "quietly mysterious",
        "joyful": "light and warm",
        "calm": "quiet and restrained",
        "angry": "frustrated and sharp",
        "sad": "somber and intimate",
        "neutral": "subtle and observational",
        "tense": "tense and uneasy",
    }
    return mood_map.get(lowered, lowered or "subtle and observational")


def _guess_setting(scene_text: str) -> str:
    """Extract setting with multiple fallback patterns."""
    # Pattern 1: preposition + proper noun location (excluding common character names)
    match = re.search(
        r"\b(in|at|inside|outside|near|on|through|into|across)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,4})",
        scene_text,
    )
    if match:
        location_name = match.group(2).strip()
        # Ignore if it contains a known character name (substring or possessive) to avoid character-as-location clash
        if not any(char in location_name.lower() for char in {"harry", "ron", "hermione", "dudley", "vernon", "petunia", "dursley", "potter", "lily", "james", "snape", "dumbledore", "hagrid"}):
            return location_name

    # Pattern 2: known place types
    place_match = re.search(
        r"\b(kitchen|bedroom|hallway|street|corridor|garden|office|classroom|"
        r"forest|castle|dungeon|shop|store|hospital|station|platform|room|house|"
        r"school|library|tower|cave|field|alley|rooftop|cellar|attic|hall|"
        r"train|hut|cabin|church|temple|palace|bridge|river|lake|mountain|"
        r"market|inn|tavern|prison|yard|courtyard|gate|path|road|village|city)\b",
        scene_text, re.IGNORECASE,
    )
    if place_match:
        return f"a {place_match.group(1).lower()}"

    return "an interior room"


def _normalize_setting(setting: str) -> str:
    cleaned = re.sub(r"\s+", " ", (setting or "an interior scene")).strip().strip(".")
    if cleaned.lower() in {"a scene", "scene", "unknown"}:
        return "an interior scene"
    if len(cleaned) > 120:
        return cleaned[:117].rstrip() + "..."
    return cleaned


# ---------------------------------------------------------------------------
# Beat / action extraction
# ---------------------------------------------------------------------------

def _extract_beats(scene_text: str, max_beats: int = 4) -> list[str]:
    protected = scene_text.strip()
    title_placeholders: dict[str, str] = {}
    for title in ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof."):
        placeholder = title.replace(".", "<DOT>")
        title_placeholders[placeholder] = title
        protected = protected.replace(title, placeholder)

    sentences = re.split(r"(?<=[.!?؟])\s+", protected)
    beats: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip().strip('"')
        for placeholder, title in title_placeholders.items():
            cleaned = cleaned.replace(placeholder, title)
        if not cleaned:
            continue
        if _is_noise_line(cleaned):
            continue
        beats.append(cleaned)
        if len(beats) >= max_beats:
            break
    return beats


def _select_primary_action(beats: list[str]) -> str:
    """Select the best action-rich sentence from the beats."""
    best_action = ""
    best_score = -1
    for beat in beats:
        lowered = beat.lower()
        if any(marker in lowered for marker in BAD_SCENE_MARKERS):
            continue
        if len(beat.split()) < 5:
            continue
        action = beat.rstrip(".")
        if len(action) > MAX_ACTION_CHARS:
            action = action[:MAX_ACTION_CHARS - 3].rstrip() + "..."
        score = 0
        if any(keyword in lowered for keyword in ACTION_KEYWORDS):
            score += 4
        if any(pattern in lowered for pattern in DESCRIPTION_ONLY_PATTERNS):
            score -= 2
        if len(action.split()) <= 25:
            score += 1
        if score > best_score:
            best_action = action
            best_score = score
    return best_action or "A key moment unfolds"
def _detect_action_type(scene_text: str, action: str) -> str:
    """Detect the dominant action type in the scene, prioritizing current visual action over past narrative text."""
    if not action:
        return "idle"

    t = action.lower()

    if re.search(r"\b(sit|sat|sitting|seated|settle|settles|settled|traffic|car|vehicle|armchair|chair)\b", t):
        return "sitting"
    if re.search(r"\b(run|runs|running|ran|sprint|sprints|sprinting|dash|dashes|dashing|hurry|hurries|hurried|hurrying|rush|rushes|rushed|rushing|flee|flees|fleeing|fled|chase|chases|chasing)\b", t):
        return "running"
    if re.search(r"\b(walk|walks|walking|walked|stride|strides|striding|stroll|strolls|strolling|march|marches|marching|pace|paces|pacing|step|steps|stepped|stepping)\b", t):
        return "walking"
    if re.search(r"\b(speak|speaks|speaking|spoke|say|says|said|talk|talks|talking|conversation|conversations|dialogue|dialogues|gesturing|gestures|gestured|mouth|mouths)\b", t):
        return "dialogue"
    if re.search(r"\b(look|looks|looking|looked|stare|stares|staring|stared|gaze|gazes|gazing|gazed|watch|watches|watching|watched|peer|peers|peering|peered|glance|glances|glancing|glanced)\b", t):
        return "observation"

    return "idle"


def get_camera_style(action_type: str) -> str:
    mapping = {
        "running": "handheld tracking shot, dynamic pursuit camera",
        "walking": "low-angle follow shot, smooth tracking",
        "sitting": "medium close-up, subtle push-in",
        "dialogue": "over-the-shoulder shot, shallow depth of field",
        "observation": "slow zoom-in, tight framing",
        "idle": "static cinematic framing with slight drift"
    }
    return mapping.get(action_type, "dynamic cinematic tracking shot")


def get_motion_directive(action_type: str) -> str:
    mapping = {
        "running": "urgent running motion, arms pumping, clothing billowing, rapid leg movement, environment blurring past",
        "walking": "full walking cycle with arm swing, weight shifting between steps, clothing swaying with movement, hair bouncing",
        "sitting": "natural fidgeting, shifting weight, hand gestures, slight body turns, breathing movement",
        "dialogue": "natural gesturing while speaking, head movements, facial expression changes, subtle body shifts",
        "observation": "visible head turn with body following, eyes tracking, slight body lean, natural reaction movement",
        "idle": "continuous body movement, natural motion cycle, visible physical action, environment reacting to character"
    }
    return mapping.get(action_type, "continuous body movement, natural motion cycle, visible physical action")


def _strip_subject_from_action(action: str, subject_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", action or "").strip()
    subject = re.escape(subject_name.strip())
    if not cleaned or not subject_name.strip():
        return cleaned
    cleaned = re.sub(rf"^{subject}\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(rf"^(he|she|they)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:1].lower() + cleaned[1:] if cleaned else action


# ---------------------------------------------------------------------------
# Context collectors for Bible generation
# ---------------------------------------------------------------------------

def _character_contexts(dataset: list[dict], max_context_chars: int = 1200) -> dict[str, str]:
    """Collect text excerpts for each character across all scenes."""
    contexts: dict[str, list[str]] = {}
    for scene in dataset:
        scene_text = str(scene.get("scene_text", ""))
        characters = scene.get("analysis", {}).get("characters", []) or []
        for character in characters:
            if not isinstance(character, str) or not character.strip():
                continue
            contexts.setdefault(character, [])
            if len(" ".join(contexts[character])) < max_context_chars:
                contexts[character].append(scene_text[:600])
    return {
        character: " ".join(parts)[:max_context_chars]
        for character, parts in contexts.items()
    }


def _location_contexts(dataset: list[dict], max_context_chars: int = 1200) -> dict[str, str]:
    """Collect text excerpts for each unique setting/location across all scenes."""
    contexts: dict[str, list[str]] = {}
    for scene in dataset:
        scene_text = str(scene.get("scene_text", ""))
        setting = scene.get("analysis", {}).get("setting", "")
        if not setting or setting.lower() in {"a scene", "unknown", "an interior room"}:
            continue
        # Normalize location key
        loc_key = setting.lower().strip()
        contexts.setdefault(loc_key, [])
        if len(" ".join(contexts[loc_key])) < max_context_chars:
            contexts[loc_key].append(scene_text[:600])
    return {
        loc: " ".join(parts)[:max_context_chars]
        for loc, parts in contexts.items()
    }


# ---------------------------------------------------------------------------
# Regex fallback Character Bible (when Mistral unavailable)
# ---------------------------------------------------------------------------

KNOWN_CHARACTER_PROFILES = {
    "harry": "Harry Potter, a young boy with round glasses, messy black hair, a faint lightning-bolt scar on his forehead, and wearing school robes",
    "harry potter": "Harry Potter, a young boy with round glasses, messy black hair, a faint lightning-bolt scar on his forehead, and wearing school robes",
    "dumbledore": "Albus Dumbledore, an elderly wizard with a long silver beard and hair, half-moon spectacles, and long sweeping robes",
    "albus dumbledore": "Albus Dumbledore, an elderly wizard with a long silver beard and hair, half-moon spectacles, and long sweeping robes",
    "voldemort": "Lord Voldemort, a pale, snake-like figure with red eyes and no nose, wrapped in dark flowing cloaks",
    "you-know-who": "Lord Voldemort, a pale, snake-like figure with red eyes and no nose, wrapped in dark flowing cloaks",
    "ron": "Ron Weasley, a tall, lanky young boy with freckles and bright red hair, wearing school robes",
    "ron weasley": "Ron Weasley, a tall, lanky young boy with freckles and bright red hair, wearing school robes",
    "hermione": "Hermione Granger, a young girl with bushy brown hair and an earnest expression, holding a stack of books",
    "hermione granger": "Hermione Granger, a young girl with bushy brown hair and an earnest expression, holding a stack of books",
    "snape": "Severus Snape, a pale man with shoulder-length greasy black hair, dark eyes, and a long black cloak",
    "severus snape": "Severus Snape, a pale man with shoulder-length greasy black hair, dark eyes, and a long black cloak",
    "hagrid": "Rubeus Hagrid, a giant man with a wild, bushy beard and hair, wearing a massive moleskin coat",
    "rubeus hagrid": "Rubeus Hagrid, a giant man with a wild, bushy beard and hair, wearing a massive moleskin coat",
    "mrs. dursley": "Petunia Dursley, a thin woman with a long neck, blonde hair, and a sharp disapproving expression",
    "petunia dursley": "Petunia Dursley, a thin woman with a long neck, blonde hair, and a sharp disapproving expression",
    "mr. dursley": "Vernon Dursley, a heavy-set, beefy man with a large mustache and almost no neck",
    "vernon dursley": "Vernon Dursley, a heavy-set, beefy man with a large mustache and almost no neck",
    "dudley": "Dudley Dursley, a blonde, portly young boy with a spoiled expression",
    "dudley dursley": "Dudley Dursley, a blonde, portly young boy with a spoiled expression",
    "mcgonagall": "Professor McGonagall, a stern-looking woman with dark hair pulled into a tight bun, glasses, and square green robes",
    "professor mcgonagall": "Professor McGonagall, a stern-looking woman with dark hair pulled into a tight bun, glasses, and square green robes",
}

KNOWN_LOCATION_PROFILES = {
    "privet drive": "number four Privet Drive, identical suburban brick houses, neatly trimmed hedges, manicured lawns, quiet street",
    "number four": "number four Privet Drive, a tidy suburban house interior, neat floral curtains, polished wooden furniture",
    "hogwarts": "Hogwarts Castle, towering stone spires, massive gothic architecture, stained glass windows, magical atmosphere",
    "gryffindor common room": "Gryffindor common room, warm crackling fireplace, plush scarlet armchairs, high arched windows, tapestries on stone walls",
    "diagon alley": "Diagon Alley, bustling cobblestone street, lopsided whimsical shops, colorful magical storefronts, broomsticks and cauldrons on display",
    "platform 9": "Platform Nine and Three-Quarters, bustling steam train station, Hogwarts Express locomotive, brick archways, billowing steam",
    "platform nine": "Platform Nine and Three-Quarters, bustling steam train station, Hogwarts Express locomotive, brick archways, billowing steam",
    "great hall": "the Great Hall of Hogwarts, floating candles, high starry night sky ceiling, long oak tables, gothic stone walls",
    "hagrid's hut": "Hagrid's hut, a small rustic stone cabin, smoking chimney, pumpkin patch outside, wooden clutter inside",
}


def _build_character_anchor_fallback(name: str, context: str) -> str:
    """Build a character visual description using regex or known profiles when LLM is unavailable."""
    name_clean = name.strip().lower()
    if name_clean in KNOWN_CHARACTER_PROFILES:
        return KNOWN_CHARACTER_PROFILES[name_clean]
    for k, profile in KNOWN_CHARACTER_PROFILES.items():
        if k in name_clean or name_clean in k:
            return profile

    lowered = f"{name} {context}".lower()
    name_lower = name.lower()
    parts: list[str] = []

    # Age/gender detection
    if any(title in name_lower for title in ["aunt ", "mrs", "miss ", "ms."]):
        parts.append("adult woman")
    elif any(title in name_lower for title in ["uncle ", "mr.", "mr ", "professor "]):
        parts.append("adult man")
    elif any(word in lowered for word in ["boy", "schoolboy", "lad", "son"]):
        parts.append("young boy")
    elif any(word in lowered for word in ["girl", "daughter"]):
        parts.append("young girl")

    # Build
    if any(word in lowered for word in ["big", "beefy", "large", "heavyset", "fat", "huge", "enormous"]):
        parts.append("heavyset build")
    elif any(word in lowered for word in ["thin", "skinny", "narrow", "slim", "slender", "gaunt"]):
        parts.append("thin build")
    elif any(word in lowered for word in ["tall", "towering"]):
        parts.append("tall build")

    # Hair
    if "black hair" in lowered or "dark hair" in lowered:
        parts.append("dark hair")
    elif "red hair" in lowered or "ginger" in lowered:
        parts.append("red hair")
    elif "blond" in lowered or "blonde" in lowered:
        parts.append("blond hair")
    elif "brown hair" in lowered:
        parts.append("brown hair")
    elif "white hair" in lowered or "silver hair" in lowered or "grey hair" in lowered:
        parts.append("silver-white hair")
    if "bushy hair" in lowered or "wild hair" in lowered or "messy hair" in lowered:
        parts.append("untidy hair")
    if "long hair" in lowered:
        parts.append("long hair")
    if "curly" in lowered:
        parts.append("curly hair")

    # Face
    if "mustache" in lowered or "moustache" in lowered:
        parts.append("mustache")
    if "glasses" in lowered or "spectacles" in lowered:
        parts.append("glasses")
    if "beard" in lowered:
        parts.append("beard")
    if "scar" in lowered:
        parts.append("visible scar")
    if "freckle" in lowered:
        parts.append("freckled face")
    if "pale" in lowered:
        parts.append("pale complexion")

    # Clothing
    if any(word in lowered for word in ["suit", "business", "tie", "formal"]):
        parts.append("formal business clothes")
    elif any(word in lowered for word in ["cloak", "robe", "robes", "wizard", "gown"]):
        parts.append("long robes")
    elif any(word in lowered for word in ["uniform", "school"]):
        parts.append("school uniform")
    elif any(word in lowered for word in ["armor", "armour", "chainmail"]):
        parts.append("wearing armor")

    if not parts:
        return name
    stable_parts = list(dict.fromkeys(parts))[:6]
    return f"{name}, " + ", ".join(stable_parts)


def _build_character_registry_fallback(dataset: list[dict]) -> dict[str, str]:
    """Build character registry using regex when LLM is unavailable."""
    contexts = _character_contexts(dataset)
    return {
        character: _build_character_anchor_fallback(character, context)
        for character, context in contexts.items()
    }


# ---------------------------------------------------------------------------
# Regex fallback Location Bible (when Mistral unavailable)
# ---------------------------------------------------------------------------

def _build_location_anchor_fallback(scene_text: str, setting: str) -> str:
    """Build location description using regex or known location profiles when LLM is unavailable."""
    setting_clean = setting.strip().lower()
    lowered = scene_text.lower()

    # 1. Prioritize checking scene text and setting for KNOWN locations first!
    if "privet" in setting_clean or "privet" in lowered:
        return KNOWN_LOCATION_PROFILES["privet drive"]
    if "number four" in setting_clean or "number four" in lowered:
        return KNOWN_LOCATION_PROFILES["number four"]
    if "hogwarts" in setting_clean or "hogwarts" in lowered:
        return KNOWN_LOCATION_PROFILES["hogwarts"]
    if "diagon alley" in setting_clean or "diagon alley" in lowered:
        return KNOWN_LOCATION_PROFILES["diagon alley"]
    if "platform 9" in setting_clean or "platform 9" in lowered or "platform nine" in setting_clean or "platform nine" in lowered:
        return KNOWN_LOCATION_PROFILES["platform 9"]
    if "great hall" in setting_clean or "great hall" in lowered:
        return KNOWN_LOCATION_PROFILES["great hall"]
    if "gryffindor" in setting_clean or "gryffindor" in lowered:
        return KNOWN_LOCATION_PROFILES["gryffindor common room"]
    if "hagrid" in setting_clean or "hagrid" in lowered:
        return KNOWN_LOCATION_PROFILES["hagrid's hut"]

    # 2. Check direct setting matches for profiles
    for k, profile in KNOWN_LOCATION_PROFILES.items():
        if k in setting_clean or setting_clean in k:
            return profile

    details: list[str] = []

    # Use re.search with word boundaries to avoid matching substrings in other words (e.g., "wood" in Oliver Wood)
    if re.search(r'\boffice\b', lowered):
        details.extend(["an office interior", "cluttered desk", "large window", "morning light"])
    elif re.search(r'\b(kitchen|breakfast|dining room)\b', lowered):
        details.extend(["a kitchen interior", "table set for meal", "warm domestic light"])
    elif re.search(r'\b(street|road|drive|alley|lane)\b', lowered):
        details.extend(["a street exterior", "buildings on both sides", "natural daylight"])
    elif re.search(r'\b(forest|woods?|jungle|clearing)\b', lowered):
        details.extend(["a forest clearing", "tall trees", "dappled light through canopy", "misty atmosphere"])
    elif re.search(r'\b(hall|corridor|passage)\b', lowered):
        details.extend(["a grand hall interior", "high ceiling", "ornate architecture", "warm light"])
    elif re.search(r'\b(train|carriage|compartment)\b', lowered):
        details.extend(["a train compartment interior", "bench seats", "countryside passing window"])
    elif re.search(r'\b(castle|tower|dungeon)\b', lowered):
        details.extend(["a stone castle interior", "torchlit corridors", "medieval architecture"])
    elif re.search(r'\b(hut|cabin|shack)\b', lowered):
        details.extend(["a small wooden cabin", "rustic interior", "dim practical lighting"])
    elif re.search(r'\b(classroom|school|lab)\b', lowered):
        details.extend(["a classroom interior", "desks arranged in rows", "chalkboard or equipment"])
    elif re.search(r'\b(library|archives)\b', lowered):
        details.extend(["a library interior", "towering bookshelves", "reading tables", "quiet dusty atmosphere"])
    elif re.search(r'\b(dungeon|cellar|basement)\b', lowered):
        details.extend(["an underground space", "stone walls", "dim cold lighting", "damp atmosphere"])
    elif re.search(r'\b(room|bedroom|living room|parlor)\b', lowered):
        details.extend(["a room interior", "consistent natural lighting", "visible windows"])
    elif re.search(r'\b(garden|park|lawn|field|yard)\b', lowered):
        details.extend(["an outdoor garden", "plants and greenery", "natural daylight"])
    else:
        details.append(_normalize_setting(setting))
        details.append("consistent natural lighting")

    # Add environmental details from text
    if "window" in lowered:
        details.append("visible window")
    if "candle" in lowered or "candlelight" in lowered:
        details.append("warm candlelight")
    if "fire" in lowered or "fireplace" in lowered:
        details.append("fireplace glow")
    if "rain" in lowered:
        details.append("rain visible outside")
    if "snow" in lowered:
        details.append("snow visible")
    if "night" in lowered or "moon" in lowered:
        details.append("nighttime atmosphere")

    return ", ".join(dict.fromkeys(details))


# ---------------------------------------------------------------------------
# Scene quality scoring
# ---------------------------------------------------------------------------

def _scene_quality_score(scene_text: str, analysis: dict) -> int:
    score = 0
    words = scene_text.split()
    if len(words) >= MIN_SCENE_WORDS:
        score += 2
    if not any(marker in scene_text.lower() for marker in BAD_SCENE_MARKERS):
        score += 2
    if analysis.get("characters"):
        score += 1
    if analysis.get("beats"):
        score += 2
    if analysis.get("setting") not in {"a scene", "Unknown", "an interior room"}:
        score += 1
    return score


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset_from_scenes(scenes: list[str]) -> list[dict]:
    dataset: list[dict] = []
    for index, scene_text in enumerate(scenes):
        llm_analysis = analyze_scene_with_mistral(scene_text) if mistral_available() else None
        analysis = {
            "setting": (llm_analysis or {}).get("setting", _guess_setting(scene_text)),
            "mood": (llm_analysis or {}).get("mood", _guess_mood(scene_text)),
            "characters": _extract_characters(scene_text),
            "beats": _extract_beats(scene_text),
        }
        if _scene_quality_score(scene_text, analysis) < 5:
            continue
        dataset.append({
            "scene_id": f"scene_{index}",
            "scene_text": scene_text,
            "analysis": analysis,
        })
    return dataset


# ---------------------------------------------------------------------------
# Bible generation (LLM with regex fallback)
# ---------------------------------------------------------------------------

def build_character_bible(dataset: list[dict]) -> dict[str, str]:
    """
    Build character visual descriptions for all characters in the dataset.
    Uses LLM when available, falls back to regex-based extraction.
    """
    char_contexts = _character_contexts(dataset)
    if not char_contexts:
        return {}

    # Try LLM first
    llm_bible = generate_character_bible_with_mistral(char_contexts)
    if llm_bible:
        # Merge: LLM results + fallback for any characters LLM missed
        fallback_bible = _build_character_registry_fallback(dataset)
        merged = dict(fallback_bible)
        merged.update(llm_bible)  # LLM overrides fallback
        return merged

    # Full fallback
    return _build_character_registry_fallback(dataset)


def build_location_bible(dataset: list[dict]) -> dict[str, str]:
    """
    Build location visual descriptions for all locations in the dataset.
    Uses LLM when available, falls back to regex-based extraction.
    """
    loc_contexts = _location_contexts(dataset)
    if not loc_contexts:
        return {}

    # Try LLM first
    llm_bible = generate_location_bible_with_mistral(loc_contexts)
    if llm_bible:
        return llm_bible

    # Full fallback: generate from scene texts
    fallback: dict[str, str] = {}
    for scene in dataset:
        scene_text = scene.get("scene_text", "")
        setting = scene.get("analysis", {}).get("setting", "")
        loc_key = setting.lower().strip()
        if loc_key and loc_key not in fallback:
            fallback[loc_key] = _build_location_anchor_fallback(scene_text, setting)
    return fallback


def build_visual_style_bible(all_scenes: list[str], dataset: list[dict], genre: str) -> dict[str, str]:
    """
    Build a Visual Style Bible from a wide sample of book text (first 20-30 scenes).
    Uses LLM when available, falls back to genre-based heuristics.
    """
    # Collect text from first 30 scenes (or all if fewer) for maximum context
    bible_sample_scenes = all_scenes[:30]
    sample_text = " ".join(s[:400] for s in bible_sample_scenes)

    # Try LLM
    llm_bible = generate_visual_style_bible_with_mistral(sample_text)
    if llm_bible and len(llm_bible) >= 3:
        return llm_bible

    # Regex/heuristic fallback based on genre and text keywords
    return _infer_visual_style_fallback(sample_text, genre)


def _infer_visual_style_fallback(sample_text: str, genre: str) -> dict[str, str]:
    """Infer visual style from genre and text keywords when LLM is unavailable."""
    lowered = sample_text.lower()

    # Period detection
    period = "contemporary"
    if any(w in lowered for w in ["wand", "wizard", "witch", "magic", "spell", "potion", "hogwarts"]):
        period = "1990s magical Britain"
    elif any(w in lowered for w in ["castle", "sword", "knight", "king", "queen", "throne", "medieval"]):
        period = "medieval era"
    elif any(w in lowered for w in ["victorian", "carriage", "gaslight", "fog", "london", "detective"]):
        period = "Victorian era"
    elif any(w in lowered for w in ["spaceship", "galaxy", "planet", "laser", "android", "cyborg"]):
        period = "far future sci-fi"
    elif any(w in lowered for w in ["desert", "sand", "dune", "oasis", "caravan"]):
        period = "ancient desert civilization"

    # Genre-based defaults
    style_map = {
        "fantasy": {
            "atmosphere": "warm magical",
            "lighting": "warm practical light, candles and torches",
            "color_palette": "warm golds, deep reds, rich browns",
            "visual_tone": "epic fantasy film",
            "architecture": "gothic stone castles and towers",
            "weather": "dramatic skies, occasional storms",
        },
        "gothic": {
            "atmosphere": "dark brooding",
            "lighting": "low-key shadows, moonlight",
            "color_palette": "cold blues, dark greys, deep purples",
            "visual_tone": "dark atmospheric gothic",
            "architecture": "decaying mansions and old castles",
            "weather": "overcast, foggy, rain",
        },
        "horror": {
            "atmosphere": "dread and tension",
            "lighting": "harsh shadows, flickering light",
            "color_palette": "desaturated cold tones, deep shadows",
            "visual_tone": "psychological horror film",
            "architecture": "isolated buildings, cramped dark spaces",
            "weather": "dark and stormy",
        },
        "mystery": {
            "atmosphere": "suspenseful and secretive",
            "lighting": "pools of light in shadow, motivated practicals",
            "color_palette": "muted neutral tones, amber highlights",
            "visual_tone": "detective mystery film",
            "architecture": "urban interiors, offices, back alleys",
            "weather": "overcast and grey",
        },
        "romance": {
            "atmosphere": "warm and intimate",
            "lighting": "soft golden hour light, warm practicals",
            "color_palette": "warm pastels, soft pinks and golds",
            "visual_tone": "romantic drama film",
            "architecture": "charming houses, gardens, cafes",
            "weather": "sunny with gentle breezes",
        },
        "adventure": {
            "atmosphere": "exciting and vast",
            "lighting": "bright natural daylight, high clarity",
            "color_palette": "vibrant greens, sky blues, earthy browns",
            "visual_tone": "epic adventure film",
            "architecture": "varied landscapes, ruins, wild terrain",
            "weather": "dynamic, changing conditions",
        },
        "thriller": {
            "atmosphere": "tense and urgent",
            "lighting": "high contrast, sharp practicals",
            "color_palette": "cool steely tones, neon accents",
            "visual_tone": "tense thriller film",
            "architecture": "modern urban, industrial",
            "weather": "night time, rain",
        },
        "drama": {
            "atmosphere": "intimate and emotional",
            "lighting": "soft naturalistic light",
            "color_palette": "earthy warm tones, realistic skin tones",
            "visual_tone": "character-driven drama film",
            "architecture": "domestic interiors, everyday environments",
            "weather": "ordinary, matching emotional tone",
        },
    }

    base = style_map.get(genre, style_map["drama"])
    base["period"] = period
    return base


# ---------------------------------------------------------------------------
# Scene → video item (core transformation)
# ---------------------------------------------------------------------------

def scene_to_video_item(
    scene: dict,
    scene_id: str,
    book_profile: dict,
    character_bible: dict[str, str],
    location_bible: dict[str, str],
) -> dict:
    analysis = scene.get("analysis", {}) or {}
    setting = _normalize_setting(str(analysis.get("setting", "an interior scene")))
    mood = str(analysis.get("mood", "neutral")).strip().lower()
    chars = analysis.get("characters", []) or []
    beats = analysis.get("beats", []) or []
    scene_text = scene.get("scene_text", "")
    duration = min(int(book_profile.get("scene_seconds", MAX_SCENE_SECONDS)), MAX_SCENE_SECONDS)

    # --- 1. Visual description (narrative_to_visual or fallback) ---
    visual_description = ""
    if scene_text:
        visual_description = narrative_to_visual(scene_text)

    # Fallback: use best action beat
    if not visual_description:
        raw_main_char = chars[0] if chars else "a character"
        action = _strip_subject_from_action(_select_primary_action(beats), raw_main_char)
        visual_description = action

    # --- 2. Character anchor from Bible ---
    main_char_name = chars[0] if chars else "a character"
    # Look up in Bible (case-insensitive)
    char_anchor = main_char_name
    for bible_name, bible_desc in character_bible.items():
        if bible_name.lower() == main_char_name.lower() or main_char_name.lower() in bible_name.lower():
            char_anchor = bible_desc
            break

    # --- 3. Location anchor from Bible ---
    setting_lower = setting.lower()
    location_anchor = setting
    for loc_key, loc_desc in location_bible.items():
        if loc_key in setting_lower or setting_lower in loc_key:
            location_anchor = loc_desc
            break
    # If no Bible match, use regex fallback
    if location_anchor == setting:
        location_anchor = _build_location_anchor_fallback(scene_text, setting)

    # --- 4. Detect action type for camera/motion ---
    action_type = _detect_action_type(scene_text, visual_description)

    # --- 5. Dynamic camera based on action + mood ---
    lowered_visual = visual_description.lower()
    if "traffic" in lowered_visual or "car" in lowered_visual or "steering wheel" in lowered_visual:
        camera = "dashboard perspective looking out, traffic visible through windshield"
    elif mood in {"mysterious", "ominous", "tense", "fearful"}:
        if action_type == "sitting":
            camera = "tense close-up, slow creeping dolly in, focus on character's eyes"
        elif action_type == "walking":
            camera = "low-angle tracking shot following the character, steady camera, pockets of shadow"
        elif action_type == "running":
            camera = "urgent handheld camera, tracking shot with motion blur, tense framing"
        elif action_type == "observation":
            camera = "slow orbit around the subject, revealing their startled expression, shallow focus"
        elif action_type == "dialogue":
            camera = "over-the-shoulder shot, tight framing, slow creeping push-in"
        else:
            camera = "slow creeping dolly in, restrained handheld micro-movement, tense framing"
    elif mood in {"joyful", "calm"} and action_type == "idle":
        camera = "gentle smooth pan, wider lens, airy framing"
    else:
        camera = get_camera_style(action_type)

    # --- 6. Dynamic motion based on action type ---
    motion_directive = get_motion_directive(action_type)

    # --- 7. Environment hints ---
    env_hints = _extract_environment_hints(scene_text, setting)

    return {
        "scene_id": scene_id,
        "shot_id": 1,
        "duration": duration,
        "camera": camera,
        "subject": char_anchor,
        "setting": setting,
        "location_anchor": location_anchor,
        "visual_description": visual_description,
        "environment": env_hints,
        "motion_directive": motion_directive,
        "mood": _normalize_mood_label(mood),
    }


def _extract_environment_hints(scene_text: str, setting: str) -> str:
    lowered = scene_text.lower()
    hints: list[str] = []
    if "window" in lowered:
        hints.append("large window")
    if "owl" in lowered or "owls" in lowered:
        hints.append("birds visible outside")
    if "street" in lowered:
        hints.append("street activity outside")
    if "train" in lowered:
        hints.append("train carriage movement")
    if "forest" in lowered:
        hints.append("dark trees in background")
    if "candle" in lowered:
        hints.append("floating candles")
    if "fire" in lowered or "fireplace" in lowered:
        hints.append("flickering firelight")
    if "rain" in lowered:
        hints.append("rain falling")
    if "wind" in lowered:
        hints.append("wind movement in hair and clothing")
    if "crowd" in lowered or "people" in lowered:
        hints.append("background crowd movement")
    if not hints:
        hints.append(setting)
    return ", ".join(dict.fromkeys(hints))


# ---------------------------------------------------------------------------
# Negative prompt
# ---------------------------------------------------------------------------

def build_negative_prompt(book_profile: dict) -> str:
    base = [
        "frozen frame", "still image", "slideshow look",
        "low quality", "blurry", "flicker", "jitter",
        "watermark", "logo", "subtitles", "text",
        "morphing", "ghosting", "warped motion", "deformed",
        "frame blending", "strobing",
        "painting", "illustration",
        "wax figure", "plastic skin", "melted face",
        "overly glossy skin", "distorted hands",
        "amateur video", "generic stock footage", "flat lighting",
    ]
    if not book_profile.get("allow_animation", False):
        base += ["cartoon", "anime", "stylized character", "toy-like character"]
    return ", ".join(base)


# ---------------------------------------------------------------------------
# Book profile helpers
# ---------------------------------------------------------------------------

def heuristic_book_profile(scenes: Iterable[str]) -> dict:
    text = " ".join([scene for scene in list(scenes)[:120] if isinstance(scene, str)]).lower()
    fantasy = ["magic", "wizard", "spell", "wand", "enchanted", "dragon", "fairy", "mermaid", "curse"]
    childlike = ["nursery", "bedtime", "toy", "storybook", "mother", "father", "schoolboy"]
    fscore = sum(word in text for word in fantasy)
    cscore = sum(word in text for word in childlike)

    if fscore >= 2:
        return {
            "audience": "general",
            "visual_style": "cinematic fantasy film",
            "realism_level": "grounded fantasy realism",
            "color_palette": "rich warm magical tones",
            "scene_seconds": MAX_SCENE_SECONDS,
            "allow_animation": False,
        }
    if cscore >= 2:
        return {
            "audience": "children",
            "visual_style": "warm storybook live-action",
            "realism_level": "stylized but grounded",
            "color_palette": "bright warm pastels",
            "scene_seconds": MAX_SCENE_SECONDS,
            "allow_animation": True,
        }
    return {}


def _detect_genre_fallback(dataset: list[dict]) -> str:
    text = " ".join(item.get("scene_text", "") for item in dataset[:40]).lower()
    
    # Fail-safe override for Harry Potter
    if "harry potter" in text or "dumbledore" in text or "privet drive" in text:
        return "fantasy"

    scores = {
        "fantasy":   sum(word in text for word in ["magic", "wizard", "spell", "dragon", "enchanted", "curse", "wand", "witch", "potion", "cloak", "owl", "muggle", "hogwarts", "castle"]),
        "adventure": sum(word in text for word in ["journey", "quest", "danger", "travel", "explore", "battle"]),
        "mystery":   sum(word in text for word in ["secret", "clue", "mystery", "strange", "unknown", "hidden"]),
        "horror":    sum(word in text for word in ["blood", "monster", "fear", "terror", "scream", "grave"]),
        "gothic":    sum(word in text for word in ["castle", "gloom", "haunted", "shadow", "ominous", "melancholy"]),
        "romance":   sum(word in text for word in ["love", "kiss", "heart", "beloved", "longing", "romance"]),
        "thriller":  sum(word in text for word in ["chase", "pursuit", "threat", "danger", "escape", "suspense"]),
        "drama":     sum(word in text for word in ["family", "conflict", "emotion", "relationship", "struggle", "life"]),
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
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        label_vecs = embedder.encode([GENRE_DEFS[label] for label in CANDIDATE_LABELS], normalize_embeddings=True)
        text_vecs = embedder.encode(texts, normalize_embeddings=True)

        avg_scores: dict[str, float] = {}
        for label_index, label in enumerate(CANDIDATE_LABELS):
            avg_scores[label] = sum(float(vec @ label_vecs[label_index]) for vec in text_vecs) / max(1, len(text_vecs))

        ranked = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)
        primary, score = ranked[0]
        return primary if score >= 0.35 else "drama"
    except Exception:
        return _detect_genre_fallback(dataset)


def apply_genre_preset(base_style: dict, active_genre: str) -> dict:
    style = dict(base_style)
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

def apply_context_override(prompt: str) -> str:
    p = prompt.lower()
    # Check if there are modern/domestic/suburban keywords
    if any(k in p for k in ["street", "car", "drive", "kitchen", "bedroom", "house", "garden", "suburban", "road", "hallway", "office"]):
        # 1. Modify architecture
        prompt = prompt.replace(
            "gothic stone castles and towers",
            "neat British suburban architecture, cozy domestic interiors, realistic 1990s England"
        )
        # 2. Add lighting layers
        if "candle" in p or "torch" in p or "night" in p:
            prompt = prompt.replace("warm practical light, candles and torches", "warm domestic practical lighting, soft indoor lamps, evening atmosphere")
            prompt = prompt.replace("candles and torches", "warm domestic practical lighting, soft indoor lamps")
        else:
            prompt = prompt.replace("warm practical light, candles and torches", "warm domestic practical lighting, soft natural daylight")
            prompt = prompt.replace("candles and torches", "warm domestic practical lighting, soft natural daylight")
            
    return prompt


def clean_duplicate_prepositions(prompt: str) -> str:
    prompt = re.sub(r"\bin in\b", "in", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\bat at\b", "at", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\bthe the\b", "the", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\bof of\b", "of", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\bon on\b", "on", prompt, flags=re.IGNORECASE)
    return prompt


def wan_adapter(base_prompt: str, book_profile: dict, style_bible: dict[str, str] | None = None) -> str:
    style = book_profile.get("visual_style", "cinematic realism")
    realism = book_profile.get("realism_level", "realistic")
    palette = book_profile.get("color_palette", "natural tones")

    # Inject Visual Style Bible if available
    style_prefix = f"{style}, {realism}"
    if style_bible:
        tone = style_bible.get("visual_tone", "")
        period = style_bible.get("period", "")
        atmosphere = style_bible.get("atmosphere", "")
        architecture = style_bible.get("architecture", "")
        
        # Context-aware architecture check to avoid visual clash in domestic/suburban scenes
        base_lower = base_prompt.lower()
        if any(w in base_lower for w in ["kitchen", "bedroom", "room", "office", "car", "traffic", "street", "drive", "house"]):
            if "privet" in base_lower or "suburban" in base_lower:
                architecture = "neat British suburban architecture"
            else:
                architecture = "cozy domestic interiors"
                
        if tone:
            style_prefix = f"{tone}, {period}" if period else tone
        if atmosphere:
            style_prefix += f", {atmosphere} atmosphere"
        if architecture:
            style_prefix += f", {architecture}"

    return (
        f"{style_prefix}, coherent motion, stable camera, color palette: {palette}. "
        f"Natural human face, realistic skin texture, real clothing fabric, practical set design. "
        f"{base_prompt}"
    )


def compile_video_prompt_scene(
    item: dict,
    book_profile: dict,
    visual_style: dict,
    style_bible: dict[str, str] | None = None,
    scene_memory: dict | None = None,
) -> dict:
    subject = item.get("subject", "a character")
    location_anchor = item.get("location_anchor", item.get("setting", "an interior scene"))
    visual_description = item.get("visual_description", "A key moment unfolds")
    mood = item.get("mood", "subtle and observational")
    motion_directive = item.get("motion_directive", "continuous body movement, visible physical action")
    camera = item.get("camera", "dynamic medium tracking shot")
    environment = item.get("environment", "")

    # Preposition duplication check on location_anchor itself
    loc_clean = location_anchor.strip()
    if loc_clean.lower().startswith(("in ", "at ", "inside ", "on ", "into ")):
        loc_clean = re.sub(r'^(in|at|inside|on|into)\s+', '', loc_clean, flags=re.IGNORECASE)

    # Use Visual Style Bible for lighting/color if available, else genre preset
    lighting = visual_style['lighting']
    color = visual_style['color']
    if style_bible:
        bible_lighting = style_bible.get("lighting", "")
        bible_palette = style_bible.get("color_palette", "")
        bible_weather = style_bible.get("weather", "")
        if bible_lighting:
            lighting = bible_lighting
        if bible_palette:
            color = bible_palette
        if bible_weather and bible_weather not in environment:
            environment = f"{environment}, {bible_weather}" if environment else bible_weather

    # Continuity memory layer
    continuity_hints = []
    if scene_memory:
        # Check characters
        chars = item.get("analysis", {}).get("characters", []) or []
        for char in chars:
            if char.lower() in scene_memory["characters_seen"]:
                continuity_hints.append(f"maintaining consistent visual features of {char}")
        # Check locations
        setting_name = item.get("setting", "")
        if setting_name and setting_name.lower() in scene_memory["locations_seen"]:
            continuity_hints.append(f"maintaining setting continuity for {setting_name}")

    if continuity_hints:
        continuity_str = ", ".join(continuity_hints)
        environment = f"{environment}, {continuity_str}" if environment else continuity_str

    base_prompt = (
        f"{subject} {visual_description}, "
        f"in {loc_clean}. "
        f"{camera}; {motion_directive}. "
        f"{mood} mood, {lighting}, {color}. "
    )
    if environment:
        base_prompt += f"Environment: {environment}. "
    base_prompt += "Single continuous shot, no cuts, no time jump."

    prompt = wan_adapter(base_prompt, book_profile, style_bible)
    
    # Layer style overrides and clean duplicate prepositions
    prompt = apply_context_override(prompt)
    prompt = clean_duplicate_prepositions(prompt)
    
    return {
        "prompt": prompt.strip(),
        "negative_prompt": build_negative_prompt(book_profile),
        "duration_seconds": int(item.get("duration", book_profile.get("scene_seconds", MAX_SCENE_SECONDS))),
        "camera": camera,
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def build_final_video_prompts(raw_text: str, max_scenes: int | None = 3, scene_window: int = 3) -> list[ScenePrompt]:
    paragraphs = clean_book_text(raw_text)
    scenes = segment_scenes(paragraphs, window=scene_window)

    # Remove TOC, copyright, and other front-matter scenes
    scenes = [s for s in scenes if not _is_toc_or_front_matter(s)]

    # Keep ALL scenes for Bible generation (first 30 for maximum context),
    # but only render max_scenes worth of video.
    all_scenes_for_bible = scenes[:30]

    if max_scenes is not None:
        render_scenes = scenes[:max_scenes * 2]
    else:
        render_scenes = scenes

    book_profile = dict(BOOK_PROFILE)

    if mistral_available():
        inferred_profile = infer_book_profile_with_mistral(all_scenes_for_bible)
        if isinstance(inferred_profile, dict):
            book_profile.update({key: value for key, value in inferred_profile.items() if value not in (None, "")})

    heuristic_profile = heuristic_book_profile(all_scenes_for_bible)
    for key, value in heuristic_profile.items():
        if key not in book_profile or book_profile.get(key) in [None, "", "cinematic realism", "realistic", "natural tones", "general", False]:
            book_profile[key] = value

    book_profile["scene_seconds"] = min(int(book_profile.get("scene_seconds", MAX_SCENE_SECONDS)), MAX_SCENE_SECONDS)

    # Build dataset from ALL available scenes for richer Bible generation
    bible_dataset = build_dataset_from_scenes(all_scenes_for_bible)

    # Build dataset for rendering (may be a subset)
    render_dataset = build_dataset_from_scenes(render_scenes)
    if max_scenes is not None:
        render_dataset = render_dataset[:max_scenes]

    active_genre = detect_genre(bible_dataset, max_scenes=40)
    visual_style = apply_genre_preset(GLOBAL_VISUAL_STYLE, active_genre)

    # Build ALL 3 Bibles from wide context (first 30 scenes) — works for ANY book
    character_bible = build_character_bible(bible_dataset)
    location_bible = build_location_bible(bible_dataset)
    style_bible = build_visual_style_bible(all_scenes_for_bible, bible_dataset, active_genre)

    video_items = [
        scene_to_video_item(scene, scene["scene_id"], book_profile, character_bible, location_bible)
        for scene in render_dataset
    ]
    scene_text_map = {scene["scene_id"]: scene["scene_text"] for scene in render_dataset}

    # Initialize Scene Consistency Memory
    scene_memory = {
        "characters_seen": set(),
        "locations_seen": set(),
        "mood_history": []
    }

    final_video_prompts: list[ScenePrompt] = []
    for item in video_items:
        source_scene_id = str(item["scene_id"])
        output_scene_id = f"scene_{len(final_video_prompts)}"
        compiled = compile_video_prompt_scene(item, book_profile, visual_style, style_bible, scene_memory)
        
        # Update scene consistency memory
        for char in item.get("analysis", {}).get("characters", []):
            scene_memory["characters_seen"].add(char.lower())
        loc = item.get("setting", "")
        if loc:
            scene_memory["locations_seen"].add(loc.lower())
        scene_memory["mood_history"].append(item.get("mood", ""))

        final_video_prompts.append(
            ScenePrompt(
                scene_id=output_scene_id,
                shot_id=int(item.get("shot_id", 1)),
                scene_excerpt=scene_text_map[source_scene_id][:240],
                prompt=compiled["prompt"],
                negative_prompt=compiled["negative_prompt"],
                duration_seconds=compiled["duration_seconds"],
                camera=compiled["camera"],
            )
        )

    return final_video_prompts


def scene_prompt_to_dict(scene_prompt: ScenePrompt) -> dict:
    return asdict(scene_prompt)
