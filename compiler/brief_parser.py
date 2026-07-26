"""Parse natural language briefs into structured requirements for ComfyUI workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class BriefRequirements:
    """Structured representation of a parsed creative brief."""

    subject: str
    styles: list[str] = field(default_factory=list)
    camera: dict[str, str] = field(default_factory=dict)
    lighting: dict[str, str] = field(default_factory=dict)
    mood: str = ""
    modifiers: list[str] = field(default_factory=list)


# Keyword banks for extraction
_STYLE_KEYWORDS: dict[str, list[str]] = {
    "photorealistic": ["photorealistic", "photo-realistic", "realistic", "photograph", "photo"],
    "anime": ["anime", "manga", "cel-shaded", "cel shaded"],
    "oil_painting": ["oil painting", "oil-painted", "painterly"],
    "watercolor": ["watercolor", "water colour", "aquarelle"],
    "digital_art": ["digital art", "digital painting", "illustration"],
    "3d_render": ["3d render", "3d rendering", "cg", "cgi"],
    "sketch": ["sketch", "pencil drawing", "line drawing"],
    "cinematic": ["cinematic", "film still", "movie scene", "anamorphic"],
    "flat_design": ["flat design", "minimalist design", "vector art"],
    "concept_art": ["concept art", "matte painting", "environment art"],
    "pixel_art": ["pixel art", "8-bit", "retro"],
    "comic_book": ["comic book", "graphic novel", "sequential art"],
}

_CAMERA_KEYWORDS: dict[str, list[str]] = {
    # Explicit shot types first (highest priority)
    "wide_angle": ["wide angle", "wide-angle", "wide shot", "establishing shot"],
    "close_up": ["close-up", "closeup", "close up", "macro"],
    "full_body": ["full body", "full-body", "full length"],
    "aerial": ["aerial", "bird's eye", "birds eye", "overhead", "top-down"],
    "panoramic": ["panoramic", "wide panoramic"],
    "fisheye": ["fisheye", "fisheye lens", "fish eye"],
    # Angles
    "low_angle": ["low angle", "low-angle", "worm's eye"],
    "high_angle": ["high angle", "high-angle"],
    "dutch_angle": ["dutch angle", "dutch-angle", "tilted"],
    "over_the_shoulder": ["over the shoulder", "over-the-shoulder"],
    # Effects (applied after shot type)
    "bokeh": ["bokeh", "shallow depth of field", "blurred background"],
    # Broader terms (lowest priority)
    "portrait": ["portrait", "headshot", "bust"],
}

_LIGHTING_KEYWORDS: dict[str, list[str]] = {
    "natural": ["natural light", "daylight", "sunlight", "golden hour"],
    "studio": ["studio lighting", "studio light", "controlled lighting"],
    "dramatic": ["dramatic lighting", "dramatic light", "high contrast lighting"],
    "soft": ["soft light", "soft lighting", "diffused light", "diffused"],
    "rim": ["rim light", "rim lighting", "backlit", "backlight"],
    "neon": ["neon", "neon light", "neon glow", "cyberpunk lighting"],
    "candlelight": ["candlelight", "candle light", "warm glow"],
    "moonlight": ["moonlight", "moon light", "nightlight"],
    "volumetric": ["volumetric", "volumetric light", "god rays", "crepuscular"],
    "harsh": ["harsh light", "hard light", "hard lighting"],
    "underwater": ["underwater light", "underwater lighting"],
    "ethereal": ["ethereal light", "ethereal lighting", "magical light"],
}

_MOOD_KEYWORDS: dict[str, list[str]] = {
    "serene": ["serene", "peaceful", "calm", "tranquil", "zen"],
    "dramatic": ["dramatic", "intense", "powerful"],
    "mysterious": ["mysterious", "enigmatic", "eerie", "haunting"],
    "joyful": ["joyful", "happy", "cheerful", "vibrant", "lively"],
    "melancholic": ["melancholic", "somber", "moody", "nostalgic", "wistful"],
    "epic": ["epic", "grand", "majestic", "sweeping"],
    "intimate": ["intimate", "personal", "close", "tender"],
    "dark": ["dark", "gloomy", "shadowy", "noir", "grim"],
    "ethereal": ["ethereal", "dreamy", "otherworldly", "celestial"],
    "gritty": ["gritty", "raw", "urban", "street"],
    "whimsical": ["whimsical", "playful", "fantastical", "fairytale"],
    "tense": ["tense", "suspenseful", "anxious", "uneasy"],
}


def extract_subject(brief: str) -> str:
    """Extract the main subject from a creative brief.

    Identifies the primary subject using common subject patterns and
    returns a cleaned-up subject string.
    """
    brief_lower = brief.lower().strip()

    # Pattern: "a/an [subject] [doing something]"
    match = re.search(r"(?:a|an)\s+(.+?)(?:\s+(?:in|with|on|at|wearing|standing|sitting|looking|posed?|shot))", brief_lower)
    if match:
        return match.group(1).strip()

    # Pattern: "[Subject] [verb phrase]"
    match = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", brief)
    if match:
        return match.group(1).strip()

    # Fallback: take first N words as subject
    words = brief_lower.split()
    if len(words) <= 6:
        return brief_lower
    return " ".join(words[:6])


def extract_style(brief: str) -> list[str]:
    """Extract visual style keywords from a creative brief.

    Returns a list of matched style identifiers from the known style bank.
    """
    brief_lower = brief.lower()
    matched: list[str] = []

    for style_key, keywords in _STYLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in brief_lower:
                matched.append(style_key)
                break

    return matched if matched else ["photorealistic"]


def extract_camera(brief: str) -> dict[str, str]:
    """Extract camera/composition keywords from a creative brief.

    Returns a dict with 'shot_type' and 'angle' keys at minimum.
    """
    brief_lower = brief.lower()
    result: dict[str, str] = {"shot_type": "medium", "angle": "eye_level"}

    for cam_key, keywords in _CAMERA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in brief_lower:
                if cam_key in ("wide_angle", "close_up", "portrait", "full_body", "aerial", "panoramic"):
                    # First explicit shot type wins; don't let broader terms overwrite
                    if result["shot_type"] == "medium":
                        result["shot_type"] = cam_key
                elif cam_key in ("low_angle", "high_angle", "dutch_angle"):
                    result["angle"] = cam_key
                elif cam_key == "bokeh":
                    result["depth_of_field"] = "shallow"
                elif cam_key == "fisheye":
                    result["lens"] = "fisheye"
                break

    # Look for focal length patterns
    focal_match = re.search(r"(\d+)mm", brief_lower)
    if focal_match:
        result["focal_length"] = focal_match.group(1) + "mm"

    return result


def extract_lighting(brief: str) -> dict[str, str]:
    """Extract lighting keywords from a creative brief.

    Returns a dict describing the lighting setup.
    """
    brief_lower = brief.lower()
    result: dict[str, str] = {"type": "natural"}

    for light_key, keywords in _LIGHTING_KEYWORDS.items():
        for keyword in keywords:
            if keyword in brief_lower:
                result["type"] = light_key
                break

    # Directional cues
    if "left" in brief_lower and ("light" in brief_lower or "lit" in brief_lower):
        result["direction"] = "left"
    elif "right" in brief_lower and ("light" in brief_lower or "lit" in brief_lower):
        result["direction"] = "right"
    elif "behind" in brief_lower or "back" in brief_lower:
        result["direction"] = "behind"
    elif "above" in brief_lower or "overhead" in brief_lower:
        result["direction"] = "above"

    return result


def extract_mood(brief: str) -> str:
    """Extract the overall mood/atmosphere from a creative brief.

    Returns the best-matching mood keyword.
    """
    brief_lower = brief.lower()

    for mood_key, keywords in _MOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in brief_lower:
                return mood_key

    return "neutral"


def extract_modifiers(brief: str) -> list[str]:
    """Extract additional quality/style modifiers from the brief."""
    brief_lower = brief.lower()
    modifiers: list[str] = []

    quality_terms = [
        "high quality", "highly detailed", "ultra detailed", "8k", "4k",
        "masterpiece", "best quality", "professional", "award winning",
        "trending on artstation", "octane render", "unreal engine",
        "ray tracing", "hdr", "sharp focus", "intricate",
    ]

    for term in quality_terms:
        if term in brief_lower:
            modifiers.append(term)

    return modifiers


def parse_brief(brief: str) -> BriefRequirements:
    """Parse a full creative brief into structured requirements.

    This is the main entry point that orchestrates all extraction functions.
    """
    return BriefRequirements(
        subject=extract_subject(brief),
        styles=extract_style(brief),
        camera=extract_camera(brief),
        lighting=extract_lighting(brief),
        mood=extract_mood(brief),
        modifiers=extract_modifiers(brief),
    )
