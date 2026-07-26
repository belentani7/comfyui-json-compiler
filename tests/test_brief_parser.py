"""Tests for the brief parser module."""

from __future__ import annotations

from compiler.brief_parser import (
    BriefRequirements,
    extract_camera,
    extract_lighting,
    extract_mood,
    extract_modifiers,
    extract_style,
    extract_subject,
    parse_brief,
)


class TestExtractSubject:
    """Tests for extract_subject."""

    def test_simple_subject(self) -> None:
        result = extract_subject("A dragon on a mountain")
        assert "dragon" in result.lower()

    def test_subject_with_action(self) -> None:
        result = extract_subject("A warrior standing on a cliff at sunset")
        assert "warrior" in result.lower()

    def test_minimal_subject(self) -> None:
        result = extract_subject("a cat sitting on a windowsill")
        assert "cat" in result.lower()

    def test_long_brief(self) -> None:
        result = extract_subject(
            "A beautiful sunset over the ocean with waves crashing on the shore"
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestExtractStyle:
    """Tests for extract_style."""

    def test_photorealistic(self) -> None:
        result = extract_style("A photorealistic portrait of a woman")
        assert "photorealistic" in result

    def test_anime(self) -> None:
        result = extract_style("An anime girl with pink hair")
        assert "anime" in result

    def test_oil_painting(self) -> None:
        result = extract_style("Oil painting of a landscape")
        assert "oil_painting" in result

    def test_cinematic(self) -> None:
        result = extract_style("Cinematic shot of a city at night")
        assert "cinematic" in result

    def test_multiple_styles(self) -> None:
        result = extract_style("Digital art anime cel-shaded illustration")
        assert len(result) >= 2
        assert "digital_art" in result
        assert "anime" in result

    def test_default_to_photorealistic(self) -> None:
        result = extract_style("A cat sitting on a chair")
        assert result == ["photorealistic"]

    def test_3d_render(self) -> None:
        result = extract_style("3D render of a mechanical robot")
        assert "3d_render" in result


class TestExtractCamera:
    """Tests for extract_camera."""

    def test_wide_angle(self) -> None:
        result = extract_camera("Wide angle shot of a cityscape")
        assert result["shot_type"] == "wide_angle"

    def test_close_up(self) -> None:
        result = extract_camera("Close-up of a flower petal")
        assert result["shot_type"] == "close_up"

    def test_portrait(self) -> None:
        result = extract_camera("Portrait of a young woman")
        assert result["shot_type"] == "portrait"

    def test_aerial(self) -> None:
        result = extract_camera("Aerial view of a forest")
        assert result["shot_type"] == "aerial"

    def test_low_angle(self) -> None:
        result = extract_camera("Low angle shot of a skyscraper")
        assert result["angle"] == "low_angle"

    def test_high_angle(self) -> None:
        result = extract_camera("High angle shot looking down")
        assert result["angle"] == "high_angle"

    def test_bokeh(self) -> None:
        result = extract_camera("Portrait with bokeh background")
        assert result.get("depth_of_field") == "shallow"

    def test_focal_length(self) -> None:
        result = extract_camera("85mm portrait lens shot")
        assert result.get("focal_length") == "85mm"

    def test_default_camera(self) -> None:
        result = extract_camera("A cat sitting")
        assert result["shot_type"] == "medium"
        assert result["angle"] == "eye_level"


class TestExtractLighting:
    """Tests for extract_lighting."""

    def test_natural_light(self) -> None:
        result = extract_lighting("Natural light portrait")
        assert result["type"] == "natural"

    def test_studio_lighting(self) -> None:
        result = extract_lighting("Studio lighting setup for product")
        assert result["type"] == "studio"

    def test_dramatic_lighting(self) -> None:
        result = extract_lighting("Dramatic lighting with strong shadows")
        assert result["type"] == "dramatic"

    def test_neon(self) -> None:
        result = extract_lighting("Neon lights reflecting in rain")
        assert result["type"] == "neon"

    def test_volumetric(self) -> None:
        result = extract_lighting("Volumetric light rays through fog")
        assert result["type"] == "volumetric"

    def test_direction(self) -> None:
        result = extract_lighting("Light from the left side")
        assert result.get("direction") == "left"

    def test_backlit(self) -> None:
        result = extract_lighting("Backlit silhouette at sunset")
        assert result["type"] == "rim"

    def test_default_lighting(self) -> None:
        result = extract_lighting("A cat on a table")
        assert result["type"] == "natural"


class TestExtractMood:
    """Tests for extract_mood."""

    def test_serene(self) -> None:
        assert extract_mood("A peaceful morning by the lake") == "serene"

    def test_dramatic(self) -> None:
        assert extract_mood("Dramatic storm over the mountains") == "dramatic"

    def test_mysterious(self) -> None:
        assert extract_mood("Mysterious figure in the fog") == "mysterious"

    def test_joyful(self) -> None:
        assert extract_mood("Joyful celebration with colorful confetti") == "joyful"

    def test_dark(self) -> None:
        assert extract_mood("Dark alley with noir atmosphere") == "dark"

    def test_epic(self) -> None:
        assert extract_mood("Epic battle between two armies") == "epic"

    def test_whimsical(self) -> None:
        assert extract_mood("Whimsical fairy tale scene") == "whimsical"

    def test_default_neutral(self) -> None:
        assert extract_mood("A cat sitting on a chair") == "neutral"


class TestExtractModifiers:
    """Tests for extract_modifiers."""

    def test_quality_terms(self) -> None:
        result = extract_modifiers("highly detailed 8k masterpiece")
        assert "highly detailed" in result
        assert "8k" in result

    def test_render_terms(self) -> None:
        result = extract_modifiers("octane render unreal engine")
        assert "octane render" in result
        assert "unreal engine" in result

    def test_empty_modifiers(self) -> None:
        result = extract_modifiers("a simple cat")
        assert result == []


class TestParseBrief:
    """Tests for the full parse_brief function."""

    def test_simple_brief(self, sample_brief_simple: str) -> None:
        result = parse_brief(sample_brief_simple)
        assert isinstance(result, BriefRequirements)
        assert "dragon" in result.subject.lower()
        assert "digital_art" in result.styles
        assert result.mood == "dramatic"

    def test_complex_brief(self, sample_brief_complex: str) -> None:
        result = parse_brief(sample_brief_complex)
        assert isinstance(result, BriefRequirements)
        assert result.camera["shot_type"] == "close_up"
        assert result.lighting["type"] == "natural"
        assert result.mood == "serene"

    def test_minimal_brief(self, sample_brief_minimal: str) -> None:
        result = parse_brief(sample_brief_minimal)
        assert isinstance(result, BriefRequirements)
        assert len(result.subject) > 0
        assert len(result.styles) > 0

    def test_returns_brief_requirements(self) -> None:
        result = parse_brief("anything")
        assert hasattr(result, "subject")
        assert hasattr(result, "styles")
        assert hasattr(result, "camera")
        assert hasattr(result, "lighting")
        assert hasattr(result, "mood")
        assert hasattr(result, "modifiers")
