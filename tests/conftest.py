"""Shared fixtures for comfyui-json-compiler tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_brief_simple() -> str:
    """A simple creative brief for testing."""
    return "A majestic dragon perched on a mountain peak at sunset, digital art, dramatic lighting"


@pytest.fixture
def sample_brief_complex() -> str:
    """A complex creative brief with many parameters."""
    return (
        "A photorealistic portrait of a young woman with silver hair, "
        "close-up shot, shallow depth of field, bokeh background, "
        "soft natural light from the left, serene mood, "
        "highly detailed, 8k, sharp focus, "
        "wearing a dark velvet dress, looking at the camera"
    )


@pytest.fixture
def sample_brief_minimal() -> str:
    """A minimal creative brief."""
    return "a cat sitting on a windowsill"


@pytest.fixture
def sample_workflow() -> dict:
    """A minimal valid ComfyUI workflow for testing."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            "_meta": {"title": "Load Checkpoint"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "test prompt", "clip": ["1", 1]},
            "_meta": {"title": "Positive Prompt"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "bad quality", "clip": ["1", 1]},
            "_meta": {"title": "Negative Prompt"},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "_meta": {"title": "Empty Latent"},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": 42,
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
            "_meta": {"title": "Sampler"},
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            "_meta": {"title": "VAE Decode"},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "test"},
            "_meta": {"title": "Save Image"},
        },
    }


@pytest.fixture
def sample_workflow_invalid() -> dict:
    """A workflow with intentional errors for validation testing."""
    return {
        "1": {
            "class_type": "UnknownNodeType",
            "inputs": {"param": "value"},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["99", 0],  # References non-existent node
                "positive": ["1", 0],
                "negative": ["1", 0],
                "latent_image": ["1", 0],
                "seed": "not_an_int",  # Wrong type
            },
        },
    }


@pytest.fixture
def sample_requirements():
    """Pre-built BriefRequirements for testing node mapper."""
    from compiler.brief_parser import BriefRequirements

    return BriefRequirements(
        subject="a warrior standing on a cliff",
        styles=["cinematic", "concept_art"],
        camera={"shot_type": "wide_angle", "angle": "low_angle"},
        lighting={"type": "dramatic", "direction": "behind"},
        mood="epic",
        modifiers=["highly detailed", "8k"],
    )


@pytest.fixture
def templates_dir() -> Path:
    """Path to the templates directory."""
    return PROJECT_ROOT / "compiler" / "templates"
