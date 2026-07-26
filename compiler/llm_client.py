"""LLM integration for compiling natural language briefs into ComfyUI workflows."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from compiler.brief_parser import BriefRequirements, parse_brief
from compiler.node_mapper import build_node_graph
from compiler.workflow_generator import generate, optimize_workflow, validate_workflow

# Provider configurations
_PROVIDERS: dict[str, dict[str, str]] = {
    "claude": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "default_model": "qwen-vl-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
}

_PLANNING_SYSTEM_PROMPT = """You are a ComfyUI workflow planner. Given a creative brief, produce a JSON object
with these keys:
- subject: main subject description (string)
- styles: visual styles (array of strings)
- camera: camera settings (object with shot_type, angle, focal_length, depth_of_field)
- lighting: lighting setup (object with type, direction)
- mood: overall mood (string)
- modifiers: quality/style modifiers (array of strings)
- negative_prompt: things to avoid (string)
- resolution: output resolution (object with width, height)
- checkpoint: preferred checkpoint model name (string)
- steps: sampling steps (integer)
- cfg: classifier-free guidance scale (float)

Be specific and creative. Extract ALL relevant details from the brief.
Output ONLY valid JSON, no markdown fences."""

_VISUAL_QA_PROMPT = """You are a visual quality assessor for ComfyUI workflows.
Given a workflow JSON and a creative brief, evaluate:
1. Does the workflow match the brief's intent?
2. Are the parameters reasonable for the desired output?
3. Are there missing nodes or connections that would improve the result?

Output a JSON object with:
- score: 1-10 quality score
- suggestions: array of improvement suggestions
- issues: array of problems found
Output ONLY valid JSON, no markdown fences."""


def _get_api_key(provider: str) -> str:
    """Retrieve API key from environment variables."""
    provider_config = _PROVIDERS.get(provider, _PROVIDERS["claude"])
    api_key = os.environ.get(provider_config["api_key_env"], "")
    if not api_key:
        raise ValueError(
            f"API key not found. Set {provider_config['api_key_env']} environment variable."
        )
    return api_key


def _call_claude(
    prompt: str,
    system: str = "",
    model: str | None = None,
) -> str:
    """Call the Claude API and return the response text."""
    api_key = _get_api_key("claude")
    config = _PROVIDERS["claude"]
    model = model or config["default_model"]

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    with httpx.Client(timeout=60.0) as client:
        response = client.post(config["base_url"], headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return data["content"][0]["text"]


def _call_openai_compatible(
    provider: str,
    prompt: str,
    system: str = "",
    model: str | None = None,
) -> str:
    """Call an OpenAI-compatible API (Qwen, OpenAI, etc.)."""
    api_key = _get_api_key(provider)
    config = _PROVIDERS[provider]
    model = model or config["default_model"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(config["base_url"], headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"]


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def _enhance_with_llm(brief: str, model: str) -> dict[str, Any]:
    """Use Claude to extract structured requirements from a brief."""
    response = _call_claude(brief, _PLANNING_SYSTEM_PROMPT, model)
    return _parse_json_response(response)


def _visual_qa(workflow: dict[str, Any], brief: str) -> dict[str, Any]:
    """Use Qwen (vision-capable) to evaluate the generated workflow."""
    prompt = f"Creative brief: {brief}\n\nWorkflow:\n{json.dumps(workflow, indent=2)}\n\n{_VISUAL_QA_PROMPT}"
    try:
        response = _call_openai_compatible("qwen", prompt)
        return _parse_json_response(response)
    except (ValueError, httpx.HTTPError):
        # Qwen not available, return a default assessment
        return {"score": 7, "suggestions": [], "issues": []}


def compile_brief(brief: str, model: str = "claude") -> dict[str, Any]:
    """Compile a natural language brief into a complete ComfyUI workflow.

    Two-phase process:
    1. Use Claude for structured planning and node graph construction
    2. Use Qwen (if available) for visual quality assessment

    Args:
        brief: Natural language creative brief.
        model: LLM model identifier for planning (default: "claude").

    Returns:
        Dict with keys: workflow, requirements, validation_errors, qa_assessment.
    """
    # Phase 1: Parse with local rule-based extraction
    local_reqs = parse_brief(brief)

    # Phase 2: Enhance with LLM (if API key available)
    llm_enhanced: dict[str, Any] | None = None
    try:
        llm_enhanced = _enhance_with_llm(brief, model)
    except (ValueError, httpx.HTTPError):
        pass

    # Merge LLM results into requirements if available
    if llm_enhanced:
        subject = llm_enhanced.get("subject", local_reqs.subject)
        styles = llm_enhanced.get("styles", local_reqs.styles)
        camera = {**local_reqs.camera, **llm_enhanced.get("camera", {})}
        lighting = {**local_reqs.lighting, **llm_enhanced.get("lighting", {})}
        mood = llm_enhanced.get("mood", local_reqs.mood)
        modifiers = llm_enhanced.get("modifiers", local_reqs.modifiers)

        requirements = BriefRequirements(
            subject=subject,
            styles=styles if isinstance(styles, list) else [styles],
            camera=camera if isinstance(camera, dict) else {},
            lighting=lighting if isinstance(lighting, dict) else {},
            mood=mood if isinstance(mood, str) else str(mood),
            modifiers=modifiers if isinstance(modifiers, list) else [],
        )
    else:
        requirements = local_reqs

    # Phase 3: Build node graph and generate workflow
    node_graph = build_node_graph(requirements)
    workflow = generate(requirements, node_graph)

    # Phase 4: Validate
    validation_errors = validate_workflow(workflow)

    # Phase 5: Optimize
    optimized = optimize_workflow(workflow)

    # Phase 6: Visual QA (if Qwen available)
    qa_assessment: dict[str, Any] = {"score": 7, "suggestions": [], "issues": []}
    try:
        qa_assessment = _visual_qa(optimized, brief)
    except (ValueError, httpx.HTTPError):
        pass

    return {
        "workflow": optimized,
        "requirements": {
            "subject": requirements.subject,
            "styles": requirements.styles,
            "camera": requirements.camera,
            "lighting": requirements.lighting,
            "mood": requirements.mood,
            "modifiers": requirements.modifiers,
        },
        "validation_errors": validation_errors,
        "qa_assessment": qa_assessment,
        "llm_used": llm_enhanced is not None,
    }
