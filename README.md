# comfyui-json-compiler

Translate natural language creative briefs into valid ComfyUI JSON workflows using LLMs.

## Overview

This tool converts human-readable image descriptions into executable [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows. It uses a hybrid approach:

1. **Rule-based extraction** ÔÇö fast, offline parsing of brief keywords (styles, camera, lighting, mood)
2. **LLM-enhanced planning** ÔÇö optional Claude integration for nuanced brief interpretation
3. **Visual QA** ÔÇö optional Qwen vision model for workflow quality assessment

## Installation

```bash
git clone https://github.com/belentani7/comfyui-json-compiler.git
cd comfyui-json-compiler
pip install -e ".[dev]"
```

### Environment Variables (optional, for LLM features)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # For Claude planning
export DASHSCOPE_API_KEY="sk-..."        # For Qwen visual QA
```

## Usage

### Python API

```python
from compiler import compile_brief

result = compile_brief(
    "A cinematic shot of a warrior on a cliff, "
    "dramatic lighting, low angle, epic mood, "
    "highly detailed, 8k"
)

# Complete workflow ready to POST to ComfyUI API
workflow = result["workflow"]

# Structured requirements extracted from brief
requirements = result["requirements"]

# Validation results
errors = result["validation_errors"]

# QA assessment (if Qwen available)
qa = result["qa_assessment"]
```

### Direct API

```python
from compiler import extract_subject, extract_style, extract_camera, build_node_graph, generate

# Parse a brief component by component
subject = extract_subject("A red sports car on a highway at sunset")
styles = extract_style("photorealistic cinematic shot")
camera = extract_camera("wide angle with bokeh background")

# Build and generate
from compiler.brief_parser import BriefRequirements
reqs = BriefRequirements(subject=subject, styles=styles, camera=camera)
graph = build_node_graph(reqs)
workflow = generate(reqs, graph)
```

### Load Pre-built Templates

```python
import json
from pathlib import Path

templates_dir = Path("compiler/templates")

# Available templates
templates = [
    "character_sheet.json",
    "environment.json",
    "product_photo.json",
    "cinematic_shot.json",
]

for template in templates:
    workflow = json.loads((templates_dir / template).read_text())
    # POST workflow to ComfyUI API...
```

### CLI Usage

```bash
# Generate workflow from brief
python -c "
from compiler import compile_brief
import json
result = compile_brief('A serene landscape at golden hour')
print(json.dumps(result['workflow'], indent=2))
"
```

### ComfyUI Integration

```python
import httpx
from compiler import compile_brief

def send_to_comfyui(brief: str, server: str = "http://127.0.0.1:8188"):
    result = compile_brief(brief)

    if result["validation_errors"]:
        print("Validation errors:", result["validation_errors"])
        return None

    response = httpx.post(
        f"{server}/prompt",
        json={"prompt": result["workflow"]}
    )
    return response.json()
```

## Architecture

```
compiler/
Ôö£ÔöÇÔöÇ __init__.py           # Public API exports
Ôö£ÔöÇÔöÇ brief_parser.py       # NL brief ÔåÆ structured requirements
Ôö£ÔöÇÔöÇ node_mapper.py        # Requirements ÔåÆ ComfyUI node graph
Ôö£ÔöÇÔöÇ workflow_generator.py # Node graph ÔåÆ valid ComfyUI JSON
Ôö£ÔöÇÔöÇ validator.py          # JSON syntax & connection validation
Ôö£ÔöÇÔöÇ llm_client.py         # LLM integration (Claude + Qwen)
ÔööÔöÇÔöÇ templates/            # Pre-built workflow templates
    Ôö£ÔöÇÔöÇ character_sheet.json
    Ôö£ÔöÇÔöÇ environment.json
    Ôö£ÔöÇÔöÇ product_photo.json
    ÔööÔöÇÔöÇ cinematic_shot.json
```

### Pipeline

```
Natural Language Brief
        Ôåô
   brief_parser.py      (rule-based extraction)
        Ôåô
   llm_client.py        (LLM enhancement, optional)
        Ôåô
   node_mapper.py       (map to ComfyUI nodes)
        Ôåô
   workflow_generator.py (emit valid JSON)
        Ôåô
   validator.py         (verify connections & types)
        Ôåô
   ComfyUI-ready workflow
```

## Supported Brief Features

| Feature | Keywords | Example |
|---------|----------|---------|
| **Styles** | photorealistic, anime, oil_painting, watercolor, cinematic, 3d_render, sketch, concept_art, pixel_art, comic_book | "oil painting of..." |
| **Camera** | wide_angle, close_up, portrait, full_body, aerial, low_angle, high_angle, dutch_angle, bokeh, fisheye | "close-up with bokeh" |
| **Lighting** | natural, studio, dramatic, soft, rim, neon, volumetric, candlelight | "dramatic rim lighting" |
| **Mood** | serene, dramatic, mysterious, joyful, melancholic, epic, dark, ethereal, whimsical | "mysterious atmosphere" |
| **Quality** | highly detailed, 8k, masterpiece, sharp focus, octane render | "8k, highly detailed" |
| **Focal Length** | Nmm patterns | "85mm portrait" |

## Testing

```bash
pytest
pytest --cov=compiler
```

## License

Apache License 2.0 ÔÇö see [LICENSE](LICENSE).
