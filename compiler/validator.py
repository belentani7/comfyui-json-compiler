"""Validate ComfyUI JSON workflows for correctness."""

from __future__ import annotations

from typing import Any

# Known ComfyUI node types (not exhaustive, covers core + popular extensions)
KNOWN_NODE_TYPES: set[str] = {
    # Core nodes
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "KSampler",
    "KSamplerAdvanced",
    "VAEDecode",
    "VAEEncode",
    "VAEDecodeTiled",
    "EmptyLatentImage",
    "EmptyLatentBatch",
    "SaveImage",
    "PreviewImage",
    "LoadImage",
    "ImageScale",
    "ImageCrop",
    "LatentUpscale",
    "LatentComposite",
    "ConditioningCombine",
    "ConditioningSetArea",
    "ConditioningSetTimestepRange",
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "ControlNetLoader",
    # Preprocessors
    "DepthAnythingPreprocessor",
    "CannyEdgePreprocessor",
    "LineArtPreprocessor",
    "NormalMapPreprocessor",
    "MiDaS-DepthMapPreprocessor",
    "MiDaS-NormalMapPreprocessor",
    "SAMSegment",
    # Style transfer
    "IPAdapterAdvanced",
    "IPAdapter",
    "IPAdapterModelLoader",
    "CLIPVisionLoader",
    "FaceLock",
    "InstantID",
    "ReferenceOnly",
    # Camera
    "CameraParameters",
    # Upscale
    "UpscaleModelLoader",
    "ImageUpscaleWithModel",
    "RealESRGANModelLoader",
    # Misc
    "Noise",
    "RandomNoise",
    "BasicGuider",
    "BasicScheduler",
    "KSamplerSelect",
    "SamplerCustom",
    "VAEApply",
    "RepeatLatentBatch",
    "LatentBatch",
    "ImageBatch",
    "JoinImageWithAlpha",
    "SplitImageWithAlpha",
    "ImageInvert",
    "ImageBlend",
    "ImagePadForOutpaint",
    "SetLatentNoiseMask",
    "CropImage",
    "ResizeImage",
}

# Required input parameters per node type
REQUIRED_INPUTS: dict[str, set[str]] = {
    "CheckpointLoaderSimple": {"ckpt_name"},
    "CLIPTextEncode": {"text", "clip"},
    "KSampler": {"model", "positive", "negative", "latent_image"},
    "VAEDecode": {"samples", "vae"},
    "VAEEncode": {"pixels", "vae"},
    "EmptyLatentImage": {"width", "height"},
    "SaveImage": {"images"},
    "LoadImage": {"image"},
    "DepthAnythingPreprocessor": {"image"},
    "CannyEdgePreprocessor": {"image"},
    "LineArtPreprocessor": {"image"},
    "NormalMapPreprocessor": {"image"},
    "IPAdapterAdvanced": {"model", "ipadapter", "image"},
    "FaceLock": {"model", "image"},
    "InstantID": {"model", "image", "ipadapter", "clip_vision"},
    "ReferenceOnly": {"model", "image"},
    "CameraParameters": {"fov", "distance"},
    "UpscaleModelLoader": {"model_name"},
    "ImageUpscaleWithModel": {"upscale_model", "image"},
    "ControlNetApply": {"conditioning", "control_net", "image"},
    "ControlNetApplyAdvanced": {"positive", "negative", "control_net", "image"},
    "ConditioningCombine": {"conditioning_1", "conditioning_2"},
}

# Expected parameter types
PARAMETER_TYPES: dict[str, dict[str, type | tuple[type, ...]]] = {
    "KSampler": {
        "seed": (int,),
        "steps": (int,),
        "cfg": (int, float),
        "denoise": (int, float),
        "sampler_name": (str,),
        "scheduler": (str,),
    },
    "EmptyLatentImage": {
        "width": (int,),
        "height": (int,),
        "batch_size": (int,),
    },
    "DepthAnythingPreprocessor": {
        "resolution": (int,),
    },
    "CannyEdgePreprocessor": {
        "low_threshold": (int,),
        "high_threshold": (int,),
    },
    "CameraParameters": {
        "fov": (int, float),
        "distance": (int, float),
        "tilt": (int, float),
        "roll": (int, float),
        "pan": (int, float),
    },
    "ImageScale": {
        "upscale_method": (str,),
        "width": (int,),
        "height": (int,),
        "crop": (str,),
    },
}


def _is_wire_link(value: Any) -> bool:
    """Check if a value is a ComfyUI wire link [node_id, slot]."""
    return isinstance(value, list) and len(value) == 2


def validate_node_types(workflow: dict[str, Any]) -> list[str]:
    """Validate that all node types are recognized by ComfyUI.

    Returns a list of error messages for unrecognized node types.
    """
    errors: list[str] = []

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        if not class_type:
            errors.append(f"Node {node_id}: missing class_type")
        elif class_type not in KNOWN_NODE_TYPES:
            errors.append(f"Node {node_id}: unknown node type '{class_type}'")

    return errors


def validate_connections(workflow: dict[str, Any]) -> list[str]:
    """Validate that all wire connections reference existing nodes and slots.

    Returns a list of error messages for invalid connections.
    """
    errors: list[str] = []
    all_node_ids = set(workflow.keys())

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict) or "inputs" not in node_data:
            continue

        for param_name, param_value in node_data.get("inputs", {}).items():
            if not _is_wire_link(param_value):
                continue

            source_id = str(param_value[0])
            source_slot = param_value[1]

            if source_id not in all_node_ids:
                errors.append(
                    f"Node {node_id}, param '{param_name}': "
                    f"references non-existent node '{source_id}'"
                )
            elif not isinstance(source_slot, int) or source_slot < 0:
                errors.append(
                    f"Node {node_id}, param '{param_name}': "
                    f"invalid slot index {source_slot}"
                )

    return errors


def validate_parameters(workflow: dict[str, Any]) -> list[str]:
    """Validate parameter types and required inputs for each node.

    Returns a list of error messages for parameter issues.
    """
    errors: list[str] = []

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        # Check required inputs
        required = REQUIRED_INPUTS.get(class_type, set())
        for param in required:
            if param not in inputs:
                errors.append(f"Node {node_id} ({class_type}): missing required param '{param}'")

        # Check parameter types
        type_specs = PARAMETER_TYPES.get(class_type, {})
        for param_name, expected_types in type_specs.items():
            if param_name not in inputs:
                continue
            value = inputs[param_name]
            # Skip wire links
            if _is_wire_link(value):
                continue
            if not isinstance(value, expected_types):
                errors.append(
                    f"Node {node_id} ({class_type}), param '{param_name}': "
                    f"expected {expected_types}, got {type(value).__name__}"
                )

    return errors


def validate_all(workflow: dict[str, Any]) -> list[str]:
    """Run all validation checks and return combined errors."""
    errors: list[str] = []
    if not workflow:
        errors.append("Workflow is empty")
        return errors
    errors.extend(validate_node_types(workflow))
    errors.extend(validate_connections(workflow))
    errors.extend(validate_parameters(workflow))
    return errors
