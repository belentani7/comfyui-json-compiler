"""Map structured requirements to ComfyUI node configurations."""

from __future__ import annotations

from dataclasses import dataclass, field

from compiler.brief_parser import BriefRequirements


# Node type constants
class NodeTypes:
    """Canonical ComfyUI node type identifiers."""

    CHECKPOINT_LOADER = "CheckpointLoaderSimple"
    CLIP_TEXT_ENCODE = "CLIPTextEncode"
    KSampler = "KSampler"
    VAE_DECODE = "VAEDecode"
    EMPTY_LATENT = "EmptyLatentImage"
    SAVE_IMAGE = "SaveImage"
    PREVIEW_IMAGE = "PreviewImage"

    # ControlNet / preprocessors
    DEPTH_ANYTHING = "DepthAnythingPreprocessor"
    CANNY_EDGE = "CannyEdgePreprocessor"
    LINEART = "LineArtPreprocessor"
    NORMAL_MAP = "NormalMapPreprocessor"
    DEPTHESTIMATION = "DepthEstimation"

    # Style transfer
    IP_ADAPTER = "IPAdapterAdvanced"
    IP_ADAPTER_MODEL_LOADER = "IPAdapterModelLoader"
    FACE_LOCK = "FaceLock"
    INSTANT_ID = "InstantID"
    REFERENCE_ONLY = "ReferenceOnly"

    # Camera
    CAMERA_PARAMETERS = "CameraParameters"

    # Upscale
    UPSCALE_MODEL = "UpscaleModelLoader"
    IMAGE_UPSCALE = "ImageUpscaleWithModel"
    LATENT_UPSCALE = "LatentUpscale"

    # Image
    LOAD_IMAGE = "LoadImage"

    # Conditioning
    CONDITIONING_CONCAT = "ConditioningCombine"
    CONDITIONING_SET_AREA = "ConditioningSetArea"


# Mapping from preprocessor keyword to node type
PREPROCESSOR_MAP: dict[str, str] = {
    "depth": NodeTypes.DEPTH_ANYTHING,
    "canny": NodeTypes.CANNY_EDGE,
    "lineart": NodeTypes.LINEART,
    "line_art": NodeTypes.LINEART,
    "normal": NodeTypes.NORMAL_MAP,
    "normal_map": NodeTypes.NORMAL_MAP,
}

# Mapping from style keyword to node type
STYLE_NODE_MAP: dict[str, str] = {
    "ip_adapter": NodeTypes.IP_ADAPTER,
    "face": NodeTypes.FACE_LOCK,
    "face_lock": NodeTypes.FACE_LOCK,
    "instantid": NodeTypes.INSTANT_ID,
    "reference": NodeTypes.REFERENCE_ONLY,
}

# Mapping from camera shot type to default parameters
CAMERA_DEFAULTS: dict[str, dict[str, float]] = {
    "wide_angle": {"fov": 90.0, "distance": 5.0},
    "close_up": {"fov": 45.0, "distance": 1.5},
    "portrait": {"fov": 50.0, "distance": 2.0},
    "full_body": {"fov": 60.0, "distance": 3.5},
    "aerial": {"fov": 120.0, "distance": 15.0},
    "low_angle": {"fov": 60.0, "distance": 3.0, "tilt": -30.0},
    "high_angle": {"fov": 60.0, "distance": 3.0, "tilt": 30.0},
    "dutch_angle": {"fov": 60.0, "distance": 3.0, "roll": 15.0},
    "fisheye": {"fov": 170.0, "distance": 2.0},
    "panoramic": {"fov": 140.0, "distance": 10.0},
}

# Mood to sampler settings
MOOD_SAMPLER_MAP: dict[str, dict[str, float | str]] = {
    "serene": {"cfg": 7.0, "sampler_name": "euler", "scheduler": "normal"},
    "dramatic": {"cfg": 9.0, "sampler_name": "dpmpp_2m", "scheduler": "karras"},
    "mysterious": {"cfg": 8.5, "sampler_name": "dpmpp_sde", "scheduler": "karras"},
    "joyful": {"cfg": 7.5, "sampler_name": "euler", "scheduler": "normal"},
    "melancholic": {"cfg": 8.0, "sampler_name": "dpmpp_2m", "scheduler": "karras"},
    "epic": {"cfg": 8.5, "sampler_name": "dpmpp_2m_sde", "scheduler": "karras"},
    "dark": {"cfg": 9.0, "sampler_name": "dpmpp_sde", "scheduler": "karras"},
    "ethereal": {"cfg": 7.0, "sampler_name": "euler_ancestral", "scheduler": "normal"},
    "gritty": {"cfg": 8.5, "sampler_name": "dpmpp_2m", "scheduler": "karras"},
    "whimsical": {"cfg": 7.5, "sampler_name": "euler_ancestral", "scheduler": "normal"},
    "tense": {"cfg": 9.5, "sampler_name": "dpmpp_sde", "scheduler": "karras"},
    "intimate": {"cfg": 7.0, "sampler_name": "euler", "scheduler": "normal"},
    "neutral": {"cfg": 7.5, "sampler_name": "euler", "scheduler": "normal"},
}


@dataclass
class NodeSpec:
    """Specification for a single ComfyUI node."""

    node_id: str
    node_type: str
    inputs: dict[str, str | int | float | list[str]] = field(default_factory=dict)
    title: str = ""
    color: str = ""
    bgcolor: str = ""


@dataclass
class NodeGraph:
    """Complete graph of nodes with connections."""

    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    connections: list[tuple[str, str, str, str]] = field(default_factory=list)

    def add_node(self, spec: NodeSpec) -> None:
        self.nodes[spec.node_id] = spec

    def connect(self, from_node: str, from_slot: str, to_node: str, to_slot: str) -> None:
        self.connections.append((from_node, from_slot, to_node, to_slot))


def _resolve_style_nodes(styles: list[str]) -> list[str]:
    """Resolve style keywords to node type names."""
    nodes: list[str] = []
    style_lower = [s.lower().replace(" ", "_") for s in styles]

    for style in style_lower:
        if style in STYLE_NODE_MAP:
            nodes.append(STYLE_NODE_MAP[style])

    # Default: if photorealistic or no style match, no extra style node needed
    return nodes


def _build_controlnet_nodes(reqs: BriefRequirements) -> list[NodeSpec]:
    """Build ControlNet preprocessor nodes based on brief requirements."""
    nodes: list[NodeSpec] = []
    node_counter = 0

    # Check camera info for depth-of-field hints → depth preprocessor
    if reqs.camera.get("depth_of_field") == "shallow":
        node_id = f"preprocessor_{node_counter}"
        nodes.append(NodeSpec(
            node_id=node_id,
            node_type=NodeTypes.DEPTH_ANYTHING,
            inputs={"resolution": 512},
            title="Depth Estimation",
        ))
        node_counter += 1

    # Check mood-based preprocessor needs
    mood = reqs.mood.lower()
    if mood in ("dramatic", "epic", "tense"):
        node_id = f"preprocessor_{node_counter}"
        nodes.append(NodeSpec(
            node_id=node_id,
            node_type=NodeTypes.CANNY_EDGE,
            inputs={"low_threshold": 100, "high_threshold": 200},
            title="Edge Detection",
        ))

    return nodes


def _build_camera_node(reqs: BriefRequirements) -> NodeSpec | None:
    """Build a CameraParameters node from camera requirements."""
    shot_type = reqs.camera.get("shot_type", "medium")
    defaults = CAMERA_DEFAULTS.get(shot_type, CAMERA_DEFAULTS["portrait"])

    angle = reqs.camera.get("angle", "eye_level")
    if angle == "low_angle":
        defaults = {**defaults, "tilt": defaults.get("tilt", 0) - 15.0}
    elif angle == "high_angle":
        defaults = {**defaults, "tilt": defaults.get("tilt", 0) + 15.0}

    return NodeSpec(
        node_id="camera_params",
        node_type=NodeTypes.CAMERA_PARAMETERS,
        inputs={
            "fov": defaults.get("fov", 60.0),
            "distance": defaults.get("distance", 3.0),
            "tilt": defaults.get("tilt", 0.0),
            "roll": defaults.get("roll", 0.0),
            "pan": defaults.get("pan", 0.0),
        },
        title=f"Camera: {shot_type.replace('_', ' ').title()}",
    )


def _get_sampler_settings(mood: str) -> dict[str, float | str]:
    """Get sampler configuration for a given mood."""
    return MOOD_SAMPLER_MAP.get(mood.lower(), MOOD_SAMPLER_MAP["neutral"])


def _build_positive_prompt(reqs: BriefRequirements) -> str:
    """Construct the positive prompt string from requirements."""
    parts: list[str] = []

    if reqs.subject:
        parts.append(reqs.subject)

    if reqs.styles:
        style_display = ", ".join(s.replace("_", " ") for s in reqs.styles)
        parts.append(style_display)

    if reqs.mood and reqs.mood != "neutral":
        parts.append(reqs.mood)

    if reqs.modifiers:
        parts.append(", ".join(reqs.modifiers))

    return ", ".join(parts)


def _build_negative_prompt() -> str:
    """Standard negative prompt for quality."""
    return (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
        "deformed, ugly, duplicate, morbid, mutilated"
    )


def build_node_graph(reqs: BriefRequirements) -> NodeGraph:
    """Build a complete ComfyUI node graph from parsed requirements.

    Orchestrates node creation for the core pipeline:
    checkpoint loader → CLIP encode → sampler → VAE decode → save image

    Plus optional nodes for ControlNet, style transfer, and camera.
    """
    graph = NodeGraph()
    sampler_settings = _get_sampler_settings(reqs.mood)

    # 1. Checkpoint Loader
    graph.add_node(NodeSpec(
        node_id="checkpoint_loader",
        node_type=NodeTypes.CHECKPOINT_LOADER,
        inputs={"ckpt_name": "sd_xl_base_1.0.safetensors"},
        title="Load Checkpoint",
    ))

    # 2. Positive CLIP Encode
    positive_prompt = _build_positive_prompt(reqs)
    graph.add_node(NodeSpec(
        node_id="positive_clip",
        node_type=NodeTypes.CLIP_TEXT_ENCODE,
        inputs={"text": positive_prompt, "clip": ["checkpoint_loader", 1]},
        title="Positive Prompt",
    ))

    # 3. Negative CLIP Encode
    negative_prompt = _build_negative_prompt()
    graph.add_node(NodeSpec(
        node_id="negative_clip",
        node_type=NodeTypes.CLIP_TEXT_ENCODE,
        inputs={"text": negative_prompt, "clip": ["checkpoint_loader", 1]},
        title="Negative Prompt",
    ))

    # 4. Empty Latent Image
    graph.add_node(NodeSpec(
        node_id="empty_latent",
        node_type=NodeTypes.EMPTY_LATENT,
        inputs={"width": 1024, "height": 1024, "batch_size": 1},
        title="Empty Latent",
    ))

    # 5. KSampler
    graph.add_node(NodeSpec(
        node_id="ksampler",
        node_type=NodeTypes.KSampler,
        inputs={
            "model": ["checkpoint_loader", 0],
            "positive": ["positive_clip", 0],
            "negative": ["negative_clip", 0],
            "latent_image": ["empty_latent", 0],
            "seed": -1,
            "steps": 30,
            "cfg": sampler_settings.get("cfg", 7.5),
            "sampler_name": sampler_settings.get("sampler_name", "euler"),
            "scheduler": sampler_settings.get("scheduler", "normal"),
            "denoise": 1.0,
        },
        title="Sampler",
    ))

    # 6. VAE Decode
    graph.add_node(NodeSpec(
        node_id="vae_decode",
        node_type=NodeTypes.VAE_DECODE,
        inputs={
            "samples": ["ksampler", 0],
            "vae": ["checkpoint_loader", 2],
        },
        title="VAE Decode",
    ))

    # 7. Save Image
    graph.add_node(NodeSpec(
        node_id="save_image",
        node_type=NodeTypes.SAVE_IMAGE,
        inputs={"images": ["vae_decode", 0], "filename_prefix": "comfyui_compile"},
        title="Save Image",
    ))

    # Optional: Camera Parameters node
    camera_node = _build_camera_node(reqs)
    if camera_node:
        graph.add_node(camera_node)

    # Optional: ControlNet preprocessors
    controlnet_nodes = _build_controlnet_nodes(reqs)
    for node in controlnet_nodes:
        graph.add_node(node)

    # Optional: Style transfer nodes
    style_nodes = _resolve_style_nodes(reqs.styles)
    for i, node_type in enumerate(style_nodes):
        graph.add_node(NodeSpec(
            node_id=f"style_transfer_{i}",
            node_type=node_type,
            inputs={"strength": 0.8},
            title=f"Style: {node_type}",
        ))

    return graph
