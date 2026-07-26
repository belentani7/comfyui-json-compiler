"""Generate valid ComfyUI workflow JSON from node graphs."""

from __future__ import annotations

import random
from typing import Any

from compiler.node_mapper import NodeGraph, NodeSpec


def _serialize_input(value: str | int | float | list[str]) -> dict[str, Any]:
    """Serialize a node input value into ComfyUI's wire format.

    Link references (lists like ["node_id", slot]) become wire links.
    Primitive values become direct inputs.
    """
    if isinstance(value, list) and len(value) == 2:
        return {"0": [str(value[0]), int(value[1])]}
    return {"0": value}


def _serialize_node(spec: NodeSpec) -> dict[str, Any]:
    """Serialize a NodeSpec into ComfyUI's node dict format."""
    node_dict: dict[str, Any] = {
        "class_type": spec.node_type,
        "_meta": {"title": spec.title or spec.node_type},
    }

    inputs: dict[str, Any] = {}
    for key, value in spec.inputs.items():
        if isinstance(value, list) and len(value) == 2:
            # Wire link: [source_node_id, source_slot]
            inputs[key] = [str(value[0]), int(value[1])]
        else:
            inputs[key] = value

    node_dict["inputs"] = inputs
    return node_dict


def generate(reqs: Any, node_graph: NodeGraph) -> dict[str, Any]:
    """Generate a complete ComfyUI workflow JSON from requirements and node graph.

    Produces a workflow dict ready to be POSTed to ComfyUI's API or
    loaded in the ComfyUI web interface.
    """
    workflow: dict[str, Any] = {}

    # Assign integer IDs for ComfyUI (it uses string keys but integer values)
    node_id_map: dict[str, str] = {}
    for i, node_id in enumerate(node_graph.nodes.keys()):
        node_id_map[node_id] = str(i + 1)

    # Serialize each node
    for original_id, spec in node_graph.nodes.items():
        serialized = _serialize_node(spec)

        # Rewrite wire links to use the new integer IDs
        new_inputs: dict[str, Any] = {}
        for key, value in spec.inputs.items():
            if isinstance(value, list) and len(value) == 2:
                source_id = str(value[0])
                source_slot = int(value[1])
                if source_id in node_id_map:
                    new_inputs[key] = [node_id_map[source_id], source_slot]
                else:
                    new_inputs[key] = value
            else:
                new_inputs[key] = value
        serialized["inputs"] = new_inputs

        # Add group header info for visual organization
        if original_id.startswith("preprocessor"):
            serialized["_meta"]["section"] = "controlnet"
        elif original_id.startswith("style"):
            serialized["_meta"]["section"] = "style_transfer"

        workflow[node_id_map[original_id]] = serialized

    return workflow


def validate_workflow(workflow: dict[str, Any]) -> list[str]:
    """Validate a ComfyUI workflow for structural correctness.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not workflow:
        errors.append("Workflow is empty")
        return errors

    # Check that each node has required fields
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            errors.append(f"Node {node_id}: expected dict, got {type(node_data).__name__}")
            continue

        if "class_type" not in node_data:
            errors.append(f"Node {node_id}: missing 'class_type'")

        if "inputs" not in node_data:
            errors.append(f"Node {node_id}: missing 'inputs'")
        elif not isinstance(node_data["inputs"], dict):
            errors.append(f"Node {node_id}: 'inputs' must be a dict")

    # Check that wire links reference existing nodes
    all_node_ids = set(workflow.keys())
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict) or "inputs" not in node_data:
            continue
        for param_name, param_value in node_data["inputs"].items():
            if isinstance(param_value, list) and len(param_value) == 2:
                ref_id = str(param_value[0])
                if ref_id not in all_node_ids:
                    errors.append(
                        f"Node {node_id}, param '{param_name}': "
                        f"references non-existent node '{ref_id}'"
                    )

    return errors


def optimize_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Optimize a workflow by removing unreachable nodes and compacting IDs.

    Returns a new optimized workflow dict.
    """
    if not workflow:
        return workflow

    # Find all nodes referenced by wire links
    reachable: set[str] = set()
    to_visit: list[str] = []

    # Start from save_image / output nodes (nodes with no outgoing links)
    outgoing_refs: dict[str, set[str]] = {nid: set() for nid in workflow}
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict) or "inputs" not in node_data:
            continue
        for param_value in node_data["inputs"].values():
            if isinstance(param_value, list) and len(param_value) == 2:
                ref_id = str(param_value[0])
                if ref_id in outgoing_refs:
                    outgoing_refs[ref_id].add(node_id)

    # BFS from nodes that are never referenced (output nodes)
    for node_id in workflow:
        is_output = len(outgoing_refs.get(node_id, set())) == 0
        if is_output:
            to_visit.append(node_id)

    while to_visit:
        current = to_visit.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        node_data = workflow.get(current, {})
        if isinstance(node_data, dict) and "inputs" in node_data:
            for param_value in node_data["inputs"].values():
                if isinstance(param_value, list) and len(param_value) == 2:
                    ref_id = str(param_value[0])
                    if ref_id not in reachable:
                        to_visit.append(ref_id)

    # Keep only reachable nodes and remap IDs
    optimized: dict[str, Any] = {}
    old_to_new: dict[str, str] = {}
    new_id = 1

    for old_id in workflow:
        if old_id in reachable:
            old_to_new[old_id] = str(new_id)
            new_id += 1

    for old_id in workflow:
        if old_id not in old_to_new:
            continue
        node_data = workflow[old_id]
        new_node_data: dict[str, Any] = {
            "class_type": node_data.get("class_type", "Unknown"),
            "_meta": node_data.get("_meta", {}),
        }
        new_inputs: dict[str, Any] = {}
        for param_name, param_value in node_data.get("inputs", {}).items():
            if isinstance(param_value, list) and len(param_value) == 2:
                ref_id = str(param_value[0])
                if ref_id in old_to_new:
                    new_inputs[param_name] = [old_to_new[ref_id], param_value[1]]
                else:
                    new_inputs[param_name] = param_value
            else:
                new_inputs[param_name] = param_value
        new_node_data["inputs"] = new_inputs
        optimized[old_to_new[old_id]] = new_node_data

    return optimized
