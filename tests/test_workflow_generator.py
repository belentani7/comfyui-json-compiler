"""Tests for the workflow generator module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from compiler.brief_parser import BriefRequirements
from compiler.node_mapper import NodeGraph, NodeSpec, build_node_graph
from compiler.workflow_generator import generate, optimize_workflow, validate_workflow


class TestGenerate:
    """Tests for workflow generation."""

    def test_generates_valid_json(
        self, sample_requirements: BriefRequirements, sample_workflow: dict
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        result = generate(sample_requirements, node_graph)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_nodes_have_class_type(
        self, sample_requirements: BriefRequirements
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        result = generate(sample_requirements, node_graph)
        for node_id, node_data in result.items():
            assert "class_type" in node_data, f"Node {node_id} missing class_type"

    def test_all_nodes_have_inputs(
        self, sample_requirements: BriefRequirements
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        result = generate(sample_requirements, node_graph)
        for node_id, node_data in result.items():
            assert "inputs" in node_data, f"Node {node_id} missing inputs"
            assert isinstance(node_data["inputs"], dict)

    def test_wire_links_are_lists(
        self, sample_requirements: BriefRequirements
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        result = generate(sample_requirements, node_graph)
        for node_id, node_data in result.items():
            for param, value in node_data.get("inputs", {}).items():
                if isinstance(value, list):
                    assert len(value) == 2, (
                        f"Node {node_id}, param {param}: wire link must be [node_id, slot]"
                    )

    def test_node_ids_are_strings(self, sample_requirements: BriefRequirements) -> None:
        node_graph = build_node_graph(sample_requirements)
        result = generate(sample_requirements, node_graph)
        for node_id in result:
            assert isinstance(node_id, str)

    def test_empty_graph(self) -> None:
        empty_graph = NodeGraph()
        result = generate(None, empty_graph)
        assert result == {}


class TestValidateWorkflow:
    """Tests for workflow validation."""

    def test_valid_workflow(self, sample_workflow: dict) -> None:
        errors = validate_workflow(sample_workflow)
        assert errors == []

    def test_empty_workflow(self) -> None:
        errors = validate_workflow({})
        assert len(errors) > 0
        assert "empty" in errors[0].lower()

    def test_missing_class_type(self) -> None:
        workflow = {"1": {"inputs": {"param": "value"}}}
        errors = validate_workflow(workflow)
        assert any("class_type" in e for e in errors)

    def test_missing_inputs(self) -> None:
        workflow = {"1": {"class_type": "KSampler"}}
        errors = validate_workflow(workflow)
        assert any("inputs" in e for e in errors)

    def test_invalid_wire_reference(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"model": ["99", 0]},
            }
        }
        errors = validate_workflow(workflow)
        assert any("non-existent" in e for e in errors)

    def test_node_dict_value_error(self) -> None:
        workflow = {"1": "not_a_dict"}
        errors = validate_workflow(workflow)
        assert any("expected dict" in e for e in errors)


class TestOptimizeWorkflow:
    """Tests for workflow optimization."""

    def test_removes_unreachable_nodes(
        self, sample_requirements: BriefRequirements
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        original = generate(sample_requirements, node_graph)

        # Add an orphan node
        original["99"] = {
            "class_type": "PreviewImage",
            "inputs": {"images": ["6", 0]},
        }

        optimized = optimize_workflow(original)
        assert "99" not in optimized

    def test_preserves_reachable_nodes(
        self, sample_requirements: BriefRequirements
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        original = generate(sample_requirements, node_graph)
        original_count = len(original)

        optimized = optimize_workflow(original)
        # Should have same or fewer nodes
        assert len(optimized) <= original_count

    def test_remaps_ids_sequentially(
        self, sample_requirements: BriefRequirements
    ) -> None:
        node_graph = build_node_graph(sample_requirements)
        original = generate(sample_requirements, node_graph)
        optimized = optimize_workflow(original)

        ids = sorted(int(k) for k in optimized.keys())
        assert ids == list(range(1, len(ids) + 1))

    def test_empty_workflow(self) -> None:
        result = optimize_workflow({})
        assert result == {}


class TestGenerateFromNodeGraph:
    """Tests that node graph correctly feeds into workflow generation."""

    def test_core_pipeline_nodes_present(self) -> None:
        reqs = BriefRequirements(
            subject="a test subject",
            styles=["photorealistic"],
            camera={},
            lighting={},
            mood="neutral",
            modifiers=[],
        )
        graph = build_node_graph(reqs)
        workflow = generate(reqs, graph)

        class_types = {n["class_type"] for n in workflow.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "CLIPTextEncode" in class_types
        assert "KSampler" in class_types
        assert "VAEDecode" in class_types
        assert "SaveImage" in class_types

    def test_controlnet_nodes_added_for_dof(self) -> None:
        reqs = BriefRequirements(
            subject="a portrait",
            styles=[],
            camera={"shot_type": "close_up", "depth_of_field": "shallow"},
            lighting={},
            mood="neutral",
            modifiers=[],
        )
        graph = build_node_graph(reqs)
        workflow = generate(reqs, graph)

        class_types = {n["class_type"] for n in workflow.values()}
        assert "DepthAnythingPreprocessor" in class_types

    def test_camera_node_added(self) -> None:
        reqs = BriefRequirements(
            subject="a landscape",
            styles=[],
            camera={"shot_type": "wide_angle", "angle": "low_angle"},
            lighting={},
            mood="neutral",
            modifiers=[],
        )
        graph = build_node_graph(reqs)
        workflow = generate(reqs, graph)

        class_types = {n["class_type"] for n in workflow.values()}
        assert "CameraParameters" in class_types
