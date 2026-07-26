"""Tests for the validator module."""

from __future__ import annotations

from typing import Any

import pytest

from compiler.validator import (
    KNOWN_NODE_TYPES,
    validate_all,
    validate_connections,
    validate_node_types,
    validate_parameters,
)


class TestValidateNodeTypes:
    """Tests for node type validation."""

    def test_valid_node_types(self, sample_workflow: dict) -> None:
        errors = validate_node_types(sample_workflow)
        assert errors == []

    def test_unknown_node_type(self) -> None:
        workflow = {"1": {"class_type": "FakeNodeType", "inputs": {}}}
        errors = validate_node_types(workflow)
        assert len(errors) == 1
        assert "FakeNodeType" in errors[0]

    def test_missing_class_type(self) -> None:
        workflow = {"1": {"inputs": {}}}
        errors = validate_node_types(workflow)
        assert any("missing class_type" in e for e in errors)

    def test_known_types_are_valid(self) -> None:
        for node_type in KNOWN_NODE_TYPES:
            workflow = {"1": {"class_type": node_type, "inputs": {}}}
            errors = validate_node_types(workflow)
            assert errors == [], f"Known type {node_type} flagged as invalid"


class TestValidateConnections:
    """Tests for connection validation."""

    def test_valid_connections(self, sample_workflow: dict) -> None:
        errors = validate_connections(sample_workflow)
        assert errors == []

    def test_nonexistent_source_node(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"model": ["99", 0]},
            }
        }
        errors = validate_connections(workflow)
        assert any("non-existent node" in e for e in errors)

    def test_invalid_slot_type(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"model": ["1", "bad"]},
            }
        }
        errors = validate_connections(workflow)
        assert any("invalid slot" in e for e in errors)

    def test_negative_slot_index(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"model": ["1", -1]},
            }
        }
        errors = validate_connections(workflow)
        assert any("invalid slot" in e for e in errors)


class TestValidateParameters:
    """Tests for parameter type validation."""

    def test_valid_parameters(self, sample_workflow: dict) -> None:
        errors = validate_parameters(sample_workflow)
        assert errors == []

    def test_missing_required_param(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["0", 0],
                    # Missing: positive, negative, latent_image
                },
            }
        }
        errors = validate_parameters(workflow)
        missing_params = [e for e in errors if "missing required" in e]
        assert len(missing_params) >= 2

    def test_wrong_type_seed(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["0", 0],
                    "positive": ["0", 0],
                    "negative": ["0", 0],
                    "latent_image": ["0", 0],
                    "seed": "not_an_int",
                    "steps": 30,
                    "cfg": 7.5,
                    "denoise": 1.0,
                },
            }
        }
        errors = validate_parameters(workflow)
        assert any("seed" in e and "str" in e for e in errors)

    def test_wrong_type_steps(self) -> None:
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["0", 0],
                    "positive": ["0", 0],
                    "negative": ["0", 0],
                    "latent_image": ["0", 0],
                    "seed": 42,
                    "steps": "thirty",  # Should be int
                    "cfg": 7.5,
                    "denoise": 1.0,
                },
            }
        }
        errors = validate_parameters(workflow)
        assert any("steps" in e for e in errors)

    def test_wire_links_not_type_checked(self) -> None:
        """Wire links should not be checked for type conformance."""
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["0", 0],  # Wire link, not a direct value
                    "positive": ["0", 0],
                    "negative": ["0", 0],
                    "latent_image": ["0", 0],
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "denoise": 1.0,
                },
            }
        }
        errors = validate_parameters(workflow)
        # Should not complain about model being a list
        assert not any("model" in e and "list" in e for e in errors)

    def test_empty_latent_params(self) -> None:
        workflow = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": "not_int",
                    "height": 1024,
                    "batch_size": 1,
                },
            }
        }
        errors = validate_parameters(workflow)
        assert any("width" in e for e in errors)


class TestValidateAll:
    """Tests for the combined validation function."""

    def test_valid_workflow(self, sample_workflow: dict) -> None:
        errors = validate_all(sample_workflow)
        assert errors == []

    def test_catches_all_error_types(self, sample_workflow_invalid: dict) -> None:
        errors = validate_all(sample_workflow_invalid)
        assert len(errors) > 0
        # Should catch unknown type, non-existent reference, and wrong type
        error_text = " ".join(errors)
        assert "UnknownNodeType" in error_text or "unknown" in error_text.lower()

    def test_empty_workflow(self) -> None:
        errors = validate_all({})
        assert len(errors) > 0
