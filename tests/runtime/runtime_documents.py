"""Shared configuration documents for the runtime tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextmap.runtime import EffectiveConfig, resolve_effective_config

SHA_A = "a" * 40
SHA_B = "b" * 40

SUPPORT_POLICY = {
    "support_type": "neighborhood",
    "method": "radius",
    "radius_m": 0.5,
    "k": None,
    "max_neighbors": None,
    "preparation": {"centering": "centroid", "scale_normalization": "none"},
}


def selected_document() -> dict[str, Any]:
    return {
        "pipeline": {"stages": {"point_representation": True}},
        "resources": {"device": "cpu"},
        "components": {
            "ingestion": {"source_adapter": {"backend": "ros1_bag"}},
            "visual_perception": {
                "region_discovery": {
                    "backend": "sam3",
                    "sam3": {"checkpoint": "sam3-x", "strategy": "text_prompt", "prompt": "chair"},
                },
                "dense_features": {
                    "backend": "dinov3",
                    "dinov3": {"checkpoint": "facebook/dinov3-x", "revision": SHA_A},
                },
                "region_features": {
                    "backend": "clip",
                    "clip": {"checkpoint": "openai/clip-x", "revision": SHA_B},
                },
                "semantic_interpretation": {
                    "backend": "qwen",
                    "qwen": {
                        "model": "Qwen/Qwen-x",
                        "precision": "float32",
                        "max_new_tokens": 256,
                        "temperature": 0.0,
                    },
                },
            },
            "state_estimation": {
                "estimator": {
                    "backend": "external_pose",
                    "external_pose": {"reference_frame": "map", "body_frame": "base"},
                }
            },
            "point_representation": {
                "encoder": {
                    "backend": "geometric_descriptor",
                    "geometric_descriptor": {"support_policy": SUPPORT_POLICY},
                }
            },
            "semantic_fusion": {
                "support": {
                    "backend": "geometry-jaccard-support-v1",
                    "geometry-jaccard-support-v1": {"min_geometry_count": 5, "min_overlap": 0.3},
                },
                "accumulation": {"backend": "baseline-evidence-accumulation-v1"},
            },
        },
    }


def effective_from(tmp_path: Path, document: dict[str, Any] | None = None) -> EffectiveConfig:
    """Resolve the effective configuration of ``document`` (fully selected by default)."""
    file = tmp_path / "config.json"
    file.write_text(json.dumps(document or selected_document()), encoding="utf-8")
    return resolve_effective_config(files=[file])
