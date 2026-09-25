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
        "inputs": {"sequence": "S1"},
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
            "geometric_mapping": {
                "pose_lookup": {
                    "backend": "lookup-policy-v1",
                    "lookup-policy-v1": {"mode": "exact"},
                },
                "motion_correction": {
                    "backend": "motion-correction-v1",
                    "motion-correction-v1": {"raw": "accept", "unknown": "warn"},
                },
            },
            "sensor_association": {
                "occlusion": {
                    "backend": "conservative-depth-support-v1",
                    "conservative-depth-support-v1": {
                        "cell_size_px": 4,
                        "neighborhood_radius_cells": 2,
                        "depth_margin_m": 0.1,
                        "depth_margin_ratio": 0.02,
                    },
                },
                "tolerances": {
                    "backend": "diagnostic-tolerances-v1",
                    "diagnostic-tolerances-v1": {
                        "max_pose_time_delta_ns": 20_000_000,
                        "max_map_window_offset_ns": 10_000_000,
                        "max_reprojection_p95_px": 5.0,
                        "max_reprojection_invalid_rate": 0.25,
                    },
                },
                "pose_policy": {
                    "backend": "lookup-policy-v1",
                    "lookup-policy-v1": {"mode": "exact"},
                },
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
            "semantic_mapping": {
                "geometry_summary": {
                    "backend": "entity-geometry-summary-v1",
                    "entity-geometry-summary-v1": {
                        "sparse_point_threshold": 3,
                        "connectivity_radius_m": 0.5,
                    },
                },
            },
            "entity_resolution": {
                "retrieval": {
                    "backend": "entity-candidate-retrieval-v1",
                    "entity-candidate-retrieval-v1": {
                        "centroid_radius_m": 20.0,
                        "bounds_margin_m": 0.1,
                    },
                },
                "resolution": {
                    "backend": "conservative-staged-resolution-v1",
                    "conservative-staged-resolution-v1": {
                        "use_channels": ["geometry"],
                        "min_supporting_channels": 1,
                    },
                },
                "geometry_comparison": {
                    "backend": "entity-geometry-comparison-v1",
                    "entity-geometry-comparison-v1": {
                        "min_shared_support_jaccard": 0.5,
                        "min_bounds_iou": 0.5,
                        "min_bounds_containment": 0.9,
                        "min_conflict_gap_m": 0.5,
                        "min_extent_ratio": 0.3,
                    },
                },
            },
            "spatial_relations": {
                "frame_conventions": {
                    "backend": "map-frame-conventions-v1",
                    "map-frame-conventions-v1": {
                        "map_frame": "odom",
                        "up_axis": "+z",
                        "forward_axis": "+x",
                    },
                },
                "candidate": {
                    "backend": "bounds-neighborhood-candidates-v1",
                    "bounds-neighborhood-candidates-v1": {
                        "predicates": ["next_to", "touching"],
                        "proximity_radius_m": 0.6,
                        "directional_radius_m": 2.0,
                    },
                },
                "geometry_summary": {
                    "backend": "entity-geometry-summary-v1",
                    "entity-geometry-summary-v1": {
                        "sparse_point_threshold": 3,
                        "connectivity_radius_m": 0.5,
                    },
                },
            },
        },
    }


def effective_from(tmp_path: Path, document: dict[str, Any] | None = None) -> EffectiveConfig:
    """Resolve the effective configuration of ``document`` (fully selected by default)."""
    file = tmp_path / "config.json"
    file.write_text(json.dumps(document or selected_document()), encoding="utf-8")
    return resolve_effective_config(files=[file])
