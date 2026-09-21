"""Tests for turning JSON-like backend parameters into a capability's config dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NewType

import pytest

from contextmap.runtime.coercion import ParameterError, build_config

Frame = NewType("Frame", str)
JsonScalar = str | int | float | bool | None


class Mode(Enum):
    FAST = "fast"
    EXACT = "exact"


@dataclass(frozen=True)
class Inner:
    radius: float
    label: str = "x"


@dataclass(frozen=True)
class Sample:
    name: str
    count: int = 1
    ratio: float = 0.5
    enabled: bool = False
    note: str | None = None
    limit: int | None = None
    mode: Mode = Mode.FAST
    frame: Frame = Frame("map")
    root: Path = Path(".")
    settings: tuple[tuple[str, JsonScalar], ...] = ()
    tags: frozenset[str] = frozenset()
    modes: frozenset[Mode] = frozenset()
    scalars: tuple[float, ...] = ()
    extras: dict[str, JsonScalar] = field(default_factory=dict)
    inner: Inner | None = None
    inners: tuple[Inner, ...] = ()

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must not be negative")


class TestBuildConfig:
    def test_fills_defaults_and_coerces_every_supported_type(self) -> None:
        sample = build_config(
            Sample,
            {
                "name": "a",
                "count": 3,
                "ratio": 2,
                "enabled": True,
                "note": None,
                "mode": "exact",
                "frame": "odom",
                "root": "/data",
                "settings": {"b": 2, "a": "x"},
                "tags": ["t2", "t1", "t1"],
                "modes": ["fast", "exact"],
                "scalars": [1, 2.5],
                "extras": {"k": 1},
                "inner": {"radius": 1},
                "inners": [{"radius": 2, "label": "y"}],
            },
        )

        assert sample.count == 3
        assert sample.ratio == 2.0 and isinstance(sample.ratio, float)
        assert sample.mode is Mode.EXACT
        assert sample.frame == "odom"
        assert sample.root == Path("/data")
        assert sample.settings == (("a", "x"), ("b", 2))
        assert sample.tags == frozenset({"t1", "t2"})
        assert sample.modes == frozenset({Mode.FAST, Mode.EXACT})
        assert sample.scalars == (1.0, 2.5)
        assert sample.extras == {"k": 1}
        assert sample.inner == Inner(radius=1.0)
        assert sample.inners == (Inner(radius=2.0, label="y"),)

    def test_defaults_apply_when_a_parameter_is_absent(self) -> None:
        sample = build_config(Sample, {"name": "a"})

        assert sample == Sample(name="a")

    def test_reports_every_problem_with_its_path(self) -> None:
        with pytest.raises(ParameterError) as error:
            build_config(
                Sample,
                {
                    "count": "3",
                    "mode": "slow",
                    "unknown": 1,
                    "inner": {"radius": "big"},
                    "tags": "not-a-list",
                },
            )

        text = "\n".join(error.value.problems)
        assert "name" in text and "required" in text
        assert "count" in text
        assert "mode" in text and "fast" in text and "exact" in text
        assert "unknown" in text
        assert "inner.radius" in text
        assert "tags" in text

    def test_a_bool_is_never_taken_for_a_number_and_a_number_never_for_a_bool(self) -> None:
        with pytest.raises(ParameterError):
            build_config(Sample, {"name": "a", "count": True})
        with pytest.raises(ParameterError):
            build_config(Sample, {"name": "a", "ratio": False})
        with pytest.raises(ParameterError):
            build_config(Sample, {"name": "a", "enabled": 1})

    def test_the_capabilitys_own_validation_message_is_kept(self) -> None:
        with pytest.raises(ParameterError, match="count must not be negative"):
            build_config(Sample, {"name": "a", "count": -1})

    def test_accepts_frozen_read_only_inputs(self) -> None:
        from types import MappingProxyType

        sample = build_config(
            Sample,
            MappingProxyType(
                {"name": "a", "settings": MappingProxyType({"k": 1}), "scalars": (1, 2)}
            ),
        )

        assert sample.settings == (("k", 1),)
        assert sample.scalars == (1.0, 2.0)


class TestRealCapabilityConfigs:
    def test_builds_a_sam3_config_with_an_enum_and_pair_settings(self) -> None:
        from contextmap.visual_perception.backends.sam3 import Sam3Config, Sam3Strategy

        config = build_config(
            Sam3Config,
            {
                "checkpoint": "sam3-x",
                "strategy": "text_prompt",
                "prompt": "chair",
                "score_threshold": 0.3,
                "strategy_settings": {"points_per_side": 16},
            },
        )

        assert config.strategy is Sam3Strategy.TEXT_PROMPT
        assert config.strategy_settings == (("points_per_side", 16),)

    def test_the_capability_rejects_an_inconsistent_config_before_any_model_loads(self) -> None:
        from contextmap.visual_perception.backends.sam3 import Sam3Config

        with pytest.raises(ParameterError, match="text_prompt strategy requires a prompt"):
            build_config(Sam3Config, {"checkpoint": "x", "strategy": "text_prompt"})

    def test_names_the_valid_parameters_when_one_is_unknown(self) -> None:
        from contextmap.visual_perception.backends.sam3 import Sam3Config

        with pytest.raises(ParameterError, match="checkpoint"):
            build_config(Sam3Config, {"checkpoint": "x", "checkpont": "typo"})
