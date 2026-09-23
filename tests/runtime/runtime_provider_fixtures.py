"""Importable ``RuntimeProvider`` targets for ``resources.providers`` tests.

These exist only so a test can declare a real ``"module:attribute"`` string in
configuration and let :func:`contextmap.runtime.composition.resolve_provider` import it
for real, exactly as the installed ``contextmap`` binary would for a deployment's own
loader module -- proving the declarative mechanism end to end, not just its plumbing.
"""

from __future__ import annotations

from typing import Any

CALLS: list[tuple[str, Any]] = []
"""Every call this module's providers received, for a test to assert against."""


def load_region_discovery(config: Any, secrets: Any) -> object:
    """Return a placeholder region-discovery runtime, recording that it was asked for."""
    CALLS.append(("region_discovery", config))
    return object()


def load_semantic_interpretation(config: Any, secrets: Any) -> object:
    """Return a placeholder semantic-interpretation runtime, recording the call."""
    CALLS.append(("semantic_interpretation", config))
    return object()


def load_dense_features(config: Any, secrets: Any) -> object:
    """Return a placeholder dense-feature runtime, recording the call."""
    CALLS.append(("dense_features", config))
    return object()


def load_region_features(config: Any, secrets: Any) -> object:
    """Return a placeholder region-feature runtime, recording the call."""
    CALLS.append(("region_features", config))
    return object()


not_callable = 42
"""A non-callable attribute, for testing that :func:`resolve_provider` rejects it."""
