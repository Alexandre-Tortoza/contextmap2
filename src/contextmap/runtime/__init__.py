"""Public contract for the runtime capability.

The runtime resolves configuration, selects and constructs backends, and orchestrates
stages. It owns no scientific logic: segmentation, projection, fusion and every other
domain rule stay inside their capabilities. See ``docs/runtime-composition.md`` and
``src/contextmap/runtime/docs/README.md``.
"""

from contextmap.runtime.catalog import (
    CANONICAL_PROFILE_ID,
    BackendSpec,
    ComponentSpec,
    RuntimePreset,
    StageDeclaration,
)
from contextmap.runtime.config import (
    CONFIG_SCHEMA_VERSION,
    DEBUG_LEVELS,
    EFFECTIVE_CONFIG_FILENAME,
    ComponentConfig,
    ConfigProblem,
    ConfigurationError,
    ConfigurationSource,
    EffectiveConfig,
    InputsConfig,
    PipelineConfig,
    PoliciesConfig,
    ResolvedSecrets,
    ResourcesConfig,
    RuntimeConfig,
    check_availability,
    check_selection,
    parse_override,
    read_effective_config,
    resolve_effective_config,
    resolve_secrets,
    write_effective_config,
)

__all__ = [
    "CANONICAL_PROFILE_ID",
    "CONFIG_SCHEMA_VERSION",
    "DEBUG_LEVELS",
    "EFFECTIVE_CONFIG_FILENAME",
    "BackendSpec",
    "ComponentConfig",
    "ComponentSpec",
    "ConfigProblem",
    "ConfigurationError",
    "ConfigurationSource",
    "EffectiveConfig",
    "InputsConfig",
    "PipelineConfig",
    "PoliciesConfig",
    "ResolvedSecrets",
    "ResourcesConfig",
    "RuntimeConfig",
    "RuntimePreset",
    "StageDeclaration",
    "check_availability",
    "check_selection",
    "parse_override",
    "read_effective_config",
    "resolve_effective_config",
    "resolve_secrets",
    "write_effective_config",
]
