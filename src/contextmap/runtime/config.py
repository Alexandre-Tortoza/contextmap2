"""Versioned runtime configuration and deterministic effective-config resolution.

A runtime configuration says *what to run and with which implementations*; it never
says how a capability works. It is split into the concerns the architecture keeps
apart:

- ``pipeline``: the topology preset and which optional stages take part;
- ``components``: the backend selected for every variation point, with the
  parameters of that backend only;
- ``inputs``: which sequence and which upstream runs an execution starts from;
- ``resources``: device and workspace;
- ``policies``: how much debug evidence to persist.

An :class:`EffectiveConfig` is what remains after the layers are applied, in this
precedence (later wins): the profile defaults, the configuration files in the order
given, and the command-line overrides. Environment variables take no part in that
merge: they only carry secrets (:func:`resolve_secrets`), which are never stored, so
neither the effective configuration nor its digest can contain one.

The configuration schema version is independent of the ContextMap artifact schema
version: the former describes how a run is requested, the latter what a run produces.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from contextmap.runtime._files import publish_text
from contextmap.runtime.catalog import (
    CANONICAL_PROFILE_ID,
    COMPONENTS,
    PRESETS,
    ComponentSpec,
    RuntimePreset,
)

CONFIG_SCHEMA_VERSION = "0.1.0"
"""Version of the runtime configuration schema (not of the ContextMap schema)."""

EFFECTIVE_CONFIG_FILENAME = "effective_config.json"
"""Name under which a run persists its effective configuration."""

DEBUG_LEVELS = ("none", "standard", "full")
"""Debug levels every capability run artifact understands."""

ConfigValue: TypeAlias = (
    "bool | int | float | str | tuple[ConfigValue, ...] | Mapping[str, ConfigValue] | None"
)
"""A JSON-compatible value, frozen: sequences are tuples and mappings are read-only."""

_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "pipeline", "components", "inputs", "resources", "policies"}
)

_SECRET_TOKENS = frozenset({"secret", "secrets", "password", "passwd", "credential", "credentials"})
_SECRET_PAIRS = frozenset(
    {
        ("api", "key"),
        ("access", "key"),
        ("private", "key"),
        ("secret", "key"),
        ("auth", "token"),
        ("access", "token"),
        ("bearer", "token"),
        ("refresh", "token"),
        ("session", "token"),
    }
)
_NAME_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


@dataclass(frozen=True, kw_only=True)
class ConfigProblem:
    """One reason a configuration cannot be used as written.

    Attributes:
        path: Dotted location of the offending entry, empty when it concerns the
            configuration as a whole.
        message: What is wrong and, when known, how to fix it.
    """

    path: str
    message: str

    def __str__(self) -> str:
        """Render the problem as ``path: message``."""
        return f"{self.path}: {self.message}" if self.path else self.message


class ConfigurationError(Exception):
    """Raised when a runtime configuration is invalid or cannot be resolved.

    Attributes:
        problems: Every problem found, so a user fixes them in one pass.
    """

    def __init__(self, problems: Sequence[ConfigProblem]) -> None:
        """Build the error from every problem found."""
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"invalid runtime configuration:\n{lines}")

    @classmethod
    def single(cls, message: str, *, path: str = "") -> ConfigurationError:
        """Build an error that carries exactly one problem."""
        return cls([ConfigProblem(path=path, message=message)])


@dataclass(frozen=True, kw_only=True)
class PipelineConfig:
    """Topology selection.

    Attributes:
        preset: Identity of the topology preset, for example ``"canonical/1"``.
        stages: Whether each stage of the preset takes part, fully resolved: a
            stage the configuration was silent about carries its preset default.
    """

    preset: str
    stages: Mapping[str, bool]


@dataclass(frozen=True, kw_only=True)
class ComponentConfig:
    """The backend chosen for one variation point, with its own parameters.

    Attributes:
        backend: Identity of the selected backend, or ``None`` while unselected.
        parameters: Parameters of the selected backend only. The capability that owns
            the backend validates and interprets them; the runtime only carries them.
    """

    backend: str | None
    parameters: Mapping[str, ConfigValue]


@dataclass(frozen=True, kw_only=True)
class InputsConfig:
    """What an execution starts from.

    Attributes:
        sequence: Name or reference of the ingested sequence, when already chosen.
        selections: Exact upstream run or artifact reference chosen per stage.
    """

    sequence: str | None
    selections: Mapping[str, str]


@dataclass(frozen=True, kw_only=True)
class ResourcesConfig:
    """Where and on what an execution runs.

    Attributes:
        device: Device handed to every backend that declares a device parameter and
            does not set one itself, or ``None`` to leave each backend's own choice.
        workspace: Directory that holds this execution's artifacts, when chosen.
    """

    device: str | None
    workspace: str | None


@dataclass(frozen=True, kw_only=True)
class PoliciesConfig:
    """How much diagnostic evidence an execution persists.

    Attributes:
        debug_level: ``"none"``, ``"standard"`` or ``"full"``. Debug output is never a
            contractual dependency of a downstream stage.
    """

    debug_level: str


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    """A validated, fully resolved runtime configuration.

    Attributes:
        schema_version: Version of the configuration schema this document follows.
        pipeline: Topology selection.
        components: Selected backend per variation point of every enabled stage.
        inputs: Sequence and upstream selections.
        resources: Device and workspace.
        policies: Debug policy.
    """

    schema_version: str
    pipeline: PipelineConfig
    components: Mapping[str, ComponentConfig]
    inputs: InputsConfig
    resources: ResourcesConfig
    policies: PoliciesConfig

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible document this configuration is persisted as.

        Returns:
            A new nested dictionary; feeding it back through the resolver yields an
            equal configuration.
        """
        nested: dict[str, dict[str, Any]] = {}
        for component_id, component in self.components.items():
            capability, slot = component_id.split(".", 1)
            entry: dict[str, Any] = {"backend": component.backend}
            if component.backend is not None:
                entry[component.backend] = _thaw(component.parameters)
            nested.setdefault(capability, {})[slot] = entry
        return {
            "schema_version": self.schema_version,
            "pipeline": {
                "preset": self.pipeline.preset,
                "stages": dict(self.pipeline.stages),
            },
            "components": nested,
            "inputs": {
                "sequence": self.inputs.sequence,
                "selections": dict(self.inputs.selections),
            },
            "resources": {
                "device": self.resources.device,
                "workspace": self.resources.workspace,
            },
            "policies": {"debug_level": self.policies.debug_level},
        }


@dataclass(frozen=True, kw_only=True)
class ConfigurationSource:
    """One layer that contributed to an effective configuration.

    Attributes:
        kind: ``"profile"``, ``"file"`` or ``"override"``.
        identity: Profile identity, file path as given, or the dotted key an override
            set. An override's value is never recorded here.
        content_hash: SHA-256 of the file bytes for a file layer, otherwise ``None``.
    """

    kind: Literal["profile", "file", "override"]
    identity: str
    content_hash: str | None


@dataclass(frozen=True, kw_only=True)
class EffectiveConfig:
    """The configuration an execution actually uses, with its identity.

    Attributes:
        config: The fully resolved configuration.
        digest: ``"sha256:<hex>"`` of the canonical form of ``config``. Two
            configurations that resolve to the same values have the same digest,
            whatever files or overrides produced them.
        sources: The layers applied, lowest precedence first, for auditing.
    """

    config: RuntimeConfig
    digest: str
    sources: tuple[ConfigurationSource, ...]


def resolve_effective_config(
    *,
    profile: str = CANONICAL_PROFILE_ID,
    files: Sequence[str | os.PathLike[str]] = (),
    overrides: Sequence[str] = (),
) -> EffectiveConfig:
    """Resolve the effective configuration from its layers.

    Precedence, lowest to highest: the profile defaults, each file in the order
    given, then each override in the order given. Nested mappings merge key by key;
    every other value replaces. Nothing is read from the environment.

    Args:
        profile: Identity of the base profile and topology preset.
        files: JSON (``.json``) or TOML (``.toml``) configuration files.
        overrides: ``"dotted.path=value"`` overrides; the value is read as a JSON
            literal when it is one and as plain text otherwise.

    Returns:
        The validated effective configuration, its digest and its sources.

    Raises:
        ConfigurationError: If the profile is unknown, a file cannot be read, or the
            merged document is invalid or selects an incompatible combination.
    """
    document = _profile_document(profile)
    sources = [ConfigurationSource(kind="profile", identity=profile, content_hash=None)]
    for file in files:
        layer, content_hash = _load_file(Path(file))
        document = _deep_merge(document, layer)
        sources.append(
            ConfigurationSource(kind="file", identity=str(file), content_hash=content_hash)
        )
    for text in overrides:
        path, value = parse_override(text)
        document = _with_value(document, path, value)
        sources.append(
            ConfigurationSource(kind="override", identity=".".join(path), content_hash=None)
        )
    config = _build_config(document)
    return EffectiveConfig(
        config=config, digest=_digest(config.to_document()), sources=tuple(sources)
    )


def parse_override(text: str) -> tuple[tuple[str, ...], Any]:
    """Parse one ``"dotted.path=value"`` override.

    Args:
        text: The override as typed by a user.

    Returns:
        The path segments and the value: a JSON literal (``true``, ``12``, ``[1]``,
        ``"x"``, ``null``) when the text is one, otherwise the text itself.

    Raises:
        ConfigurationError: If there is no ``=``, or a path segment is empty.
    """
    key, separator, raw = text.partition("=")
    segments = tuple(key.split("."))
    if not separator or not key or any(not segment for segment in segments):
        raise ConfigurationError.single(
            f"override {text!r} must look like 'dotted.path=value' with no empty path segment"
        )
    try:
        value: Any = json.loads(raw)
    except ValueError:
        value = raw
    return segments, value


def check_selection(config: RuntimeConfig) -> tuple[ConfigProblem, ...]:
    """Report the variation points of enabled stages that still have no backend.

    The runtime never picks a backend on the user's behalf, so a configuration that
    resolves can still be incomplete. This lightweight check needs neither models nor
    optional modules and is meant to run before any execution.

    Args:
        config: A resolved configuration.

    Returns:
        One problem per unselected variation point; empty when the selection is
        complete.
    """
    problems = []
    for component_id, component in config.components.items():
        problem = check_component_selection(component_id, component)
        if problem is not None:
            problems.append(problem)
    return tuple(problems)


def check_component_selection(
    component_id: str, component: ComponentConfig
) -> ConfigProblem | None:
    """Report a variation point that has no backend selected.

    Args:
        component_id: Identity of the variation point, ``"<capability>.<slot>"``.
        component: Its resolved configuration.

    Returns:
        The problem, or ``None`` when a backend is selected.
    """
    if component.backend is not None:
        return None
    supported = ", ".join(sorted(COMPONENTS[component_id].backends))
    capability, slot = component_id.split(".", 1)
    return ConfigProblem(
        path=f"components.{component_id}",
        message=(
            f"no backend selected; choose one of: {supported} "
            f"(components.{capability}.{slot}.backend=<id>)"
        ),
    )


def check_availability(
    config: RuntimeConfig,
    *,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
) -> tuple[ConfigProblem, ...]:
    """Report selected backends whose optional modules or secrets are missing.

    Modules are only looked up, never imported, so this stays cheap and loads no
    model. A secret is reported by name; its value is never read into the report.

    Args:
        config: A resolved configuration.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether a top-level module can be
            imported; defaults to an :mod:`importlib` lookup.

    Returns:
        One problem per missing module or secret; empty when everything is present.
    """
    problems: list[ConfigProblem] = []
    for component_id, component in config.components.items():
        problems.extend(
            check_component_availability(
                component_id, component, environ=environ, module_available=module_available
            )
        )
    return tuple(problems)


def check_component_availability(
    component_id: str,
    component: ComponentConfig,
    *,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    check_modules: bool = True,
) -> tuple[ConfigProblem, ...]:
    """Report what one selected backend is missing, without loading anything.

    Args:
        component_id: Identity of the variation point, ``"<capability>.<slot>"``.
        component: Its resolved configuration.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether a top-level module can be
            imported; defaults to an :mod:`importlib` lookup.
        check_modules: Whether the backend's optional modules are required. A caller
            that supplies the model runtime itself passes ``False``: the modules are
            then the supplied runtime's concern, not the bundled code's.

    Returns:
        One problem per missing module or secret; empty when nothing is missing or no
        backend is selected.
    """
    if component.backend is None:
        return ()
    environment = os.environ if environ is None else environ
    available = module_available or _module_available
    spec = COMPONENTS[component_id].backends[component.backend]
    path = f"components.{component_id}"
    problems = []
    for module in spec.requires if check_modules else ():
        if not available(module):
            hint = f" ({spec.install_hint})" if spec.install_hint else ""
            problems.append(
                ConfigProblem(
                    path=path,
                    message=(
                        f"backend {component.backend!r} needs the optional module "
                        f"{module!r}, which is not installed{hint}"
                    ),
                )
            )
    for name in spec.secrets:
        if not environment.get(name):
            problems.append(
                ConfigProblem(
                    path=path,
                    message=(
                        f"backend {component.backend!r} needs the environment secret "
                        f"{name}; set it in the environment, it is never stored in "
                        "configuration"
                    ),
                )
            )
    return tuple(problems)


class ResolvedSecrets:
    """The environment secrets the selected backends declare, held in memory only.

    A value is reachable only through :meth:`get`; ``repr`` and ``str`` show names
    and never values, so a secret cannot leak into a log line or a manifest by
    formatting this object.

    Attributes:
        names: Every secret the selected backends declare, present or not.
        missing: The declared secrets absent from the environment.
    """

    __slots__ = ("_values", "missing", "names")

    def __init__(self, names: Sequence[str], values: Mapping[str, str]) -> None:
        """Keep the declared names and the values found for them."""
        self.names = tuple(names)
        self._values = dict(values)
        self.missing = tuple(name for name in self.names if name not in self._values)

    def get(self, name: str) -> str:
        """Return one secret's value.

        Args:
            name: A declared secret name.

        Returns:
            The value from the environment.

        Raises:
            ConfigurationError: If the secret is undeclared or not in the environment.
        """
        if name not in self._values:
            raise ConfigurationError.single(
                f"secret {name} is not available; set it in the environment"
            )
        return self._values[name]

    def __repr__(self) -> str:
        """Render names only."""
        return f"ResolvedSecrets(names={list(self.names)}, missing={list(self.missing)})"

    __str__ = __repr__


def resolve_secrets(
    config: RuntimeConfig, *, environ: Mapping[str, str] | None = None
) -> ResolvedSecrets:
    """Collect the secrets the selected backends declare from the environment.

    Args:
        config: A resolved configuration.
        environ: Environment to read; defaults to ``os.environ``.

    Returns:
        The secrets found and the ones missing. Nothing here is persisted.
    """
    environment = os.environ if environ is None else environ
    names: list[str] = []
    for component_id, component in config.components.items():
        if component.backend is None:
            continue
        for name in COMPONENTS[component_id].backends[component.backend].secrets:
            if name not in names:
                names.append(name)
    values = {name: environment[name] for name in names if environment.get(name)}
    return ResolvedSecrets(sorted(names), values)


def write_effective_config(effective: EffectiveConfig, directory: str | os.PathLike[str]) -> Path:
    """Persist the effective configuration, its digest and its sources.

    The write is atomic (a reader sees the whole file or none of it) and never
    replaces an existing file: writing the same configuration again is a no-op, and
    writing a different one over it is refused, because a published run is immutable.

    Args:
        effective: The effective configuration to persist.
        directory: Directory that receives ``effective_config.json``; created if
            missing.

    Returns:
        Path of the persisted file.

    Raises:
        ConfigurationError: If the directory already holds a different effective
            configuration.
    """
    target = Path(directory)
    final = target / EFFECTIVE_CONFIG_FILENAME
    text = (
        json.dumps(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "digest": effective.digest,
                "config": effective.config.to_document(),
                "sources": [
                    {
                        "kind": source.kind,
                        "identity": source.identity,
                        "content_hash": source.content_hash,
                    }
                    for source in effective.sources
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        return publish_text(target, EFFECTIVE_CONFIG_FILENAME, text)
    except FileExistsError:
        existing = read_effective_config(final)
        if existing.digest != effective.digest:
            raise ConfigurationError.single(
                f"{final} already holds a different effective configuration "
                f"({existing.digest}); a published run is never rewritten"
            ) from None
    return final


def read_effective_config(path: str | os.PathLike[str]) -> EffectiveConfig:
    """Read a persisted effective configuration and verify it.

    Args:
        path: Path of an ``effective_config.json`` file.

    Returns:
        The effective configuration, exactly as persisted.

    Raises:
        ConfigurationError: If the file is unreadable, follows another schema
            version, does not match its digest, or is not in normalized form.
    """
    source = Path(path)
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigurationError.single(f"cannot read {source}: {error}") from error
    if not isinstance(stored, dict) or not {"schema_version", "digest", "config", "sources"} <= (
        stored.keys()
    ):
        raise ConfigurationError.single(f"{source} is not an effective configuration document")
    if stored["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError.single(
            f"{source} follows schema_version {stored['schema_version']!r}, this runtime "
            f"reads {CONFIG_SCHEMA_VERSION!r}"
        )
    document = stored["config"]
    if _digest(document) != stored["digest"]:
        raise ConfigurationError.single(f"{source} does not match its digest; it was altered")
    config = _build_config(document)
    if config.to_document() != document:
        raise ConfigurationError.single(f"{source} holds a configuration that is not resolved")
    return EffectiveConfig(
        config=config,
        digest=stored["digest"],
        sources=tuple(
            ConfigurationSource(
                kind=entry["kind"],
                identity=entry["identity"],
                content_hash=entry["content_hash"],
            )
            for entry in stored["sources"]
        ),
    )


# --- camadas ----------------------------------------------------------------------


def _profile_document(profile: str) -> dict[str, Any]:
    """Return the defaults of a profile: the preset's topology with nothing selected."""
    preset = PRESETS.get(profile)
    if preset is None:
        known = ", ".join(sorted(PRESETS))
        raise ConfigurationError.single(f"unknown profile {profile!r}; known profiles: {known}")
    components: dict[str, dict[str, Any]] = {}
    for stage in preset.stages:
        for component_id in stage.components:
            capability, slot = component_id.split(".", 1)
            components.setdefault(capability, {})[slot] = {"backend": None}
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "pipeline": {
            "preset": preset.preset_id,
            "stages": {stage.stage_id: stage.default_enabled for stage in preset.stages},
        },
        "components": components,
        "inputs": {"sequence": None, "selections": {}},
        "resources": {"device": None, "workspace": None},
        "policies": {"debug_level": "none"},
    }


def _load_file(path: Path) -> tuple[dict[str, Any], str]:
    """Read one JSON or TOML layer and hash its bytes."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".toml"}:
        raise ConfigurationError.single(
            f"{path}: unsupported configuration file format {suffix!r}; use .json or .toml"
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigurationError.single(
            f"{path}: cannot read configuration file: {error}"
        ) from error
    try:
        text = raw.decode("utf-8")
        loaded = (
            tomllib.loads(text)
            if suffix == ".toml"
            else json.loads(text, parse_constant=_reject_constant)
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ConfigurationError.single(f"{path}: invalid {suffix[1:]} syntax: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError.single(f"{path}: the document root must be a mapping")
    return loaded, f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a valid JSON number")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings key by key; any other value in the overlay replaces the base."""
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _with_value(document: Mapping[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Return a copy of ``document`` with ``value`` set at ``path``."""
    result = copy.deepcopy(dict(document))
    node = result
    for depth, segment in enumerate(path[:-1]):
        child = node.setdefault(segment, {})
        if not isinstance(child, dict):
            here = ".".join(path[: depth + 1])
            raise ConfigurationError.single(
                f"cannot set {'.'.join(path)!r}: {here!r} is not a mapping"
            )
        node = child
    node[path[-1]] = value
    return result


# --- validação --------------------------------------------------------------------


def _build_config(document: object) -> RuntimeConfig:
    """Validate a merged document and normalize it into a :class:`RuntimeConfig`."""
    problems: list[ConfigProblem] = []
    config = _parse_document(document, problems)
    if config is not None:
        problems.extend(_incompatibilities(config))
    if problems or config is None:
        raise ConfigurationError(problems)
    return config


def _parse_document(document: object, problems: list[ConfigProblem]) -> RuntimeConfig | None:
    if not isinstance(document, Mapping):
        problems.append(ConfigProblem(path="", message="the configuration must be a mapping"))
        return None
    for key in document:
        if key not in _TOP_LEVEL_KEYS:
            problems.append(
                ConfigProblem(
                    path=str(key),
                    message=f"unknown section; expected one of {sorted(_TOP_LEVEL_KEYS)}",
                )
            )
    version = document.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        problems.append(
            ConfigProblem(
                path="schema_version",
                message=f"is {version!r}, this runtime reads {CONFIG_SCHEMA_VERSION!r}",
            )
        )
    preset, toggles = _parse_pipeline(document.get("pipeline", {}), problems)
    raw_components = _parse_components(document.get("components", {}), problems)
    inputs = _parse_inputs(document.get("inputs", {}), problems)
    resources = _parse_resources(document.get("resources", {}), problems)
    policies = _parse_policies(document.get("policies", {}), problems)
    if preset is None or problems:
        return None

    stages = {
        stage.stage_id: toggles.get(stage.stage_id, stage.default_enabled)
        for stage in preset.stages
    }
    components: dict[str, ComponentConfig] = {}
    for stage in preset.stages:
        if not stages[stage.stage_id]:
            continue  # estágio desligado não é construído: sua configuração não conta.
        for component_id in stage.components:
            components[component_id] = _resolve_component(
                COMPONENTS[component_id], raw_components.get(component_id), resources.device
            )
    return RuntimeConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        pipeline=PipelineConfig(preset=preset.preset_id, stages=MappingProxyType(stages)),
        components=MappingProxyType(components),
        inputs=inputs,
        resources=resources,
        policies=policies,
    )


def _parse_pipeline(
    value: object, problems: list[ConfigProblem]
) -> tuple[RuntimePreset | None, dict[str, bool]]:
    section = _mapping(value, "pipeline", problems)
    _reject_unknown(section, {"preset", "stages"}, "pipeline", problems)
    preset_id = section.get("preset", CANONICAL_PROFILE_ID)
    preset = PRESETS.get(preset_id) if isinstance(preset_id, str) else None
    if preset is None:
        problems.append(
            ConfigProblem(
                path="pipeline.preset",
                message=(
                    f"unknown preset {preset_id!r}; known presets: {', '.join(sorted(PRESETS))}"
                ),
            )
        )
    toggles: dict[str, bool] = {}
    for stage_id, enabled in _mapping(
        section.get("stages", {}), "pipeline.stages", problems
    ).items():
        path = f"pipeline.stages.{stage_id}"
        if not isinstance(enabled, bool):
            problems.append(ConfigProblem(path=path, message="must be true or false"))
            continue
        if preset is not None:
            try:
                declaration = preset.stage(stage_id)
            except KeyError:
                known = ", ".join(stage.stage_id for stage in preset.stages)
                problems.append(
                    ConfigProblem(path=path, message=f"unknown stage; known stages: {known}")
                )
                continue
            if not declaration.optional and not enabled:
                problems.append(
                    ConfigProblem(
                        path=path,
                        message=f"stage {stage_id!r} is not optional and cannot be disabled",
                    )
                )
                continue
        toggles[stage_id] = enabled
    return preset, toggles


@dataclass(frozen=True, kw_only=True)
class _RawComponent:
    backend: str | None
    blocks: Mapping[str, Mapping[str, ConfigValue]]


def _parse_components(value: object, problems: list[ConfigProblem]) -> dict[str, _RawComponent]:
    raw: dict[str, _RawComponent] = {}
    for capability, slots in _mapping(value, "components", problems).items():
        capability_path = f"components.{capability}"
        known_slots = {spec.slot for spec in COMPONENTS.values() if spec.capability == capability}
        if not known_slots:
            known = sorted({spec.capability for spec in COMPONENTS.values()})
            problems.append(
                ConfigProblem(path=capability_path, message=f"unknown capability; known: {known}")
            )
            continue
        for slot, entry in _mapping(slots, capability_path, problems).items():
            path = f"{capability_path}.{slot}"
            if slot not in known_slots:
                problems.append(
                    ConfigProblem(path=path, message=f"unknown slot; known: {sorted(known_slots)}")
                )
                continue
            parsed = _parse_component(COMPONENTS[f"{capability}.{slot}"], entry, path, problems)
            if parsed is not None:
                raw[f"{capability}.{slot}"] = parsed
    return raw


def _parse_component(
    spec: ComponentSpec, entry: object, path: str, problems: list[ConfigProblem]
) -> _RawComponent | None:
    section = _mapping(entry, path, problems)
    backend = section.get("backend")
    supported = sorted(spec.backends)
    if backend is not None and backend not in spec.backends:
        problems.append(
            ConfigProblem(
                path=f"{path}.backend",
                message=f"unknown backend {backend!r}; supported: {', '.join(supported)}",
            )
        )
    blocks: dict[str, Mapping[str, ConfigValue]] = {}
    for key, block in section.items():
        if key == "backend":
            continue
        if key not in spec.backends:
            problems.append(
                ConfigProblem(
                    path=f"{path}.{key}",
                    message=f"is neither 'backend' nor a supported backend; supported: "
                    f"{', '.join(supported)}",
                )
            )
            continue
        if not isinstance(block, Mapping):
            problems.append(ConfigProblem(path=f"{path}.{key}", message="must be a mapping"))
            continue
        frozen = _freeze(block, f"{path}.{key}", problems)
        if isinstance(frozen, Mapping):
            blocks[key] = frozen
    return _RawComponent(backend=backend if isinstance(backend, str) else None, blocks=blocks)


def _resolve_component(
    spec: ComponentSpec, raw: _RawComponent | None, device: str | None
) -> ComponentConfig:
    """Keep only the selected backend's parameters and hand it the resource device."""
    if raw is None or raw.backend is None:
        return ComponentConfig(backend=None, parameters=MappingProxyType({}))
    parameters = dict(raw.blocks.get(raw.backend, {}))
    device_parameter = spec.backends[raw.backend].device_parameter
    if device_parameter is not None and device is not None and device_parameter not in parameters:
        parameters[device_parameter] = device
    return ComponentConfig(backend=raw.backend, parameters=MappingProxyType(parameters))


def _parse_inputs(value: object, problems: list[ConfigProblem]) -> InputsConfig:
    section = _mapping(value, "inputs", problems)
    _reject_unknown(section, {"sequence", "selections"}, "inputs", problems)
    selections: dict[str, str] = {}
    for stage_id, reference in _mapping(
        section.get("selections", {}), "inputs.selections", problems
    ).items():
        if isinstance(reference, str) and reference:
            selections[stage_id] = reference
        else:
            problems.append(
                ConfigProblem(
                    path=f"inputs.selections.{stage_id}",
                    message="must be a non-empty run or artifact reference",
                )
            )
    return InputsConfig(
        sequence=_optional_text(section.get("sequence"), "inputs.sequence", problems),
        selections=MappingProxyType(selections),
    )


def _parse_resources(value: object, problems: list[ConfigProblem]) -> ResourcesConfig:
    section = _mapping(value, "resources", problems)
    _reject_unknown(section, {"device", "workspace"}, "resources", problems)
    return ResourcesConfig(
        device=_optional_text(section.get("device"), "resources.device", problems),
        workspace=_optional_text(section.get("workspace"), "resources.workspace", problems),
    )


def _parse_policies(value: object, problems: list[ConfigProblem]) -> PoliciesConfig:
    section = _mapping(value, "policies", problems)
    _reject_unknown(section, {"debug_level"}, "policies", problems)
    level = section.get("debug_level", "none")
    if level not in DEBUG_LEVELS:
        problems.append(
            ConfigProblem(
                path="policies.debug_level",
                message=f"is {level!r}; expected one of {', '.join(DEBUG_LEVELS)}",
            )
        )
        level = "none"
    return PoliciesConfig(debug_level=str(level))


def _declared_channels(parameters: Mapping[str, ConfigValue]) -> tuple[ConfigValue, ...]:
    """Read the evidence channels a fusion policy declares, baseline or quality-aware."""
    baseline = parameters.get("baseline")
    scopes = [parameters]
    if isinstance(baseline, Mapping):
        scopes.append(baseline)  # a política ciente de qualidade guarda os canais em `baseline`.
    channels: list[ConfigValue] = []
    for scope in scopes:
        declared = scope.get("channels")
        if isinstance(declared, tuple):
            channels.extend(declared)
    return tuple(channels)


def _incompatibilities(config: RuntimeConfig) -> list[ConfigProblem]:
    """Find selections that contradict each other, independent of what is installed."""
    problems = []
    accumulation = config.components.get("semantic_fusion.accumulation")
    if accumulation is not None and accumulation.backend is not None:
        uses_3d = "point_representation" in _declared_channels(accumulation.parameters)
        if uses_3d and not config.pipeline.stages.get("point_representation", False):
            problems.append(
                ConfigProblem(
                    path="components.semantic_fusion.accumulation",
                    message=(
                        "the 'point_representation' evidence channel needs the "
                        "'point_representation' stage; enable it with "
                        "pipeline.stages.point_representation=true or drop the channel"
                    ),
                )
            )
    return problems


# --- valores ----------------------------------------------------------------------


def _mapping(value: object, path: str, problems: list[ConfigProblem]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    problems.append(ConfigProblem(path=path, message="must be a mapping"))
    return {}


def _reject_unknown(
    section: Mapping[str, Any], allowed: set[str], path: str, problems: list[ConfigProblem]
) -> None:
    for key in section:
        if key not in allowed:
            problems.append(
                ConfigProblem(
                    path=f"{path}.{key}",
                    message=f"unknown entry; expected one of {sorted(allowed)}",
                )
            )


def _optional_text(value: object, path: str, problems: list[ConfigProblem]) -> str | None:
    if value is None or (isinstance(value, str) and value):
        return value
    problems.append(ConfigProblem(path=path, message="must be a non-empty string or null"))
    return None


def _freeze(value: object, path: str, problems: list[ConfigProblem]) -> ConfigValue:
    """Deep-freeze a JSON-compatible value, reporting anything else."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            problems.append(ConfigProblem(path=path, message="must be a finite number"))
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, ConfigValue] = {}
        for key, item in value.items():
            child = f"{path}.{key}"
            if not isinstance(key, str):
                problems.append(ConfigProblem(path=path, message=f"key {key!r} must be text"))
                continue
            if _looks_like_secret(key):
                problems.append(
                    ConfigProblem(
                        path=child,
                        message=(
                            "looks like a secret; secrets come from the environment and are "
                            "never stored in configuration"
                        ),
                    )
                )
                continue
            frozen[key] = _freeze(item, child, problems)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze(item, f"{path}[{index}]", problems) for index, item in enumerate(value)
        )
    problems.append(
        ConfigProblem(path=path, message=f"{type(value).__name__} is not a JSON-compatible value")
    )
    return None


def _thaw(value: ConfigValue) -> Any:
    """Turn a frozen value back into plain JSON containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _looks_like_secret(name: str) -> bool:
    """Tell whether a parameter name denotes a credential rather than a setting."""
    tokens = [token.lower() for token in _NAME_TOKEN.findall(name)]
    if tokens == ["token"] or any(token in _SECRET_TOKENS for token in tokens):
        return True
    return any(pair in _SECRET_PAIRS for pair in itertools.pairwise(tokens))


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _digest(document: object) -> str:
    """Hash the canonical JSON form of a document."""
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
