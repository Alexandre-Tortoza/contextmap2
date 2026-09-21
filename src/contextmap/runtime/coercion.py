"""Build a capability's own configuration object from JSON-like backend parameters.

The runtime carries backend parameters as plain values; the capability that owns the
backend defines what they mean, in a configuration dataclass with its own validation.
Instead of hand-mapping every parameter of every backend, this module reads the types
the dataclass declares and converts the JSON values accordingly, so the runtime never
duplicates a capability's parameters, defaults or scientific rules. Whatever the
dataclass rejects is reported with its own message, before any model is loaded.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")


class ParameterError(ValueError):
    """Raised when parameters do not fit the configuration they are meant for.

    Attributes:
        problems: Every problem found, each prefixed by the parameter path.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        """Build the error from every problem found."""
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems))


def build_config(target: type[T], parameters: Any) -> T:
    """Construct ``target`` from JSON-like parameters.

    Supported field types are the ones the capabilities' configuration dataclasses use:
    ``str``, ``int``, ``float``, ``bool``, ``Path``, enums (by value), ``NewType`` over
    those, optionals, ``tuple[X, ...]``, ``frozenset[X]``, ``Mapping[str, X]``, nested
    dataclasses and the ``tuple[tuple[str, X], ...]`` pair-list idiom, which is written
    as a mapping and stored sorted by key. A ``bool`` is never taken for a number and
    vice versa.

    Args:
        target: The configuration dataclass to build.
        parameters: The parameters, as found in the runtime configuration; a value that
            is not a mapping is reported as a problem.

    Returns:
        The constructed configuration.

    Raises:
        ParameterError: If a parameter is unknown, missing, of the wrong type, or if
            the dataclass rejects the resulting values.
    """
    problems: list[str] = []
    built = _build_dataclass(target, parameters, "", problems)
    if problems or built is None:
        raise ParameterError(problems)
    return built


def _build_dataclass(target: type[T], value: object, path: str, problems: list[str]) -> T | None:
    if not isinstance(value, Mapping):
        problems.append(f"{_label(path)}must be a mapping")
        return None
    hints = typing.get_type_hints(target)
    fields = {field.name: field for field in dataclasses.fields(target) if field.init}  # type: ignore[arg-type]
    failed = False
    for key in value:
        if key not in fields:
            valid = ", ".join(sorted(fields))
            problems.append(f"{_join(path, str(key))}: unknown parameter; valid: {valid}")
            failed = True
    missing = [
        name
        for name, field in fields.items()
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        and name not in value
    ]
    if missing:
        problems.append(f"{_label(path)}missing required parameter(s): {', '.join(missing)}")
        failed = True
    arguments: dict[str, object] = {}
    for name in fields:
        if name not in value:
            continue
        before = len(problems)
        coerced = _coerce(value[name], hints[name], _join(path, name), problems)
        if len(problems) > before:
            failed = True
        else:
            arguments[name] = coerced
    if failed:
        return None
    try:
        return target(**arguments)
    except (ValueError, TypeError) as error:
        problems.append(f"{_label(path)}{error}")
        return None


def _coerce(value: object, annotation: object, path: str, problems: list[str]) -> object:
    if annotation is Any or annotation is object:
        return value
    if annotation is type(None):
        return _expect(value is None, None, path, "must be null", problems)
    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:  # NewType
        base = _coerce(value, supertype, path, problems)
        return cast("Callable[[str], object]", annotation)(base) if isinstance(base, str) else base
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        return _coerce_union(value, arguments, path, problems)
    if annotation is bool:
        return _expect(isinstance(value, bool), value, path, "must be true or false", problems)
    if annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        return _expect(ok, value, path, "must be an integer", problems)
    if annotation is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        return _expect(ok, float(value) if ok else value, path, "must be a number", problems)  # type: ignore[arg-type]
    if annotation is str:
        return _expect(isinstance(value, str), value, path, "must be text", problems)
    if annotation is Path:
        if isinstance(value, str) and value:
            return Path(value)
        problems.append(f"{path}: must be a non-empty path")
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError:
            allowed = ", ".join(repr(member.value) for member in annotation)
            problems.append(f"{path}: {value!r} is not one of {allowed}")
            return None
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build_dataclass(annotation, value, path, problems)
    if origin is tuple:
        return _coerce_tuple(value, arguments, path, problems)
    if origin in (frozenset, set):
        return _coerce_set(value, arguments[0], path, problems)
    if origin in (dict, Mapping) or annotation is dict:
        return _coerce_mapping(value, arguments, path, problems)
    problems.append(f"{path}: unsupported parameter type {annotation!r}")
    return None


def _coerce_union(
    value: object, options: tuple[Any, ...], path: str, problems: list[str]
) -> object:
    if value is None:
        if type(None) in options:
            return None
        problems.append(f"{path}: must not be null")
        return None
    candidates = [option for option in options if option is not type(None)]
    if len(candidates) == 1:
        # Optional[X]: os problemas do próprio X explicam melhor do que "deve ser X".
        return _coerce(value, candidates[0], path, problems)
    for option in candidates:
        attempt: list[str] = []
        coerced = _coerce(value, option, path, attempt)
        if not attempt:
            return coerced
    expected = " or ".join(_describe(option) for option in candidates)
    problems.append(f"{path}: must be {expected}")
    return None


def _coerce_tuple(
    value: object, arguments: tuple[Any, ...], path: str, problems: list[str]
) -> object:
    variadic = len(arguments) == 2 and arguments[1] is Ellipsis
    element = arguments[0] if variadic else None
    # Pares chave/valor escritos como mapeamento: guardados ordenados por chave.
    if variadic and isinstance(value, Mapping) and typing.get_origin(element) is tuple:
        pair = typing.get_args(element)
        if len(pair) == 2 and pair[0] is str:
            return tuple(
                (key, _coerce(item, pair[1], _join(path, str(key)), problems))
                for key, item in sorted(value.items())
            )
    if isinstance(value, str) or not isinstance(value, Sequence):
        problems.append(f"{path}: must be a list")
        return None
    if variadic:
        return tuple(
            _coerce(item, element, f"{path}[{index}]", problems) for index, item in enumerate(value)
        )
    if len(value) != len(arguments):
        problems.append(f"{path}: must have exactly {len(arguments)} items")
        return None
    return tuple(
        _coerce(item, kind, f"{path}[{index}]", problems)
        for index, (item, kind) in enumerate(zip(value, arguments, strict=True))
    )


def _coerce_set(value: object, element: object, path: str, problems: list[str]) -> object:
    if isinstance(value, str) or not isinstance(value, Sequence | set | frozenset):
        problems.append(f"{path}: must be a list")
        return None
    return frozenset(
        _coerce(item, element, f"{path}[{index}]", problems) for index, item in enumerate(value)
    )


def _coerce_mapping(
    value: object, arguments: tuple[Any, ...], path: str, problems: list[str]
) -> object:
    if not isinstance(value, Mapping):
        problems.append(f"{path}: must be a mapping")
        return None
    item_type = arguments[1] if len(arguments) == 2 else Any
    return {
        str(key): _coerce(item, item_type, _join(path, str(key)), problems)
        for key, item in value.items()
    }


def _expect(ok: bool, result: object, path: str, message: str, problems: list[str]) -> object:
    if ok:
        return result
    problems.append(f"{path}: {message}")
    return None


def _describe(annotation: object) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _label(path: str) -> str:
    return f"{path}: " if path else ""
