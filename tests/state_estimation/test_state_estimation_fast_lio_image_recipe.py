"""Static checks of the FAST-LIO image recipe (``docker/fast-lio-ros1/Dockerfile``).

The recipe is only as reproducible as what it pins. These tests parse the Dockerfile
(no docker, no network) and fail on anything a rebuild could resolve differently
tomorrow: a base image chosen by a moving tag, an APT package without an exact version,
or a source ref that is not a full commit hash. They cannot prove a rebuild is
identical (the upstream archives must still serve the pinned versions); that is what the
validation records, next to the digests actually used.
"""

import re
import shlex
from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "fast-lio-ros1" / "Dockerfile"

_DIGEST_BASE = re.compile(r"\S+@sha256:[0-9a-f]{64}")
_PINNED_PACKAGE = re.compile(r"[a-z0-9][a-z0-9+.\-]*(?::[a-z0-9]+)?=[^\s=*?]+")
_APT_INSTALL = re.compile(r"\bapt(?:-get)?\s+(?:-\S+\s+)*install\s+([^&;|]+)")
_SOURCE_REF = re.compile(r"ARG\s+(\w+_REF)=(\S*)")
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")

# A receita anterior da PR #419: base por tag móvel e pacotes APT sem versão.
_PREVIOUS_RECIPE = """\
FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]
RUN apt-get update \\
    && apt-get install -y --no-install-recommends build-essential cmake git libapr1-dev \\
    && rm -rf /var/lib/apt/lists/*
"""


def _instructions(dockerfile: str) -> list[str]:
    """Return the logical instructions: comments dropped, ``\\`` continuations joined."""
    logical: list[str] = []
    pending = ""
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        logical.append(pending + line)
        pending = ""
    if pending:
        logical.append(pending.strip())
    return logical


def _unpinned_bases(dockerfile: str) -> list[str]:
    """Return the ``FROM`` base images that are not pinned by a sha256 digest."""
    bases = []
    for instruction in _instructions(dockerfile):
        parts = instruction.split()
        if parts[0].upper() != "FROM":
            continue
        image = next(part for part in parts[1:] if not part.startswith("--"))
        if not _DIGEST_BASE.fullmatch(image):
            bases.append(image)
    return bases


def _apt_packages(dockerfile: str) -> list[str]:
    """Return every package named by an ``apt-get install`` in the Dockerfile."""
    packages: list[str] = []
    for instruction in _instructions(dockerfile):
        for match in _APT_INSTALL.finditer(instruction):
            packages += [token for token in shlex.split(match[1]) if not token.startswith("-")]
    return packages


def _unpinned_apt_packages(dockerfile: str) -> list[str]:
    """Return the APT packages installed without an exact ``name=version``."""
    return [
        package for package in _apt_packages(dockerfile) if not _PINNED_PACKAGE.fullmatch(package)
    ]


def _unpinned_source_refs(dockerfile: str) -> list[str]:
    """Return the ``ARG *_REF`` names whose default is not a full commit hash."""
    return [
        name
        for instruction in _instructions(dockerfile)
        for name, value in _SOURCE_REF.findall(instruction)
        if not _COMMIT_HASH.fullmatch(value)
    ]


def test_the_recipe_pins_its_base_image_by_digest() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert any(i.split()[0].upper() == "FROM" for i in _instructions(dockerfile))
    assert _unpinned_bases(dockerfile) == []


def test_the_recipe_pins_every_apt_package_it_installs_to_an_exact_version() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    # Sem pacotes o teste seria vazio: a receita instala ao menos o git.
    assert _apt_packages(dockerfile)
    assert _unpinned_apt_packages(dockerfile) == []


def test_the_recipe_pins_every_source_ref_by_commit_hash() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert len(_SOURCE_REF.findall(dockerfile)) == 3
    assert _unpinned_source_refs(dockerfile) == []


def test_a_base_image_chosen_by_a_moving_tag_is_reported_as_unpinned() -> None:
    assert _unpinned_bases(_PREVIOUS_RECIPE) == ["osrf/ros:noetic-desktop-full"]
    # O digest fixa a base mesmo quando a tag continua ao lado, só para leitura.
    pinned = "FROM osrf/ros:noetic-desktop-full@sha256:" + "a" * 64 + "\n"
    assert _unpinned_bases(pinned) == []
    assert _unpinned_bases("FROM osrf/ros@sha256:" + "a" * 63) == ["osrf/ros@sha256:" + "a" * 63]


def test_an_apt_package_without_an_exact_version_is_reported_as_unpinned() -> None:
    assert _unpinned_apt_packages(_PREVIOUS_RECIPE) == [
        "build-essential",
        "cmake",
        "git",
        "libapr1-dev",
    ]
    partly_pinned = (
        "RUN apt-get install -y git=1:2.25.1-1ubuntu3.14 \\\n    cmake=3.16* \\\n    libapr1-dev\n"
    )
    assert _unpinned_apt_packages(partly_pinned) == ["cmake=3.16*", "libapr1-dev"]


def test_a_source_ref_that_is_a_branch_is_reported_as_unpinned() -> None:
    assert _unpinned_source_refs("ARG FAST_LIO_REF=main\nARG OTHER_REF=" + "a" * 40) == [
        "FAST_LIO_REF"
    ]
