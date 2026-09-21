"""Opening a ContextMapArtifact directory: the manifest and the checks that need no parsing.

Both the reader and the validator start here. Nothing in this module writes: it only reads and
compares, and a directory that fails is reported with an explicit error, never repaired or
completed from anywhere else (there is no debug fallback, no registry and no second copy).
"""

from __future__ import annotations

import json
from pathlib import Path

from contextmap.artifact.errors import (
    ArtifactIntegrityError,
    IncompleteContextMapArtifactError,
    ManifestError,
    MissingPayloadError,
    UnsupportedArtifactSchemaError,
)
from contextmap.artifact.layout import BUNDLE_ARTIFACT_TYPE, CONTRACTUAL_FILES, MANIFEST
from contextmap.artifact.manifest import (
    ContextMapArtifactManifest,
    decode_manifest,
    manifest_content_identity,
)
from contextmap.artifact.versioning import (
    UnsupportedSchemaVersionError,
    require_supported_schema_version,
)
from contextmap.shared import check_file_inventory


def load_manifest(root: Path) -> ContextMapArtifactManifest:
    """Read and validate the manifest of an artifact directory.

    The checks run in an order that reports the most useful cause: a directory without a
    manifest is an incomplete write; an unsupported format or schema version is reported as such
    before anything else in the manifest is judged; only then is the recorded content identity
    compared with the one recomputed from the manifest, which detects an edited manifest.

    Args:
        root: The artifact directory.

    Returns:
        The manifest.

    Raises:
        IncompleteContextMapArtifactError: If ``root`` is not a directory or has no manifest.
        ManifestError: If the manifest is not valid JSON or is malformed.
        UnsupportedFormatVersionError: If the format version is not supported.
        UnsupportedArtifactSchemaError: If the schema version cannot be read by this code.
        ArtifactIntegrityError: If the manifest does not match its own content identity.
    """
    if not root.is_dir():
        raise IncompleteContextMapArtifactError(f"{root.name!r} is not a directory")
    path = root / MANIFEST
    if not path.is_file():
        raise IncompleteContextMapArtifactError(
            f"{root.name!r} has no {MANIFEST}: it is not a finished artifact "
            "(an interrupted write leaves none)"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ManifestError(f"{MANIFEST} of {root.name!r} is not valid JSON ({error})") from error
    if isinstance(record, dict) and record.get("artifact_type") == BUNDLE_ARTIFACT_TYPE:
        raise ManifestError(
            f"{root.name!r} is a bundle, not an artifact: open its 'artifact' directory"
        )
    manifest = decode_manifest(record)
    try:
        require_supported_schema_version(manifest.schema_version)
    except UnsupportedSchemaVersionError as error:
        raise UnsupportedArtifactSchemaError(str(error)) from error
    recomputed = manifest_content_identity(manifest)
    if recomputed != manifest.content_identity:
        raise ArtifactIntegrityError(
            f"{MANIFEST} of {root.name!r} does not match its content identity: "
            f"recorded {manifest.content_identity}, recomputed {recomputed}"
        )
    return manifest


def check_files_present(
    root: Path, manifest: ContextMapArtifactManifest, *, verify_hashes: bool
) -> None:
    """Check that every contractual file is inventoried, present and the size it should be.

    Sizes catch a missing, truncated or extended file with a ``stat`` each, so opening an
    artifact stays cheap. Hashes catch a change that keeps the size and read every byte, so they
    are opt-in.

    Args:
        root: The artifact directory.
        manifest: Its manifest.
        verify_hashes: Also hash every inventoried file.

    Raises:
        IncompleteContextMapArtifactError: If the manifest does not inventory a file every
            artifact must have.
        MissingPayloadError: If an inventoried file is not on disk.
        ArtifactIntegrityError: If a file has another size, or another hash when verified.
    """
    inventoried = {entry.path: entry for entry in manifest.file_inventory}
    for required in CONTRACTUAL_FILES:
        if required not in inventoried:
            raise IncompleteContextMapArtifactError(
                f"the {MANIFEST} does not inventory the required file {required!r}"
            )
    for entry in manifest.file_inventory:
        file_path = root / entry.path
        if not file_path.is_file():
            raise MissingPayloadError(f"missing file {entry.path!r} listed in the {MANIFEST}")
        size = file_path.stat().st_size
        if size != entry.size_bytes:
            raise ArtifactIntegrityError(
                f"{entry.path}: size {size} does not match the manifest ({entry.size_bytes}); "
                "the file was truncated or changed"
            )
        if verify_hashes:
            problems = check_file_inventory(root, [entry])
            if problems:
                raise ArtifactIntegrityError(f"{entry.path}: {problems[0]}")
