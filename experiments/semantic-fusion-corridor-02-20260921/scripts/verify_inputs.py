"""Compare the input runs and dataset files found locally with the bundle manifest.

Usage: verify_inputs.py [--input-root DIR] [--validation-dir DIR]

Only ``manifest.json`` files of the runs and two small dataset files are read; no payload is
loaded. A run is found by its ``run_id`` (not by directory name), so a workspace regenerated
with other run indexes still matches. The check is on the *contractual outputs*: the digest
covers ``outputs/`` only, because ``metrics/`` holds timings that change on every execution.

It compares the recorded inventories with the local ones. That the files of a run still match
its own inventory is ``verify_integrity()`` of the reader of that capability, not of this script.

Point Representation is the one input run regenerated on every execution of ``s04_fusion.py``
(the frozen upstream runs are read-only). Its ``PointRepresentationRef.provenance.code_version``
stamps the checkout revision, so its full outputs digest never matches across executions even
when the geometry is identical; that group is checked by its stable identity fields
(``representation_count``, ``representation_space_id``, ``dimension``) instead of a digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BUNDLE.parents[1]


def outputs_digest(file_inventory: list[dict[str, Any]]) -> tuple[str, int]:
    """Digest of the contractual outputs of a run, read from its manifest inventory.

    Args:
        file_inventory: The ``file_inventory`` list of a run ``manifest.json``.

    Returns:
        The SHA-256 (hex) of the text made of one ``<path>``, a tab, the ``<content_hash>`` and a
        newline per entry under ``outputs/``, sorted by path, and the number of entries.
        ``content_hash`` keeps its stored form (``sha256:<hex>``).
    """
    entries = sorted(
        (entry["path"], entry["content_hash"])
        for entry in file_inventory
        if entry["path"].startswith("outputs/")
    )
    text = "".join(f"{path}\t{content_hash}\n" for path, content_hash in entries)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(entries)


def sha256_file(path: Path) -> str:
    """SHA-256 (hex) of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(template: str, input_root: Path, validation_dir: Path) -> Path:
    return Path(
        template.replace("<VALIDATION_DIR>", str(validation_dir)).replace(
            "<REPO_ROOT>", str(input_root)
        )
    )


def _find_run(location: Path, run_id: str) -> dict[str, Any] | None:
    for directory in sorted(location.glob("run-*")):
        manifest = directory / "manifest.json"
        if manifest.is_file():
            record: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
            if record.get("run_id") == run_id:
                return record
    return None


def verify(input_root: Path, validation_dir: Path) -> list[str]:
    """Compare local inputs with the bundle manifest.

    Args:
        input_root: Root of the checkout holding ``datasets/`` and ``outputs/``.
        validation_dir: Directory holding the regenerated stage artifacts.

    Returns:
        One line per problem; empty when every recorded input matches.
    """
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    for relative, expected in manifest["dataset"]["inputs"].items():
        path = input_root / relative
        if not path.is_file():
            problems.append(f"missing dataset file: {relative}")
        elif sha256_file(path) != expected["sha256"]:
            problems.append(f"dataset file differs: {relative}")
    for kind, group in manifest["input_runs"].items():
        location = _resolve(group["location"], input_root, validation_dir)
        identity_only = kind == "point_representation"
        for expected in group["runs"]:
            local = _find_run(location, expected["run_id"])
            if local is None:
                problems.append(f"{kind}: run {expected['run_id']!r} not found under {location}")
                continue
            if identity_only:
                mismatched = [
                    key
                    for key in ("representation_count", "representation_space_id", "dimension")
                    if local.get(key) != expected.get(key)
                ]
                if mismatched:
                    problems.append(f"{kind}: run {expected['run_id']!r} differs in {mismatched}")
                continue
            digest, count = outputs_digest(local["file_inventory"])
            if (digest, count) != (expected["outputs_digest"], expected["outputs_files"]):
                problems.append(
                    f"{kind}: run {expected['run_id']!r} has other contractual outputs "
                    f"(digest {digest}, {count} files; recorded {expected['outputs_digest']}, "
                    f"{expected['outputs_files']} files)"
                )
    return problems


def main() -> int:
    """Run the check and print the result; the exit code is 1 when something differs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(os.environ.get("CONTEXTMAP_INPUT_ROOT", REPOSITORY_ROOT)),
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "CONTEXTMAP_VALIDATION_DIR",
                REPOSITORY_ROOT / "workspace/corridor-02/validation-semantic-fusion-20260921",
            )
        ),
    )
    args = parser.parse_args()
    problems = verify(args.input_root, args.validation_dir)
    for problem in problems:
        print(problem)
    print("all recorded inputs match" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
