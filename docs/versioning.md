# Versioning and releases

ContextMap2 uses Semantic Versioning tags in the form `vMAJOR.MINOR.PATCH`.

## Validation stage

While Solution 1 is still being validated, releases remain under major version zero:

```text
v0.1.0
v0.2.0
v0.2.1
```

Use:

- `PATCH` for compatible fixes, documentation corrections, and small internal improvements that do not intentionally change the artifact contract;
- `MINOR` for new capabilities, measurable pipeline changes, or artifact/schema evolution during the validation stage;
- `MAJOR` only after a stable external artifact contract exists and incompatible changes need to be communicated clearly.

## Release requirements

A release tag should point to a commit that:

- passes CI;
- has the relevant evaluation evidence recorded;
- has no known artifact corruption or serialization regression;
- documents incompatible schema changes;
- is reproducible from committed configuration and source code.

## Automation

Pushing a tag matching `v*.*.*` starts the release workflow. The workflow validates the tag shape, builds the Python distribution, uploads the build output as a workflow artifact, and creates a GitHub Release.

Releases in the `v0.x.y` series are marked as pre-releases automatically.

The repository does not publish to PyPI during the Solution 1 validation stage.