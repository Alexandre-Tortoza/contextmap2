# Contributing

ContextMap2 is currently validating Solution 1. Contributions should favor reproducibility, measurable map quality, and small changes that can be evaluated independently.

## Branch flow

Use the following flow for normal development:

```text
feature / research branch
        ↓
       dev
        ↓
        qa
        ↓
       main
```

Recommended branch prefixes:

- `feat/`, new product behavior;
- `fix/`, defect correction;
- `research/`, implementation based on a research hypothesis;
- `experiment/`, temporary or comparative experiment;
- `refactor/`, structural change without intended behavioral change;
- `test/`, test or benchmark work;
- `docs/`, documentation only;
- `ci/`, pipeline or automation changes;
- `chore/`, repository maintenance.

## Pull requests

Pull requests should be focused and include:

- the problem or hypothesis;
- the implementation approach;
- how the change was validated;
- metrics or artifacts when the change affects map quality;
- known limitations or follow-up work.

Use semantic PR titles, for example:

```text
feat: add normalized frame contract
fix: preserve calibration provenance during serialization
research: evaluate temporal fusion strategy
experiment: compare region discovery backends
refactor: isolate artifact serialization
```

## Quality gates

Run the complete local check before requesting review:

```bash
make check
```

The CI verifies formatting, linting, static typing, and tests.

## Research changes

Research-oriented changes should avoid silently replacing a baseline. When testing a new method:

1. state the expected improvement;
2. preserve a reproducible baseline when practical;
3. define the metric that can confirm or reject the hypothesis;
4. record the configuration used for the run;
5. keep dataset-specific workarounds outside the general path unless their value is demonstrated more broadly.

## Dependencies

Avoid adding heavy runtime dependencies until they are required by a validated implementation path. Model-specific dependencies should remain isolated from domain contracts whenever possible.

## Releases

Do not create release tags for unreviewed experimental states. Version tags are created from validated commits according to [docs/versioning.md](docs/versioning.md).