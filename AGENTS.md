# Agent Guide

This file defines the implementation rules for coding agents and contributors working on ContextMap2.

Read this file before creating or changing code, tests, contracts, artifacts, configuration, issues, or documentation.

The goal is not to maximize abstraction or code volume. The goal is to keep the research implementation simple, modular, testable, reproducible, and scientifically defensible.

## 1. Repository context

ContextMap2, in Solution 1, builds a minimal end-to-end path that transforms synchronized robotic observations into a persistent, versioned, and auditable `ContextMapArtifact`.

Current conceptual flow:

```text
recorded source
    -> ingestion
    -> canonical observations
    -> visual perception + state estimation
    -> geometric mapping
    -> 2D <-> 3D sensor association
    -> semantic fusion
    -> semantic mapping
    -> entity resolution
    -> spatial relations
    -> ContextMapArtifact
```

`point_representation` may participate as optional 3D evidence when there is a hypothesis and an evaluation that justify its use.

The public product of this repository is the map artifact. Viewers, natural-language search, navigation, planning, agents, and other consuming applications are outside the Solution 1 boundary and must consume the public artifact.

## 2. Source of truth

Before implementing, consult the source closest to the problem.

Recommended order:

1. `AGENTS.md`;
2. `docs/README.md`;
3. relevant global documentation;
4. capability documentation in `src/contextmap/<capability>/docs/`, when it exists;
5. the capability public API;
6. the real implementation;
7. existing tests.

Main global documents:

```text
docs/PIPELINE.md
docs/architecture.md
docs/CONTRACTS.md
docs/ARTIFACTS.md
docs/module-api.md
docs/shared-primitives.md
docs/runtime-composition.md
docs/development.md
```

Documentation may describe target architecture that is not implemented yet. Never infer that a capability, class, backend, or artifact exists only because it appears in documentation.

For implemented behavior, inspect the real code and tests. For boundaries, ownership, and intended dependency direction, consult the architecture documentation. If code and documentation diverge, treat the divergence explicitly within the scope of the change.

## 3. Main architecture rule

Organize the system by **capability**, not by framework, model, dataset, or technology.

Main Solution 1 capabilities:

```text
ingestion
visual_perception
state_estimation
geometric_mapping
sensor_association
point_representation
semantic_fusion
semantic_mapping
entity_resolution
spatial_relations
artifact
runtime
evaluation
```

Each capability owns the concepts it introduces semantically and exposes a small public API.

Ownership examples:

```text
visual_perception   -> Region2D, VisualFeature, SemanticClaim
state_estimation    -> PoseEstimate, Trajectory
geometric_mapping   -> GeometryPoint, GeometryReference, GeometricMap
sensor_association  -> SpatialObservation
semantic_fusion     -> FusionSupport, FusedEvidence
semantic_mapping    -> Entity
entity_resolution   -> ResolvedEntity, resolution decisions
spatial_relations   -> Relation, RelationEvidence
artifact            -> ContextMap, ContextMapArtifact
```

A consumer depends on the producer's public API, not on the producer's internal layout.

Allowed:

```python
from contextmap.visual_perception import SemanticClaim
```

Avoid in cross-capability code:

```python
from contextmap.visual_perception.models import SemanticClaim
from contextmap.visual_perception.backends.sam3 import SamNativeMask
```

Concrete backend imports belong in the composition root under `runtime`.

## 4. Separate evidence, belief, and knowledge

Do not collapse semantic levels prematurely.

### Evidence

Evidence describes something observed, measured, or produced by inference in a specific execution.

Examples:

```text
SourceObservation
RGB / LiDAR / depth
Region2D
mask
bounding box
VisualFeature
embedding
SemanticClaim
projection record
SpatialObservation
model response
score
pose/calibration used
```

Evidence must preserve identity, provenance, and acquisition context.

### Belief

Belief represents persistent state or a hypothesis accumulated from multiple pieces of evidence.

Examples:

```text
FusionSupport
FusedEvidence
Entity
ResolvedEntity
aggregated embedding
semantic state
accumulated confidence/uncertainty
```

A single observation must not automatically become persistent truth.

### Knowledge

Knowledge represents higher-level structure derived from entities and consolidated evidence.

Examples:

```text
Relation
RelationEvidence
containment
adjacency
spatial topology
ContextMap composition
```

Relations and inferences never replace the evidence that supports them.

## 5. Provenance and uncertainty are part of the domain

Whenever applicable, explicitly preserve:

```text
source observation ID
frame/timestamp
sensor ID
coordinate frame
units
pose and transform used
calibration identity
model/backend/checkpoint
policy/prompt version
source region/mask
embedding space
confidence or score with defined semantics
source artifact/run
code/config identity
```

Do not treat heterogeneous scores as equivalent probabilities.

Do not create a generic `shared.Confidence`. CLIP similarity, detector confidence, geometric quality, fusion weight, entity uncertainty, and relation support have different semantics.

Do not average, sum, or threshold these signals together without an explicit and testable semantic definition.

## 6. Open vocabulary by default

Labels are semantic hypotheses, not absolute identities of the world.

When useful for the contract, preserve:

```text
original embedding
queried text
proposed label
score
model/checkpoint
observation provenance
alternative hypotheses
```

Task-specific taxonomies may exist above the core, but they must not rigidify central contracts without a current requirement.

Segmentation does not imply classification. A `Region2D` may exist without a reliable label. Dense visual features do not have labels by themselves.

## 7. 2D to 3D association

A visual correspondence does not imply a correct spatial association.

Any `sensor_association` implementation must explicitly consider, as applicable:

```text
timestamps and temporal tolerances
intrinsics
extrinsics
camera model
distortion/fisheye
field of view
projection validity
visibility
occlusion
pose used
coordinate frames
units
```

Invalid, ambiguous, or non-visible associations must be represented or rejected explicitly. Do not hide uncertainty behind silent heuristics.

## 8. Artifacts are immutable boundaries

Each relevant stage produces or contributes to an auditable artifact.

Rules:

- a finalized artifact is immutable;
- re-execution creates a new identity;
- upstream artifact selection is explicit;
- `outputs/` contains contractual data;
- `debug/` contains human diagnostics and is never a downstream dependency;
- large payloads may be referenced lazily, with metadata and a hash;
- writes must be finalized atomically or with an equivalent guarantee;
- secrets never enter manifests, persisted configuration, or provenance.

Do not modify a previous artifact to "fix" a run. Generate another artifact/run with correct lineage.

Compatibility with historical schemas should only be implemented when there is an explicit requirement. During `v0.x`, do not keep wrappers, migration code, or fallbacks only to preserve old formats without a real consumer.

## 9. TDD is the default workflow

For deterministic behavior changes, use **Red -> Green -> Refactor**.

### Red

Before production code:

1. identify the observable behavior or invariant;
2. write the narrowest test that expresses that behavior;
3. run the test and confirm that it fails for the expected reason.

A test that already passes does not demonstrate the need for the change.

### Green

Implement the minimum amount of code required to make the test pass.

Do not use the task as an excuse to:

- create future abstractions;
- refactor unrelated modules;
- add hypothetical backends;
- generalize APIs beyond the requirement.

### Refactor

With tests green:

- improve names and structure;
- remove real semantic duplication;
- simplify flow;
- preserve boundaries and ownership;
- rerun affected tests.

### Legitimate exceptions

Do not create artificial tests for changes without executable behavior, such as a purely editorial correction.

Research or ML changes may not have a unit-level `Red` for model quality. In that case, before implementation define an executable evaluation with:

```text
hypothesis
baseline
changed variable
dataset/selection
metric
success or rejection criterion
```

Contracts, transforms, indexing, projection, serialization, lineage, and deterministic invariants remain subject to normal TDD.

## 10. Testing pyramid

Use the narrowest level that protects the required behavior.

```text
unit tests
    local deterministic behavior

contract tests
    invariants of interchangeable ports/backends

architecture tests
    boundaries, imports, ownership, and dependency direction

integration tests
    boundaries between capabilities and artifact read/write behavior

end-to-end tests
    representative minimal Solution 1 path

regression tests
    real failures observed previously

benchmarks/evaluation
    scientific quality, runtime, memory, robustness, and ablations
```

A fixed bug should receive a regression test whenever it is reproducible.

Mocks should replace external or expensive dependencies, not hide behavior that should be tested with real domain types.

## 11. Clean Code

Code should communicate intent without requiring excessive repository navigation.

Practical rules:

- names describe domain and responsibility;
- functions perform one coherent transformation;
- side effects are explicit;
- units, frames, and timestamps appear in the contract when relevant;
- errors have actionable meaning;
- dependencies are visible in signatures or construction;
- important scientific branches use explicit names and policies;
- relevant magic constants become configuration or named constants with clear semantics;
- comments primarily explain **why**, not restate the code.

Avoid generic names such as:

```text
Manager
Helper
Utils
Processor
Service
```

Use `Service` only when it actually represents a capability application service, such as `SensorAssociationService`.

## 12. Apply SOLID pragmatically

SOLID exists to preserve boundaries and substitutability, not to create layers.

### SRP

A function, class, or module should have a coherent responsibility and one primary reason to change.

### OCP

Create an extension point only when real variation exists, for example multiple `RegionDiscovery` or `StateEstimator` backends.

### LSP

Implementations of the same port must respect the same public invariants, including:

```text
units
coordinate frames
timestamp semantics
shape/dimensions
ordering
lifecycle
contractual errors
```

### ISP

Interfaces should be small and consumer-oriented. Do not force an implementation to provide unrelated operations.

### DIP

Scientific logic depends on domain ports and contracts. SDKs and concrete implementations are connected in `runtime`.

## 13. KISS

Prefer the smallest solution that correctly solves the current requirement and can be tested.

A clear concrete function is better than a generic framework without a demonstrated need.

An explicit factory is better than a service locator, global DI container, or plugin registry when only a few known backends exist.

## 14. YAGNI

Do not implement something only because it might be useful later.

Do not add without a current requirement:

```text
unused backend
schema field without a consumer
base class for a single implementation
factory for trivial construction without a variation point
global registry
plugin system
empty directory
compatibility layer
historical schema migration
silent fallback
generic ownerless configuration
distributed cache
remote storage
microservice
```

Documented target architecture is not authorization to materialize empty files or speculative APIs.

## 15. DRY

Remove duplication when it represents the **same domain rule**.

Do not abstract only because two blocks look syntactically similar.

Accidental duplication may be removed. Similarity with different semantics may and should remain separate.

Example:

```text
SemanticClaim confidence
RelationEvidence support
```

Both may be `float`, but they must not share an abstraction only for that reason.

Before extracting an abstraction, confirm:

1. the rule is semantically the same;
2. it has the same owner;
3. future changes will likely need to happen together;
4. the extraction reduces total complexity.

## 16. `shared` is not a convenience dump

Use `contextmap.shared` only for truly cross-cutting primitives with no clear semantic owner and exactly the same meaning across capabilities.

Good candidates, when real usage exists:

```text
Timestamp
Vector3
Quaternion
Transform3D
cross-cutting identifiers
minimal common provenance
```

Capability concepts remain with their owner even when many modules consume them.

Do not move something to `shared` only to remove a cycle. First verify ownership, dependency direction, the need for a port, and runtime responsibility.

## 17. Capability public API

The root `src/contextmap/<capability>/__init__.py` is the cross-capability public surface.

Export only required symbols and declare `__all__` explicitly.

Example:

```python
from .models import SemanticClaim
from .ports import SemanticInterpreter

__all__ = ["SemanticClaim", "SemanticInterpreter"]
```

Backends, helpers, SDK objects, native tensors, loaders, and backend-specific configuration remain internal.

Do not export everything for convenience.

## 18. Ports and adapters

Create a `Protocol`, port, or interface when at least one concrete reason exists:

- multiple real implementations;
- intentionally replaceable backend;
- an experiment compares implementations;
- an external dependency must be isolated;
- a test needs a lightweight fake for an expensive component;
- a cross-capability boundary needs stable behavior.

If there is one simple local implementation and no real variation point, use concrete code.

The same external framework may satisfy different contracts through different adapters. The public name should represent the capability, not the library.

## 19. Runtime and composition root

`runtime` composes and executes. Capabilities own scientific logic.

Runtime may:

```text
resolve configuration
select backends
construct implementations
validate the DAG
select upstream artifacts
coordinate lifecycle
reuse/recompute artifacts
record topology and effective configuration
expose a thin CLI
```

Runtime must not own:

```text
projection math
segmentation rules
semantic interpretation logic
fusion semantics
entity-resolution heuristics
spatial predicates
scientific thresholds owned by another capability
```

There is no implicit fallback to another backend when the selected backend fails.

## 20. Configuration, policy, backend, and schema

Keep these concepts distinct:

```text
schema
    format and semantics of the public contract

policy
    versioned scientific/algorithmic rule

backend
    replaceable implementation

configuration
    effective parameters of one execution
```

Do not use a configuration change to hide a semantic schema change.

Do not let one backend's configuration leak into another capability.

## 21. External integrations

ROS, datasets, model SDKs, Torch, JAX, remote providers, and specific libraries are adapter/backend details whenever a domain type is sufficient at the boundary.

ROS is a data source, not the domain.

After Ingestion, downstream code must not depend on ROS messages, rosbag, or TF as public representations.

Likewise, a consuming capability should not receive a `torch.Tensor`, SDK object, or provider-native payload when an explicit domain contract is sufficient.

## 22. Research and models

Do not choose a model before defining the problem and the evaluation.

For a relevant change in perception, association, fusion, or representation, record before implementation:

1. problem;
2. justification;
3. input;
4. proposed processing;
5. output;
6. metric;
7. test or evaluation.

When an architectural or scientific decision is inspired by literature, clearly distinguish:

```text
what the paper actually proposes
what we are adapting
what is original to ContextMap2
```

Do not turn a paper or model name into public architecture when the capability has a better domain name.

## 23. Verifiable scientific code

Do not accept an improvement only through visual inspection when objective comparison is possible.

Preserve for reproducibility:

```text
dataset/selection
manifest
split
model/checkpoint
effective configuration
seed when relevant
policy version
code commit
metrics
artifact IDs
```

Ablations should change one relevant variable at a time whenever possible.

Performance and semantic quality are different metrics. A faster backend is not automatically better for the map.

## 24. Code structure

Python code lives under:

```text
src/contextmap/
```

A capability should start with the smallest useful structure and grow only when needed.

Possible example, not a mandatory template:

```text
src/contextmap/visual_perception/
├── __init__.py
├── models.py
├── ports.py
├── service.py
├── backends/
└── docs/
```

Do not create `models.py`, `ports.py`, `service.py`, `backends/`, `_internal/`, or subpackages before concrete content exists for them.

Repository tests live under `tests/`, organized by responsibility. Architecture tests belong in `tests/architecture/`.

## 25. Language and documentation

Use:

```text
identifiers and APIs        English
Python docstrings           English, Google style
explanatory code comments   Brazilian Portuguese
project Markdown docs       Brazilian Portuguese
AGENTS.md                   English
commits and PR titles       English, Conventional Commits
```

Docstrings are required for public modules, classes, functions, and methods.

Private code also needs a docstring when it has a contract, invariant, side effect, exception, algorithm, or non-obvious decision.

Do not require a comment above every function. Comments must add context, not duplicate the signature or docstring.

## 26. Errors and validation

Validate invariants at the closest boundary that owns them.

Important examples:

```text
invalid timestamp
incompatible frame
missing/invalid transform
incompatible calibration
projection outside the camera model
invalid shape/dimensions
incompatible embedding space
wrong upstream artifact
incompatible schema
missing required provenance
impossible configuration
```

Fail early with a semantically useful error.

Do not catch broad exceptions only to keep a scientific pipeline running silently.

Retries, skips, partial outputs, and recoverable failures must be explicit when they are part of supported behavior.

## 27. Agent implementation workflow

For every coding task:

1. read the issue/requirement and acceptance criteria;
2. identify the capability that owns the behavior;
3. inspect the relevant code, public API, tests, and documentation;
4. trace data across modules before proposing a cross-capability change;
5. define inputs, outputs, invariants, units, frames, and provenance;
6. write the test or evaluation that demonstrates the need;
7. confirm `Red` when applicable;
8. implement the minimum required for `Green`;
9. refactor without expanding scope;
10. run focused tests;
11. run `make check` before considering the task complete;
12. update documentation in the same change when a contract, ownership rule, pipeline, or public behavior changes.

Do not change unrelated files only to "clean up" the repository.

## 28. Quality gates

The canonical local gate is:

```bash
make check
```

The project uses, at minimum:

```text
Ruff
mypy
pytest
pytest-cov
build
pre-commit
```

Python 3.11 or newer is required.

Do not disable lint, type checking, or test rules only to make CI pass. Fix the cause, or justify a local and minimal exception.

## 29. Git and repository changes

The standard flow is:

```text
issue branch
    -> milestone/<slug>
    -> dev
    -> main
```

Issue branches use:

```text
<type>/<issue-number>-<slug>
```

Commits follow Conventional Commits and include a descriptive body.

Agents must not create a branch, commit, push, PR, issue, or remote change unless the task explicitly requests that action or the automation context explicitly requires it.

Never push directly to `main` or `dev`.

Corrections requested during the review of a milestone PR must be committed
and pushed directly to that PR's `milestone/*` head branch. Do not open an
auxiliary issue branch or PR for those review corrections.

## 30. Definition of Done

A code change is complete only when, as applicable:

- the required behavior is implemented;
- the new test failed before implementation, or the TDD exception is justified by the type of change;
- relevant tests pass;
- a regression test exists for a reproducible bug;
- architecture boundaries remain valid;
- the public API remains minimal;
- units, frames, timestamps, and provenance are explicit when relevant;
- artifact/lineage remain auditable;
- there is no speculative abstraction;
- relevant semantic duplication was handled without premature generalization;
- documentation was updated when needed;
- `make check` passes.

## 31. Decision test before adding complexity

Before adding an interface, registry, factory, layer, global package, base class, cache, schema field, or new stage, answer:

1. what concrete problem exists today?
2. which capability owns the problem?
3. which real consumer needs this?
4. which variation point or dependency becomes explicit?
5. which test or evaluation demonstrates value?
6. can this be solved with a smaller structure?
7. does the change improve map quality, testability, reproducibility, or clarity?

If the answers are not concrete, do not add the complexity.

## 32. Final rule

When choosing between a more generic solution and a smaller, explicit, tested solution aligned with current ownership, prefer the latter.

ContextMap2 should grow from demonstrated problems, clear contracts, tests, and objective evaluation. Not from architectural anticipation.
