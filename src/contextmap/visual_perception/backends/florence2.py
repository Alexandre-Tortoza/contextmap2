"""Florence-2 adapter dedicated to the Region Discovery capability."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

from ..discovery import (
    AuditedRegions,
    BackendDiagnostics,
    DiscoveryInput,
    DiscoveryOutput,
    DiscoveryPassConfig,
    discover_canonical_regions,
    validate_materialized_discovery_image,
)
from ..models import BackendProvenance, PreparedImage, Region2D
from ..normalization import NormalizationConfig
from ..region_models import (
    BackendScore,
    BoundingBox,
    InlineMask,
    JsonScalar,
    RegionCandidate,
    RegionProvenance,
)
from ._model_placement import verify_model_placement

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_REGION_TASKS = frozenset(
    {
        "<REGION_PROPOSAL>",
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "<REGION_TO_SEGMENTATION>",
        "<OPEN_VOCABULARY_DETECTION>",
    }
)


@dataclass(frozen=True, slots=True)
class Florence2Config:
    """Effective configuration for Florence-2 region discovery only.

    ``model_version`` has no default: it enters provenance and the digest, so a run never
    records a placeholder instead of the checkpoint version that produced it.
    """

    checkpoint: str
    task: str
    model_version: str
    device: str = "cpu"
    precision: str = "float32"
    prompt: str | None = None
    box_threshold: float = 0.0
    generation_settings: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate model and selected region-producing task."""
        if not self.checkpoint or not self.model_version or not self.device or not self.precision:
            raise ValueError(
                "Florence-2 checkpoint, version, device, and precision must not be empty"
            )
        if not self.task:
            raise ValueError("Florence-2 region discovery task must not be empty")
        if not isfinite(self.box_threshold) or not 0 <= self.box_threshold <= 1:
            raise ValueError("Florence-2 box_threshold must be between zero and one")
        if self.task not in _REGION_TASKS:
            raise ValueError("Florence-2 task must be a supported region-producing task")
        keys = [key for key, _ in self.generation_settings]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Florence-2 generation setting names must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the effective Florence-2 configuration."""
        payload = {
            "checkpoint": self.checkpoint,
            "task": self.task,
            "model_version": self.model_version,
            "device": self.device,
            "precision": self.precision,
            "prompt": self.prompt,
            "box_threshold": self.box_threshold,
            "generation_settings": list(self.generation_settings),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Florence2NativeRegion:
    """Parsed Florence-2 geometry kept inside the infrastructure adapter."""

    proposal_id: str
    box: tuple[float, float, float, float]
    score: float | None
    mask: InlineMask | None = None
    parsed_text: str | None = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate parser output before canonical normalization."""
        if not self.proposal_id:
            raise ValueError("Florence-2 proposal_id must not be empty")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("Florence-2 native score must be finite when present")


@dataclass(frozen=True, slots=True)
class Florence2NativeOutput:
    """Structured output of the Florence-2 region-task parser."""

    regions: tuple[Florence2NativeRegion, ...]
    warnings: tuple[str, ...] = ()
    parsing_diagnostics: tuple[tuple[str, JsonScalar], ...] = ()


class Florence2Runtime(Protocol):
    """Internal seam around Florence model execution and native parsing."""

    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        """Execute the configured Florence region task and parse its output."""
        ...


class _ModelInputs(Protocol):
    """Device-transfer surface of Transformers model inputs."""

    def to(self, device: str, dtype: object) -> Mapping[str, object]:
        """Move tensors to the configured device, casting floating ones to ``dtype``."""
        ...


class _Florence2Model(Protocol):
    """Minimum Transformers Florence-2 generation surface."""

    def parameters(self) -> Iterator[object]:
        """Yield the loaded weights, whose device and dtype the runtime verifies."""
        ...

    def generate(self, **kwargs: object) -> object:
        """Generate native output token ids."""
        ...


class _Florence2Processor(Protocol):
    """Minimum official Florence-2 processor surface."""

    def __call__(self, *, text: str, images: object, return_tensors: str) -> _ModelInputs:
        """Create model inputs for an image and task prompt."""
        ...

    def batch_decode(self, sequences: object, *, skip_special_tokens: bool) -> Sequence[str]:
        """Decode generated tokens while retaining geometry tokens."""
        ...

    def post_process_generation(
        self, text: str, *, task: str, image_size: tuple[int, int]
    ) -> Mapping[str, object]:
        """Parse generated text with the official task parser."""
        ...


class TransformersFlorence2Runtime:
    """Execute the official Transformers Florence-2 inference flow."""

    def __init__(
        self,
        *,
        model: _Florence2Model,
        processor: _Florence2Processor,
        image_loader: Callable[[DiscoveryInput], object],
    ) -> None:
        """Bind loaded model components to a pass-aware image loader."""
        self._model = model
        self._processor = processor
        self._image_loader = image_loader

    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        """Generate and parse one configured Florence-2 region task.

        Raises:
            ValueError: If the model's parameters are not on ``config.device`` or not in
                ``config.precision``, if the generation settings duplicate a model input,
                or if the parser output is not structured region output.
        """
        model_dtype = verify_model_placement(
            self._model, device=config.device, precision=config.precision, backend="Florence-2"
        )
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        task_prompt = f"{config.task}{config.prompt or ''}"
        image = self._image_loader(discovery_input)
        validate_materialized_discovery_image(image, discovery_input)
        # Os pixel_values saem do processor em float32: seguem o dtype verificado do modelo.
        inputs = self._processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        ).to(config.device, model_dtype)
        generation_settings = dict(config.generation_settings)
        conflicts = set(inputs).intersection(generation_settings)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"Florence-2 generation settings duplicate model inputs: {names}")
        with _torch_inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_settings)
        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=False)
        if len(decoded) != 1:
            raise ValueError("Florence-2 runtime expects exactly one decoded result per image")
        parsed = self._processor.post_process_generation(
            decoded[0],
            task=config.task,
            image_size=(width, height),
        )
        task_result = parsed.get(config.task)
        if not isinstance(task_result, Mapping):
            raise ValueError("Florence-2 task parser did not return structured region output")
        return _parse_task_result(task_result, width=width, height=height)


class Florence2RegionDiscovery:
    """Adapt Florence-2 region tasks without producing semantic claims."""

    def __init__(
        self,
        *,
        config: Florence2Config,
        runtime: Florence2Runtime,
        pass_config: DiscoveryPassConfig | None = None,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        """Build the region-only adapter with explicit model runtime."""
        self._config = config
        self._runtime = runtime
        self._pass_config = pass_config
        self._normalization_config = normalization_config

    def backend_provenance(self) -> BackendProvenance:
        """Return Florence-2 identity in the Visual Perception Core contract."""
        return BackendProvenance(
            backend_id="florence2_region_discovery",
            capability="region_discovery",
            provider="microsoft",
            model=self._config.checkpoint,
            version=self._config.model_version,
            configuration_fingerprint=self._config.digest,
        )

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        """Discover and normalize canonical regions through the public port."""
        return self.discover_audited(image).regions

    def discover_audited(self, image: PreparedImage) -> AuditedRegions:
        """Discover canonical regions together with the audit of every decision behind them."""
        return discover_canonical_regions(
            image,
            self,
            pass_config=self._pass_config,
            normalization_config=self._normalization_config,
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Run one Florence-2 region task and normalize parsed geometry."""
        started = perf_counter()
        native_output = self._runtime.predict(discovery_input, self._config)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        accepted = tuple(
            region
            for region in native_output.regions
            if region.score is None or region.score >= self._config.box_threshold
        )
        candidates = tuple(
            self._normalize(region, discovery_input, width, height) for region in accepted
        )
        duration_ms = (perf_counter() - started) * 1000
        metadata = (
            *native_output.parsing_diagnostics,
            ("raw_proposal_count", len(native_output.regions)),
            ("filtered_proposal_count", len(native_output.regions) - len(candidates)),
            ("checkpoint", self._config.checkpoint),
            ("model_version", self._config.model_version),
            ("device", self._config.device),
            ("precision", self._config.precision),
            ("task", self._config.task),
            ("prompt", self._config.prompt),
            ("config_digest", self._config.digest),
        )
        return DiscoveryOutput(
            candidates=candidates,
            diagnostics=BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=len(candidates),
                warnings=native_output.warnings,
                metadata=metadata,
            ),
        )

    def _normalize(
        self,
        region: Florence2NativeRegion,
        discovery_input: DiscoveryInput,
        width: int,
        height: int,
    ) -> RegionCandidate:
        """Convert one parsed Florence region without semantic promotion."""
        mask = region.mask
        if mask is not None and (mask.width, mask.height) != (width, height):
            raise ValueError(
                f"Florence-2 proposal {region.proposal_id} mask dimensions must match "
                "discovery pass dimensions"
            )

        score = None
        if region.score is not None:
            score = BackendScore(
                name="region_score",
                value=region.score,
                semantics=(
                    "Florence-2 task-native region score; not calibrated across discovery backends"
                ),
            )

        native_metadata = region.metadata
        if region.parsed_text is not None:
            native_metadata = (*native_metadata, ("parsed_text", region.parsed_text))
        query = self._config.task
        if self._config.prompt:
            query = f"{query}:{self._config.prompt}"
        return RegionCandidate(
            candidate_id=region.proposal_id,
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=BoundingBox(
                x_min=region.box[0],
                y_min=region.box[1],
                x_max=region.box[2],
                y_max=region.box[3],
            ),
            mask=mask,
            score=score,
            provenance=RegionProvenance(
                backend_id="florence2_region_discovery",
                backend_version=self._config.model_version,
                checkpoint=self._config.checkpoint,
                config_digest=self._config.digest,
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id=region.proposal_id,
                query=query,
            ),
            native_metadata=native_metadata,
        )


def _torch_inference_mode() -> AbstractContextManager[object]:
    """Return ``torch.inference_mode()``, so ``generate`` never records autograd state."""
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Florence-2 inference requires torch inference_mode; install torch"
        ) from error
    return cast(AbstractContextManager[object], torch.inference_mode())


def _parse_task_result(
    result: Mapping[str, object], *, width: int, height: int
) -> Florence2NativeOutput:
    """Convert official parsed boxes and polygons into detached region values."""
    regions: list[Florence2NativeRegion] = []
    boxes = _optional_sequence(result.get("bboxes"), "bboxes")
    box_labels = _labels(result, preferred="bboxes_labels", fallback="labels") if boxes else ()
    scores = _optional_sequence(result.get("scores"), "scores")
    _validate_optional_parallel(box_labels, len(boxes), "box labels")
    _validate_optional_parallel(scores, len(boxes), "box scores")

    for index, box_value in enumerate(boxes):
        box = _numeric_sequence(box_value, "bbox", length=4)
        regions.append(
            Florence2NativeRegion(
                proposal_id=f"florence2-box-{index:06d}",
                box=cast(tuple[float, float, float, float], box),
                score=None if not scores else _finite_number(scores[index], "score"),
                parsed_text=None if not box_labels else _text(box_labels[index], "box label"),
                metadata=(("geometry_source", "official_post_process_bbox"),),
            )
        )

    polygons = _optional_sequence(result.get("polygons"), "polygons")
    polygon_labels = (
        _labels(result, preferred="polygons_labels", fallback="labels") if polygons else ()
    )
    _validate_optional_parallel(polygon_labels, len(polygons), "polygon labels")
    for index, polygon_value in enumerate(polygons):
        mask, box = _rasterize_polygons(polygon_value, width=width, height=height)
        regions.append(
            Florence2NativeRegion(
                proposal_id=f"florence2-polygon-{index:06d}",
                box=box,
                score=None,
                mask=mask,
                parsed_text=(
                    None if not polygon_labels else _text(polygon_labels[index], "polygon label")
                ),
                metadata=(("geometry_source", "official_post_process_polygon"),),
            )
        )

    # Zero regiões é resultado legítimo de Region Discovery (#380), como no SAM2/SAM3: um frame
    # sem nada detectável sai vazio e auditável pelos contadores, não como falha do estágio.
    return Florence2NativeOutput(
        regions=tuple(regions),
        parsing_diagnostics=(
            ("box_count", len(boxes)),
            ("polygon_count", len(polygons)),
        ),
    )


def _rasterize_polygons(
    value: object, *, width: int, height: int
) -> tuple[InlineMask, tuple[float, float, float, float]]:
    """Rasterize Florence polygon coordinates at pixel centers without SDK objects."""
    import numpy as np

    raw = _required_sequence(value, "polygon region")
    if raw and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw):
        contour_values: Sequence[object] = (raw,)
    else:
        contour_values = raw

    contours: list[tuple[tuple[float, float], ...]] = []
    for contour_value in contour_values:
        coordinates = _required_sequence(contour_value, "polygon contour")
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError("Florence-2 polygon contours require at least three coordinate pairs")
        numbers = tuple(_finite_number(item, "polygon coordinate") for item in coordinates)
        contours.append(tuple(zip(numbers[::2], numbers[1::2], strict=True)))

    if not contours:
        raise ValueError("Florence-2 polygon region must contain at least one contour")
    x_values = [point[0] for contour in contours for point in contour]
    y_values = [point[1] for contour in contours for point in contour]
    box = (min(x_values), min(y_values), max(x_values), max(y_values))
    mask = np.zeros((height, width), dtype=np.bool_)
    for contour in contours:
        mask |= _contour_interior(contour, width=width, height=height)
    return InlineMask(mask), box


def _contour_interior(
    contour: tuple[tuple[float, float], ...], *, width: int, height: int
) -> NDArray[np.bool_]:
    """Return the pixels whose centre lies inside one contour, by the even-odd rule.

    A horizontal ray from each pixel centre crosses an edge when the edge spans the centre's row;
    the crossing abscissa depends on the row only, so each edge flips, in the rows it spans, every
    pixel left of it. The arithmetic is the per-pixel test's, in float64, so the result is too.
    """
    import numpy as np

    centres_x = np.arange(width) + 0.5
    centres_y = np.arange(height) + 0.5
    inside = np.zeros((height, width), dtype=np.bool_)
    previous_x, previous_y = contour[-1]
    for current_x, current_y in contour:
        rows = np.flatnonzero((current_y > centres_y) != (previous_y > centres_y))
        if rows.size:
            boundary_x = (previous_x - current_x) * (centres_y[rows] - current_y) / (
                previous_y - current_y
            ) + current_x
            inside[rows] ^= centres_x[None, :] < boundary_x[:, None]
        previous_x, previous_y = current_x, current_y
    return inside


def _labels(result: Mapping[str, object], *, preferred: str, fallback: str) -> Sequence[object]:
    value = result.get(preferred)
    if value is None:
        value = result.get(fallback)
    return _optional_sequence(value, preferred)


def _validate_optional_parallel(values: Sequence[object], count: int, name: str) -> None:
    if values and len(values) != count:
        raise ValueError(f"Florence-2 {name} must match its geometry count")


def _optional_sequence(value: object, name: str) -> Sequence[object]:
    if value is None:
        return ()
    return _required_sequence(value, name)


def _required_sequence(value: object, name: str) -> Sequence[object]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        raise TypeError(f"Florence-2 {name} must be a sequence")
    return cast(Sequence[object], native)


def _numeric_sequence(value: object, name: str, *, length: int) -> tuple[float, ...]:
    native = _required_sequence(value, name)
    if len(native) != length:
        raise ValueError(f"Florence-2 {name} must contain {length} numbers")
    return tuple(_finite_number(item, name) for item in native)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Florence-2 {name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"Florence-2 {name} must be finite")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Florence-2 {name} must be text")
    return value
