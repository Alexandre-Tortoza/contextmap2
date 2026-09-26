"""SAM2 adapters: automatic Region Discovery and prompted grounding-to-mask refinement.

Both adapters share one SAM2 model runtime family: the automatic path wraps the official
``SAM2AutomaticMaskGenerator``, the prompted path wraps the official
``SAM2ImagePredictor`` built from the same, already loaded, model. The model loader itself
belongs to the runtime provider; no path here loads a second SAM2.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module
from math import isfinite
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol, cast

from ..discovery import (
    BackendDiagnostics,
    DiscoveryInput,
    DiscoveryOutput,
    DiscoveryPassConfig,
    discover_canonical_regions,
    validate_materialized_discovery_image,
)
from ..grounding import GroundingGeometry
from ..models import BackendProvenance, PreparedImage, Region2D
from ..normalization import NormalizationConfig
from ..refinement import (
    REFINEMENT_CAPABILITY,
    RefinementDiagnostics,
    RefinementPrompt,
    RefinementRequestError,
    RegionRefinementCapabilities,
    RegionRefinementExecution,
    RegionRefinementRequest,
    refinement_outcome,
    validate_refinement_request,
)
from ..region_models import (
    BackendScore,
    BoundingBox,
    InlineMask,
    JsonScalar,
    RegionCandidate,
    RegionProvenance,
)


@dataclass(frozen=True, slots=True)
class Sam2Config:
    """Effective configuration for one SAM2 Region Discovery adapter."""

    checkpoint: str
    model_version: str = "unknown"
    device: str = "cpu"
    precision: str = "float32"
    predicted_iou_threshold: float = 0.0
    stability_threshold: float = 0.0
    automatic_mask_settings: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate explicit backend selection and native thresholds."""
        if not self.checkpoint or not self.model_version or not self.device or not self.precision:
            raise ValueError("SAM2 checkpoint, version, device, and precision must not be empty")
        _validate_unit_threshold(self.predicted_iou_threshold, "predicted_iou_threshold")
        _validate_unit_threshold(self.stability_threshold, "stability_threshold")
        keys = [key for key, _ in self.automatic_mask_settings]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("SAM2 automatic mask setting names must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the effective SAM2 configuration."""
        payload = {
            "checkpoint": self.checkpoint,
            "model_version": self.model_version,
            "device": self.device,
            "precision": self.precision,
            "predicted_iou_threshold": self.predicted_iou_threshold,
            "stability_threshold": self.stability_threshold,
            "automatic_mask_settings": list(self.automatic_mask_settings),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Sam2NativeProposal:
    """SDK-isolated SAM2 proposal normalized to Python scalar containers."""

    proposal_id: str
    box: tuple[float, float, float, float]
    mask: tuple[bool, ...]
    predicted_iou: float
    stability_score: float
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate native scalar output before canonical normalization."""
        if not self.proposal_id:
            raise ValueError("SAM2 proposal_id must not be empty")
        if not isfinite(self.predicted_iou) or not isfinite(self.stability_score):
            raise ValueError("SAM2 native scores must be finite")


class Sam2Runtime(Protocol):
    """Internal seam around a loaded SAM2 model/runtime."""

    def predict(
        self, discovery_input: DiscoveryInput, config: Sam2Config
    ) -> tuple[Sam2NativeProposal, ...]:
        """Run SAM2 and return scalar-only native proposals."""
        ...


class _AutomaticMaskGenerator(Protocol):
    """Minimum official SAM2 automatic-mask generator surface used here."""

    def generate(self, image: object) -> list[dict[str, object]]:
        """Return official automatic-mask records for one image."""
        ...


class Sam2AutomaticMaskRuntime:
    """Execute and isolate the official SAM2 automatic-mask generator API."""

    def __init__(
        self,
        *,
        mask_generator: _AutomaticMaskGenerator,
        image_loader: Callable[[DiscoveryInput], object],
        config_digest: str,
    ) -> None:
        """Bind a configured generator to an explicit prepared-image loader."""
        if not config_digest:
            raise ValueError("SAM2 runtime configuration digest must not be empty")
        self._mask_generator = mask_generator
        self._image_loader = image_loader
        self._config_digest = config_digest

    @classmethod
    def from_model(
        cls,
        *,
        model: object,
        image_loader: Callable[[DiscoveryInput], object],
        config: Sam2Config,
    ) -> Sam2AutomaticMaskRuntime:
        """Construct the official generator lazily from a loaded SAM2 model.

        The optional dependency remains inside this infrastructure module. The
        supplied loader must materialize the configured discovery pass as an HWC
        uint8 image accepted by SAM2.
        """
        settings = dict(config.automatic_mask_settings)
        reserved = {"model", "pred_iou_thresh", "stability_score_thresh", "output_mode"}
        conflicts = reserved.intersection(settings)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"SAM2 automatic mask settings duplicate owned fields: {names}")
        module = import_module("sam2.automatic_mask_generator")
        generator_type = module.SAM2AutomaticMaskGenerator
        generator = generator_type(
            model=model,
            pred_iou_thresh=config.predicted_iou_threshold,
            stability_score_thresh=config.stability_threshold,
            output_mode="binary_mask",
            **settings,
        )
        return cls(
            mask_generator=generator,
            image_loader=image_loader,
            config_digest=config.digest,
        )

    def predict(
        self, discovery_input: DiscoveryInput, config: Sam2Config
    ) -> tuple[Sam2NativeProposal, ...]:
        """Run official SAM2 inference and detach every SDK-native value."""
        if config.digest != self._config_digest:
            raise ValueError("SAM2 runtime configuration digest does not match the active config")
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        image = self._image_loader(discovery_input)
        validate_materialized_discovery_image(image, discovery_input)
        records = self._mask_generator.generate(image)
        return tuple(
            _parse_automatic_mask_record(record, index=index, width=width, height=height)
            for index, record in enumerate(records)
        )


class Sam2RegionDiscovery:
    """Adapt SAM2 automatic-mask output to canonical RegionCandidate values."""

    def __init__(
        self,
        *,
        config: Sam2Config,
        runtime: Sam2Runtime,
        pass_config: DiscoveryPassConfig | None = None,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        """Build the adapter with explicit effective configuration and runtime."""
        self._config = config
        self._runtime = runtime
        self._pass_config = pass_config
        self._normalization_config = normalization_config

    def backend_provenance(self) -> BackendProvenance:
        """Return SAM2 identity in the Visual Perception Core contract."""
        return BackendProvenance(
            backend_id="sam2",
            capability="region_discovery",
            provider="facebook",
            model=self._config.checkpoint,
            version=self._config.model_version,
            configuration_fingerprint=self._config.digest,
        )

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        """Discover and normalize canonical regions through the public port."""
        return discover_canonical_regions(
            image,
            self,
            pass_config=self._pass_config,
            normalization_config=self._normalization_config,
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Run SAM2 for one pass and normalize accepted native proposals."""
        started = perf_counter()
        native_proposals = self._runtime.predict(discovery_input, self._config)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        candidates = tuple(
            self._normalize(proposal, discovery_input, width, height)
            for proposal in native_proposals
            if proposal.predicted_iou >= self._config.predicted_iou_threshold
            and proposal.stability_score >= self._config.stability_threshold
        )
        duration_ms = (perf_counter() - started) * 1000
        return DiscoveryOutput(
            candidates=candidates,
            diagnostics=BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=len(candidates),
                warnings=(),
                metadata=(
                    ("raw_proposal_count", len(native_proposals)),
                    ("filtered_proposal_count", len(native_proposals) - len(candidates)),
                    ("checkpoint", self._config.checkpoint),
                    ("model_version", self._config.model_version),
                    ("device", self._config.device),
                    ("precision", self._config.precision),
                    ("config_digest", self._config.digest),
                ),
            ),
        )

    def _normalize(
        self,
        proposal: Sam2NativeProposal,
        discovery_input: DiscoveryInput,
        width: int,
        height: int,
    ) -> RegionCandidate:
        """Convert one scalar SAM2 proposal without leaking native model objects."""
        if len(proposal.mask) != width * height:
            raise ValueError("SAM2 proposal mask length must match discovery pass dimensions")
        return RegionCandidate(
            candidate_id=proposal.proposal_id,
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=BoundingBox(
                x_min=proposal.box[0],
                y_min=proposal.box[1],
                x_max=proposal.box[2],
                y_max=proposal.box[3],
            ),
            mask=InlineMask(width=width, height=height, data=proposal.mask),
            score=BackendScore(
                name="predicted_iou",
                value=proposal.predicted_iou,
                semantics=(
                    "SAM2-native predicted mask IoU; not calibrated across discovery backends"
                ),
            ),
            provenance=RegionProvenance(
                backend_id="sam2",
                backend_version=self._config.model_version,
                checkpoint=self._config.checkpoint,
                config_digest=self._config.digest,
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id=proposal.proposal_id,
            ),
            native_metadata=(
                ("stability_score", proposal.stability_score),
                *proposal.metadata,
            ),
        )


_REFINEMENT_PRECISIONS = frozenset({"float32", "float16", "bfloat16"})
_PREDICTED_IOU_SEMANTICS = (
    "SAM2-native predicted IoU of the mask for its prompt; a model self-estimate, not a "
    "calibrated probability that the mask is correct"
)


@dataclass(frozen=True, slots=True)
class Sam2RefinementConfig:
    """Effective configuration of SAM2 prompted refinement.

    Attributes:
        checkpoint: SAM2 checkpoint identity.
        model_version: SAM2 model/package version.
        device: Device the provider runs the model on.
        precision: ``float32`` runs as loaded; ``float16``/``bfloat16`` run under
            ``torch.autocast``, as the official examples do.
        mask_threshold: Logit threshold of the official ``SAM2ImagePredictor``.
        max_hole_area: Official post-processing: fill holes up to this area (0 disables).
        max_sprinkle_area: Official post-processing: remove specks up to this area.
    """

    checkpoint: str
    model_version: str = "unknown"
    device: str = "cpu"
    precision: str = "float32"
    mask_threshold: float = 0.0
    max_hole_area: float = 0.0
    max_sprinkle_area: float = 0.0

    def __post_init__(self) -> None:
        """Validate identity and predictor settings before any model is requested."""
        if not self.checkpoint or not self.model_version or not self.device:
            raise ValueError("SAM2 refinement checkpoint, version and device must not be empty")
        if self.precision not in _REFINEMENT_PRECISIONS:
            raise ValueError(
                f"SAM2 refinement precision must be one of {sorted(_REFINEMENT_PRECISIONS)}"
            )
        if not isfinite(self.mask_threshold):
            raise ValueError("SAM2 refinement mask_threshold must be finite")
        for name, value in (
            ("max_hole_area", self.max_hole_area),
            ("max_sprinkle_area", self.max_sprinkle_area),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"SAM2 refinement {name} must be finite and non-negative")

    @property
    def digest(self) -> str:
        """Return the deterministic identity of this refiner configuration."""
        payload = {"task": "prompt_refinement", **asdict(self)}
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Sam2PromptedMask:
    """SDK-isolated SAM2 answer to one prompt: a full-image mask and its predicted IoU."""

    mask: tuple[bool, ...]
    predicted_iou: float

    def __post_init__(self) -> None:
        """Reject a non-finite native score."""
        if not isfinite(self.predicted_iou):
            raise ValueError("SAM2 predicted IoU must be finite")


class Sam2PromptRuntime(Protocol):
    """Internal seam around a loaded SAM2 model answering box and point prompts.

    It receives the exact, already hash-verified image bytes, so a provider-supplied
    runtime never needs to know where the run keeps its prepared images.
    """

    def predict_prompts(
        self,
        *,
        image: bytes,
        width: int,
        height: int,
        prompts: tuple[RefinementPrompt, ...],
        config: Sam2RefinementConfig,
    ) -> tuple[Sam2PromptedMask, ...]:
        """Return one row-major ``width x height`` mask per prompt, in prompt order."""
        ...


class _Sam2ImagePredictor(Protocol):
    """Minimum official ``SAM2ImagePredictor`` surface used by the prompted runtime."""

    def set_image(self, image: Any) -> None:
        """Encode one image for the following prompts."""
        ...

    def predict(self, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Return masks, predicted IoUs and low-resolution logits for one prompt."""
        ...


class Sam2ImagePredictorRuntime:
    """Execute the official ``SAM2ImagePredictor`` box/point prompt API."""

    def __init__(
        self,
        *,
        predictor: _Sam2ImagePredictor,
        config_digest: str,
        image_decoder: Callable[[bytes], Any] | None = None,
        autocast: Callable[[Sam2RefinementConfig], AbstractContextManager[object]] | None = None,
    ) -> None:
        """Bind a predictor built for one refiner configuration.

        Args:
            predictor: Official image predictor around the loaded SAM2 model.
            config_digest: Digest of the configuration the predictor was built for.
            image_decoder: Decodes image bytes into an HWC RGB array; defaults to Pillow
                and NumPy, imported lazily.
            autocast: Builds the precision context; defaults to ``torch.autocast`` for
                ``float16``/``bfloat16`` and no context for ``float32``.
        """
        if not config_digest:
            raise ValueError("SAM2 prompted runtime configuration digest must not be empty")
        self._predictor = predictor
        self._config_digest = config_digest
        self._decode = image_decoder or _decode_rgb
        self._autocast = autocast or _refinement_autocast

    @classmethod
    def from_model(
        cls, *, model: object, config: Sam2RefinementConfig
    ) -> Sam2ImagePredictorRuntime:
        """Build the official predictor around a SAM2 model the provider already loaded."""
        module = import_module("sam2.sam2_image_predictor")
        predictor = module.SAM2ImagePredictor(
            model,
            mask_threshold=config.mask_threshold,
            max_hole_area=config.max_hole_area,
            max_sprinkle_area=config.max_sprinkle_area,
        )
        return cls(predictor=predictor, config_digest=config.digest)

    def predict_prompts(
        self,
        *,
        image: bytes,
        width: int,
        height: int,
        prompts: tuple[RefinementPrompt, ...],
        config: Sam2RefinementConfig,
    ) -> tuple[Sam2PromptedMask, ...]:
        """Encode the image once, then answer every prompt with a single mask."""
        if config.digest != self._config_digest:
            raise ValueError("SAM2 prompted runtime configuration digest does not match")
        array = self._decode(image)
        shape = tuple(getattr(array, "shape", ())[:2])
        if shape != (height, width):
            raise ValueError(
                f"decoded image dimensions {shape} differ from the prepared image {(height, width)}"
            )
        numpy = import_module("numpy")
        answers: list[Sam2PromptedMask] = []
        with self._autocast(config):
            self._predictor.set_image(array)
            for prompt in prompts:
                if prompt.box is not None:
                    box = prompt.box
                    arguments: dict[str, Any] = {
                        "box": numpy.array(
                            [box.x_min, box.y_min, box.x_max, box.y_max], dtype=numpy.float32
                        )
                    }
                else:
                    point = cast(Any, prompt.point)
                    arguments = {
                        "point_coords": numpy.array([[point.x, point.y]], dtype=numpy.float32),
                        "point_labels": numpy.array([1], dtype=numpy.int32),
                    }
                # Uma máscara por prompt: escolher entre as saídas multimask seria uma
                # política científica a mais, não declarada nesta configuração.
                masks, scores, _logits = self._predictor.predict(
                    **arguments, multimask_output=False
                )
                answers.append(
                    Sam2PromptedMask(
                        mask=tuple(
                            bool(value) for value in numpy.asarray(masks[0]).reshape(-1).tolist()
                        ),
                        predicted_iou=float(numpy.asarray(scores).reshape(-1)[0]),
                    )
                )
        return tuple(answers)


class Sam2PromptRefinement:
    """Refine grounding proposals into mask-backed regions with SAM2 box/point prompts."""

    def __init__(
        self,
        *,
        config: Sam2RefinementConfig,
        runtime: Sam2PromptRuntime,
        prepared_image_root: Path,
    ) -> None:
        """Bind the refiner to one configuration, runtime and prepared-image directory."""
        self._config = config
        self._runtime = runtime
        self._root = prepared_image_root

    def backend_provenance(self) -> BackendProvenance:
        """Return SAM2 identity as a refiner."""
        return BackendProvenance(
            backend_id="sam2",
            capability=REFINEMENT_CAPABILITY,
            provider="facebook",
            model=self._config.checkpoint,
            version=self._config.model_version,
            configuration_fingerprint=self._config.digest,
        )

    def capabilities(self) -> RegionRefinementCapabilities:
        """Declare box and point prompts, both native to SAM2."""
        return RegionRefinementCapabilities(
            prompt_geometries=frozenset({GroundingGeometry.BOX, GroundingGeometry.POINT})
        )

    def refine(self, request: RegionRefinementRequest) -> RegionRefinementExecution:
        """Validate the request, run SAM2 once per image and judge every mask.

        Raises:
            RefinementRequestError: Before inference, for an unaccepted prompt or a request
                fingerprinted for another configuration.
            ValueError: If the image no longer matches its hash, or the runtime breaks its
                seam (missing mask, mask of another size).
        """
        validate_refinement_request(request, self.capabilities())
        if request.configuration_fingerprint != self._config.digest:
            raise RefinementRequestError(
                "refinement request configuration fingerprint does not match this SAM2 "
                "refiner configuration"
            )
        payload = _read_prepared_image(self._root, request.image)
        width, height = request.image.width, request.image.height
        started = perf_counter()
        answers = self._runtime.predict_prompts(
            image=payload,
            width=width,
            height=height,
            prompts=request.prompts,
            config=self._config,
        )
        latency_ms = (perf_counter() - started) * 1000
        if len(answers) != len(request.prompts):
            raise ValueError(
                f"SAM2 runtime answered {len(answers)} of {len(request.prompts)} prompts"
            )
        provenance = self.backend_provenance()
        outcomes = []
        for prompt, answer in zip(request.prompts, answers, strict=True):
            if len(answer.mask) != width * height:
                raise ValueError(
                    f"SAM2 mask for {prompt.proposal_id!r} does not match the {width}x{height} "
                    "prepared image"
                )
            outcomes.append(
                refinement_outcome(
                    request=request,
                    prompt=prompt,
                    provenance=provenance,
                    mask=InlineMask(width=width, height=height, data=answer.mask),
                    native_scores=(
                        BackendScore(
                            name="predicted_iou",
                            value=answer.predicted_iou,
                            semantics=_PREDICTED_IOU_SEMANTICS,
                        ),
                    ),
                )
            )
        return RegionRefinementExecution(
            request=request,
            provenance=provenance,
            outcomes=tuple(outcomes),
            diagnostics=RefinementDiagnostics(latency_ms=latency_ms),
            effective_configuration=MappingProxyType(
                cast(dict[str, JsonScalar], asdict(self._config))
            ),
        )


def _read_prepared_image(root: Path, image: PreparedImage) -> bytes:
    """Read the prepared image under ``root`` and verify it is the one the request names."""
    artifact = image.payload_artifact
    if artifact is None:
        raise ValueError("the prepared image has no recorded sha256 to verify")
    reference = PurePosixPath(image.payload_reference)
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError(
            f"prepared image {image.payload_reference!r} points outside the prepared-image root"
        )
    payload = (root / reference).read_bytes()
    if sha256(payload).hexdigest() != artifact.sha256:
        raise ValueError(
            f"prepared image {image.payload_reference!r} does not match its recorded sha256"
        )
    return payload


def _decode_rgb(payload: bytes) -> Any:
    """Decode image bytes into the HWC uint8 RGB array SAM2 expects."""
    image_module = import_module("PIL.Image")
    numpy = import_module("numpy")
    with image_module.open(io.BytesIO(payload)) as image:
        return numpy.asarray(image.convert("RGB"))


def _refinement_autocast(config: Sam2RefinementConfig) -> AbstractContextManager[object]:
    """Return the ``torch.autocast`` context that realizes the configured precision."""
    if config.precision == "float32":
        return nullcontext()
    torch = import_module("torch")
    return cast(
        AbstractContextManager[object],
        torch.autocast(
            device_type=config.device.split(":", 1)[0], dtype=getattr(torch, config.precision)
        ),
    )


def _validate_unit_threshold(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"SAM2 {name} must be between zero and one")


def _parse_automatic_mask_record(
    record: Mapping[str, object], *, index: int, width: int, height: int
) -> Sam2NativeProposal:
    """Convert one documented SAM2 automatic-mask record to scalar containers."""
    box = _numeric_sequence(record.get("bbox"), "bbox", length=4)
    x, y, box_width, box_height = box
    area = _finite_number(record.get("area"), "area")
    # O SDK oficial monta o bbox a partir de índices de pixel inclusivos
    # (largura = x1 - x0), enquanto BoundingBox é semiaberto. Sem o +1 a caixa
    # fica um pixel menor que a máscara e a normalização rejeita todo candidato.
    return Sam2NativeProposal(
        proposal_id=f"sam2-{index:06d}",
        box=(x, y, x + box_width + 1, y + box_height + 1),
        mask=_binary_mask(record.get("segmentation"), width=width, height=height),
        predicted_iou=_finite_number(record.get("predicted_iou"), "predicted_iou"),
        stability_score=_finite_number(record.get("stability_score"), "stability_score"),
        metadata=(("area_pixels", area),),
    )


def _binary_mask(value: object, *, width: int, height: int) -> tuple[bool, ...]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        raise TypeError("SAM2 segmentation must be a two-dimensional binary mask")
    rows = cast(Sequence[object], native)
    if len(rows) != height:
        raise ValueError("SAM2 segmentation mask dimensions must match the discovery pass")
    flattened: list[bool] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise ValueError("SAM2 segmentation mask dimensions must match the discovery pass")
        for item in row:
            if item not in (False, True, 0, 1):
                raise TypeError("SAM2 segmentation mask must contain binary values")
            flattened.append(bool(item))
    return tuple(flattened)


def _numeric_sequence(value: object, name: str, *, length: int) -> tuple[float, ...]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if (
        not isinstance(native, Sequence)
        or isinstance(native, (str, bytes))
        or len(native) != length
    ):
        raise ValueError(f"SAM2 {name} must contain {length} numbers")
    return tuple(_finite_number(item, name) for item in native)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SAM2 {name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"SAM2 {name} must be finite")
    return result
