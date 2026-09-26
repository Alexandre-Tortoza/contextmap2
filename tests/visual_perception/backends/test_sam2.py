import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeLoadedModel
from mask_cases import BACKEND_SEEDS, GOLDEN, digest, sam2_candidates

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    AuditedRegionDiscovery,
    BoundingBox,
    InlineMask,
    PreparedImage,
    Region2D,
    RegionDiscovery,
)
from contextmap.visual_perception.backends import sam2 as sam2_module
from contextmap.visual_perception.backends.sam2 import (
    Sam2AutomaticMaskRuntime,
    Sam2Config,
    Sam2NativeProposal,
    Sam2RegionDiscovery,
)
from contextmap.visual_perception.discovery import (
    DiscoveryInput,
    DiscoveryPass,
    PassKind,
)


@dataclass(frozen=True)
class MaterializedImage:
    size: tuple[int, int]
    token: tuple[str, str]


def _materialized_image(discovery_input: DiscoveryInput, kind: str) -> MaterializedImage:
    discovery_pass = discovery_input.discovery_pass
    return MaterializedImage(
        size=(discovery_pass.input_width, discovery_pass.input_height),
        token=(kind, discovery_pass.pass_id),
    )


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-2"),
            payload_reference="outputs/frame-2.png",
            payload_artifact=ArtifactReference(
                uri="outputs/frame-2.png",
                sha256=sha256(b"frame-2").hexdigest(),
                media_type="image/png",
            ),
            width=8,
            height=6,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="tile-0002",
            kind=PassKind.TILE,
            window=BoundingBox(x_min=2, y_min=1, x_max=6, y_max=5),
        ),
        perception_run_id="run-3",
        perception_result_id="result-9",
    )


def _mask(width: int, height: int, inside: Callable[[int, int], bool]) -> InlineMask:
    return InlineMask(
        np.array([[inside(x, y) for x in range(width)] for y in range(height)], dtype=bool)
    )


class FakeSam2Runtime:
    def __init__(self) -> None:
        self.received: list[DiscoveryInput] = []

    def predict(
        self, discovery_input: DiscoveryInput, config: Sam2Config
    ) -> tuple[Sam2NativeProposal, ...]:
        self.received.append(discovery_input)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        return (
            Sam2NativeProposal(
                proposal_id="sam2-1",
                box=(0.0, 0.0, 2.0, 2.0),
                mask=_mask(width, height, lambda x, y: x < 2 and y < 2),
                predicted_iou=0.91,
                stability_score=0.87,
            ),
            Sam2NativeProposal(
                proposal_id="sam2-low",
                box=(2.0, 2.0, 4.0, 4.0),
                mask=_mask(width, height, lambda x, y: x >= 2 and y >= 2),
                predicted_iou=0.2,
                stability_score=0.4,
            ),
        )


def test_sam2_normalizes_native_proposals_with_complete_provenance() -> None:
    runtime = FakeSam2Runtime()
    config = Sam2Config(
        checkpoint="facebook/sam2-hiera-large",
        model_version="2.1",
        device="cuda:0",
        precision="float16",
        predicted_iou_threshold=0.8,
        stability_threshold=0.8,
        automatic_mask_settings=(("points_per_side", 32),),
    )
    backend = Sam2RegionDiscovery(config=config, runtime=runtime)

    output = backend.discover_candidates(_input())

    assert isinstance(backend, RegionDiscovery)
    assert len(runtime.received) == 1
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    assert candidate.candidate_id == "sam2-1"
    assert candidate.image_width == 4
    assert candidate.image_height == 4
    assert candidate.score is not None
    assert candidate.score.name == "predicted_iou"
    assert candidate.provenance.backend_id == "sam2"
    assert candidate.provenance.checkpoint == "facebook/sam2-hiera-large"
    assert candidate.provenance.discovery_pass_id == "tile-0002"
    assert candidate.provenance.config_digest == config.digest
    assert dict(candidate.native_metadata)["stability_score"] == 0.87
    assert dict(output.diagnostics.metadata)["raw_proposal_count"] == 2
    json.dumps(candidate.to_dict())
    regions = backend.discover(_input().prepared_image)
    assert all(isinstance(region, Region2D) for region in regions)
    assert regions[0].provenance == backend.backend_provenance()


def test_sam2_reports_the_audit_of_the_regions_it_discovers() -> None:
    """#611: the regions of the public port come with the passes, rejections and merges."""
    backend = Sam2RegionDiscovery(
        config=Sam2Config(checkpoint="facebook/sam2-hiera-large", model_version="2.1"),
        runtime=FakeSam2Runtime(),
    )
    image = _input().prepared_image

    audited = backend.discover_audited(image)

    assert isinstance(backend, AuditedRegionDiscovery)
    assert audited.regions == backend.discover(image)
    assert audited.audit.source_observation_id == image.source_observation_id
    assert audited.audit.backend == backend.backend_provenance()
    assert [discovery_pass.pass_id for discovery_pass in audited.audit.passes] == ["full-frame"]


def test_sam2_configuration_rejects_invalid_thresholds_and_unknown_values() -> None:
    with pytest.raises(ValueError, match="predicted_iou_threshold"):
        Sam2Config(checkpoint="sam2", model_version="2.1", predicted_iou_threshold=1.1)
    with pytest.raises(ValueError, match="unique"):
        Sam2Config(
            checkpoint="sam2",
            model_version="2.1",
            automatic_mask_settings=(("points_per_side", 16), ("points_per_side", 32)),
        )


def test_sam2_native_shape_errors_fail_explicitly() -> None:
    class InvalidRuntime:
        def predict(
            self, discovery_input: DiscoveryInput, config: Sam2Config
        ) -> tuple[Sam2NativeProposal, ...]:
            return (
                Sam2NativeProposal(
                    proposal_id="bad",
                    box=(0.0, 0.0, 2.0, 2.0),
                    mask=InlineMask(np.ones((1, 1), dtype=bool)),
                    predicted_iou=0.9,
                    stability_score=0.9,
                ),
            )

    backend = Sam2RegionDiscovery(
        config=Sam2Config(checkpoint="sam2", model_version="2.1"), runtime=InvalidRuntime()
    )

    with pytest.raises(ValueError, match="mask dimensions"):
        backend.discover_candidates(_input())


def test_sam2_official_automatic_mask_output_is_isolated_as_scalars() -> None:
    class NativeArray:
        def __init__(self, value: object) -> None:
            self._value = value

        def tolist(self) -> object:
            return self._value

    class AutomaticMaskGenerator:
        def __init__(self) -> None:
            self.received: list[object] = []

        def generate(self, image: object) -> list[dict[str, object]]:
            self.received.append(image)
            return [
                {
                    "segmentation": NativeArray(
                        [
                            [True, True, False, False],
                            [False, True, False, False],
                            [False, False, False, False],
                            [False, False, False, False],
                        ]
                    ),
                    # Convenção do SDK oficial: bbox = [x0, y0, x1 - x0, y1 - y0] com índices
                    # de pixel inclusivos; a máscara acima ocupa x=0..1 e y=0..1.
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "area": 3,
                    "predicted_iou": 0.91,
                    "stability_score": 0.88,
                }
            ]

    config = Sam2Config(checkpoint="facebook/sam2-hiera-large", model_version="2.1")
    generator = AutomaticMaskGenerator()
    runtime = Sam2AutomaticMaskRuntime(
        mask_generator=generator,
        image_loader=lambda discovery_input: _materialized_image(discovery_input, "pixels"),
        config_digest=config.digest,
    )

    proposals = runtime.predict(_input(), config)

    assert generator.received == [MaterializedImage((4, 4), ("pixels", "tile-0002"))]
    # A caixa canônica é semiaberta: a borda exclusiva é o último índice inclusivo + 1.
    assert proposals[0].box == (0.0, 0.0, 2.0, 2.0)
    assert proposals[0].mask.as_array().reshape(-1)[:6].tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    assert proposals[0].predicted_iou == 0.91
    assert dict(proposals[0].metadata)["area_pixels"] == 3


def test_sam2_runtime_rejects_configuration_or_shape_drift() -> None:
    class InvalidGenerator:
        def generate(self, image: object) -> list[dict[str, object]]:
            return [
                {
                    "segmentation": [[True]],
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "area": 1,
                    "predicted_iou": 0.9,
                    "stability_score": 0.9,
                }
            ]

    config = Sam2Config(checkpoint="sam2", model_version="2.1")
    runtime = Sam2AutomaticMaskRuntime(
        mask_generator=InvalidGenerator(),
        image_loader=lambda discovery_input: _materialized_image(discovery_input, "pixels"),
        config_digest=config.digest,
    )

    with pytest.raises(ValueError, match="mask dimensions"):
        runtime.predict(_input(), config)
    with pytest.raises(ValueError, match="configuration digest"):
        runtime.predict(_input(), Sam2Config(checkpoint="another-sam2", model_version="2.1"))


def test_sam2_runtime_rejects_materialized_image_dimension_drift() -> None:
    class EmptyGenerator:
        def generate(self, image: object) -> list[dict[str, object]]:
            return []

    config = Sam2Config(checkpoint="sam2", model_version="2.1")
    runtime = Sam2AutomaticMaskRuntime(
        mask_generator=EmptyGenerator(),
        image_loader=lambda discovery_input: MaterializedImage(
            (8, 8), ("wrong-size", discovery_input.discovery_pass.pass_id)
        ),
        config_digest=config.digest,
    )

    with pytest.raises(ValueError, match="materialized discovery image dimensions"):
        runtime.predict(_input(), config)


def test_sam2_runtime_materializes_distinct_scaled_model_inputs() -> None:
    class RecordingGenerator:
        def __init__(self) -> None:
            self.received: list[object] = []

        def generate(self, image: object) -> list[dict[str, object]]:
            self.received.append(image)
            return []

    loaded: list[MaterializedImage] = []

    def load(discovery_input: DiscoveryInput) -> MaterializedImage:
        image = _materialized_image(discovery_input, "instrumented-pixels")
        loaded.append(image)
        return image

    config = Sam2Config(checkpoint="sam2", model_version="2.1")
    generator = RecordingGenerator()
    runtime = Sam2AutomaticMaskRuntime(
        mask_generator=generator,
        image_loader=load,
        config_digest=config.digest,
    )
    baseline = _input()
    scaled = replace(
        baseline,
        discovery_pass=replace(baseline.discovery_pass, scale=2.0),
    )

    runtime.predict(baseline, config)
    runtime.predict(scaled, config)

    assert [image.size for image in loaded] == [(4, 4), (8, 8)]
    assert generator.received == loaded
    assert generator.received[0] is not generator.received[1]


def test_sam2_accepts_official_masks_whose_bbox_uses_inclusive_indices() -> None:
    class InclusiveBoxGenerator:
        """Follow the official convention: bbox = [x0, y0, x1 - x0, y1 - y0], inclusive."""

        def generate(self, image: object) -> list[dict[str, object]]:
            assert isinstance(image, MaterializedImage)
            width, height = image.size
            mask = [[2 <= x <= 4 and 1 <= y <= 3 for x in range(width)] for y in range(height)]
            return [
                {
                    "segmentation": mask,
                    "bbox": [2, 1, 2, 2],
                    "area": 9,
                    "predicted_iou": 0.9,
                    "stability_score": 0.9,
                }
            ]

    config = Sam2Config(checkpoint="facebook/sam2.1-hiera-tiny", model_version="2.1")
    runtime = Sam2AutomaticMaskRuntime(
        mask_generator=InclusiveBoxGenerator(),
        image_loader=lambda discovery_input: _materialized_image(discovery_input, "pixels"),
        config_digest=config.digest,
    )
    backend = Sam2RegionDiscovery(config=config, runtime=runtime)

    regions = backend.discover(_input().prepared_image)

    assert len(regions) == 1
    box = regions[0].bounding_box
    assert (box.x, box.y, box.width, box.height) == (2, 1, 3, 3)
    assert regions[0].area_pixels == 9


@pytest.mark.parametrize("seed", BACKEND_SEEDS)
def test_sam2_mask_conversion_matches_the_recorded_behaviour(seed: int) -> None:
    # #593: do resultado nativo do SDK ao RegionCandidate, registrado antes da vetorização.
    assert digest(sam2_candidates(seed)) == GOLDEN["sam2"][str(seed)]


def test_sam2_configuration_requires_an_explicit_model_version() -> None:
    with pytest.raises(TypeError, match="model_version"):
        Sam2Config(checkpoint="facebook/sam2.1-hiera-tiny")  # type: ignore[call-arg]


def _install_fake_sam2_sdk(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Replace the official generator module; return the models it was built around."""
    built: list[object] = []

    class SAM2AutomaticMaskGenerator:
        def __init__(self, *, model: object, **settings: object) -> None:
            built.append(model)

        def generate(self, image: object) -> list[dict[str, object]]:
            return []

    module = SimpleNamespace(SAM2AutomaticMaskGenerator=SAM2AutomaticMaskGenerator)
    monkeypatch.setattr(sam2_module, "import_module", lambda name: module)
    return built


def _placed_config(*, device: str) -> Sam2Config:
    return Sam2Config(
        checkpoint="facebook/sam2.1-hiera-tiny",
        model_version="2.1",
        device=device,
        precision="float32",
    )


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (FakeLoadedModel("cpu", "float32"), "device 'cuda:0'"),
        (FakeLoadedModel("cuda:1", "float32"), "device 'cuda:0'"),
        (FakeLoadedModel("cuda:0", "float16"), "precision 'float32'"),
    ],
)
def test_sam2_from_model_rejects_a_model_placed_unlike_the_configuration(
    model: FakeLoadedModel, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _install_fake_sam2_sdk(monkeypatch)

    with pytest.raises(ValueError, match=message):
        Sam2AutomaticMaskRuntime.from_model(
            model=model,
            image_loader=lambda discovery_input: _materialized_image(discovery_input, "pixels"),
            config=_placed_config(device="cuda:0"),
        )

    assert built == []


def test_sam2_from_model_accepts_a_model_placed_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _install_fake_sam2_sdk(monkeypatch)
    model = FakeLoadedModel("cuda:0", "float32")

    # Sem índice no config, qualquer GPU CUDA confere, como em model.to("cuda").
    Sam2AutomaticMaskRuntime.from_model(
        model=model,
        image_loader=lambda discovery_input: _materialized_image(discovery_input, "pixels"),
        config=_placed_config(device="cuda"),
    )

    assert built == [model]
