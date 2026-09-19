from collections.abc import Sequence

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    FeatureExtractor,
    FeatureScope,
    PerceptionResultId,
    PreparedImage,
    Region2D,
    RegionDiscovery,
    RegionId,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpreter,
    SemanticScorer,
    SemanticSupport,
)
from contextmap.visual_perception.models import ClaimId, HypothesisRole
from contextmap.visual_perception.models import PreparedImage as ModelPreparedImage
from contextmap.visual_perception.models import Region2D as ModelRegion2D
from contextmap.visual_perception.ports import RegionDiscovery as PortRegionDiscovery


def test_region_discovery_uses_one_public_contract_family() -> None:
    assert PreparedImage is ModelPreparedImage
    assert Region2D is ModelRegion2D
    assert RegionDiscovery is PortRegionDiscovery


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference="debug/frame-0001/prepared.jpg",
        width=640,
        height=480,
    )


def _semantic_provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="fake-interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="interpreter",
            version="0.1",
        ),
        task_identity="region-labeling",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
    )


class _FakeRegionDiscoveryA:
    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake-a",
            capability="region_discovery",
            provider="fake",
            model="a",
            version="0.1",
        )

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        return (
            Region2D(
                region_id=RegionId("region-0001"),
                bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
                provenance=self.backend_provenance(),
            ),
        )


class _FakeRegionDiscoveryB:
    """A second, independent RegionDiscovery implementation with different output."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake-b",
            capability="region_discovery",
            provider="fake",
            model="b",
            version="0.1",
        )

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        return (
            Region2D(
                region_id=RegionId("region-0001"),
                bounding_box=BoundingBox2D(x=5, y=5, width=20, height=20),
                provenance=self.backend_provenance(),
            ),
            Region2D(
                region_id=RegionId("region-0002"),
                bounding_box=BoundingBox2D(x=30, y=30, width=15, height=15),
                provenance=self.backend_provenance(),
            ),
        )


class _FakeDenseFeatureExtractor:
    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake-dense",
            capability="feature_extractor",
            provider="fake",
            model="dense",
            version="0.1",
        )

    def required_scope(self) -> FeatureScope:
        return FeatureScope.DENSE

    def extract(self, image: PreparedImage, regions: Sequence[Region2D] = ()) -> Sequence[object]:
        return ()


class _FakeSemanticInterpreter:
    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake-interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="interpreter",
            version="0.1",
        )

    def interpret_scene(self, image: PreparedImage) -> SceneContext | None:
        return None

    def interpret_regions(
        self, image: PreparedImage, regions: Sequence[Region2D]
    ) -> Sequence[SemanticClaim]:
        return tuple(
            SemanticClaim(
                claim_id=ClaimId(f"claim-{region.region_id}"),
                source_observation_id=image.source_observation_id,
                perception_result_id=PerceptionResultId("result-0001"),
                hypothesis="a fake object",
                role=HypothesisRole.PRIMARY,
                provenance=_semantic_provenance(),
                region_id=region.region_id,
            )
            for region in regions
        )


class _FakeSemanticScorer:
    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake-scorer",
            capability="semantic_scorer",
            provider="fake",
            model="scorer",
            version="0.1",
        )

    def score(
        self,
        claims: Sequence[SemanticClaim],
        image: PreparedImage,
        regions: Sequence[Region2D] = (),
    ) -> Sequence[SemanticSupport]:
        return tuple(
            SemanticSupport(
                claim_id=claim.claim_id, support_score=1.0, provenance=self.backend_provenance()
            )
            for claim in claims
        )


def _run_discovery(backend: RegionDiscovery, image: PreparedImage) -> Sequence[Region2D]:
    """Downstream-style consumption: no branching on which backend this is."""
    return backend.discover(image)


def test_two_region_discovery_backends_are_interchangeable() -> None:
    image = _image()

    regions_a = _run_discovery(_FakeRegionDiscoveryA(), image)
    regions_b = _run_discovery(_FakeRegionDiscoveryB(), image)

    assert len(regions_a) == 1
    assert len(regions_b) == 2
    assert isinstance(_FakeRegionDiscoveryA(), RegionDiscovery)
    assert isinstance(_FakeRegionDiscoveryB(), RegionDiscovery)


def test_dense_feature_extractor_declares_its_scope_without_regions() -> None:
    extractor = _FakeDenseFeatureExtractor()

    assert extractor.required_scope() is FeatureScope.DENSE
    assert extractor.extract(_image()) == ()
    assert isinstance(extractor, FeatureExtractor)


def test_semantic_interpreter_produces_region_scoped_claims() -> None:
    interpreter = _FakeSemanticInterpreter()
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
        provenance=BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="0.1",
        ),
    )

    claims = interpreter.interpret_regions(_image(), (region,))

    assert len(claims) == 1
    assert claims[0].region_id == region.region_id
    assert interpreter.interpret_scene(_image()) is None
    assert isinstance(interpreter, SemanticInterpreter)


def test_semantic_scorer_produces_support_without_mutating_claims() -> None:
    scorer = _FakeSemanticScorer()
    claim = SemanticClaim(
        claim_id=ClaimId("claim-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("result-0001"),
        hypothesis="a fake object",
        role=HypothesisRole.PRIMARY,
        provenance=_semantic_provenance(),
    )

    supports = scorer.score((claim,), _image())

    assert len(supports) == 1
    assert supports[0].claim_id == claim.claim_id
    assert claim.confidence is None  # the original claim is untouched
    assert isinstance(scorer, SemanticScorer)


def test_one_model_can_satisfy_two_capabilities_via_distinct_adapters() -> None:
    """Florence-2 as region discovery and as semantic interpretation stay separate adapters."""

    class _FlorenceRegionAdapter:
        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="florence2",
                capability="region_discovery",
                provider="microsoft",
                model="florence-2",
                version="1.0",
            )

        def discover(self, image: PreparedImage) -> Sequence[Region2D]:
            return ()

    class _FlorenceSemanticAdapter:
        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="florence2",
                capability="semantic_interpreter",
                provider="microsoft",
                model="florence-2",
                version="1.0",
            )

        def interpret_scene(self, image: PreparedImage) -> SceneContext | None:
            return None

        def interpret_regions(
            self, image: PreparedImage, regions: Sequence[Region2D]
        ) -> Sequence[SemanticClaim]:
            return ()

    region_adapter = _FlorenceRegionAdapter()
    semantic_adapter = _FlorenceSemanticAdapter()

    assert isinstance(region_adapter, RegionDiscovery)
    assert isinstance(semantic_adapter, SemanticInterpreter)
    assert region_adapter.backend_provenance().capability == "region_discovery"
    assert semantic_adapter.backend_provenance().capability == "semantic_interpreter"
