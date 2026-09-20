"""Deterministic fake encoders: the core service is tested with no model installed."""

from __future__ import annotations

from pointrep_builders import radius_policy

from contextmap.point_representation import (
    EncodedVector,
    EncoderIdentity,
    PreparedSupport,
    RepresentationSpace,
    SupportPolicy,
    UnencodableSupportError,
)


class FakeEncoder:
    """Encodes a support as ``(support size, mean local x, mean local y)``.

    A pure function of the prepared support, so runs are exactly repeatable.
    """

    def __init__(
        self,
        policy: SupportPolicy | None = None,
        *,
        family: str = "fake_encoder",
        checkpoint: str | None = None,
    ) -> None:
        self.calls: list[PreparedSupport] = []
        self._family = family
        self._space = RepresentationSpace(
            family=family,
            model="support-summary",
            version="1",
            checkpoint=checkpoint,
            dimension=3,
            dtype="float32",
            normalization="none",
            input_definition="xyz-local-prepared",
            support_semantics=policy if policy is not None else radius_policy(0.6),
            feature_names=("support_size", "mean_x", "mean_y"),
        )

    def encoder_identity(self) -> EncoderIdentity:
        return EncoderIdentity(
            backend_id=self._family,
            backend_version="1",
            configuration_fingerprint="sha256:fake-config",
            checkpoint_hash=None,
        )

    def representation_space(self) -> RepresentationSpace:
        return self._space

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        self.calls.append(prepared)
        count = len(prepared.local_coordinates_m)
        mean_x = sum(point[0] for point in prepared.local_coordinates_m) / count
        mean_y = sum(point[1] for point in prepared.local_coordinates_m) / count
        return EncodedVector(values=(float(count), mean_x, mean_y))


class RejectingSmallSupportsEncoder(FakeEncoder):
    """Declares every support below ``minimum`` points unencodable."""

    def __init__(self, policy: SupportPolicy | None = None, *, minimum: int = 4) -> None:
        super().__init__(policy)
        self._minimum = minimum

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        if len(prepared.support.geometry_refs) < self._minimum:
            raise UnencodableSupportError(f"fewer than {self._minimum} supporting points")
        return super().encode(prepared)
