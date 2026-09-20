"""Capability port: substitution point for Point Representation encoders.

A deterministic geometric descriptor and a learned 3D encoder are different
producers of the same public contract: both receive the canonical prepared
support and publish a vector for :class:`PointRepresentation`. The port exists
because that variation is real; it is not a plugin system. Which encoder runs is
decided by the composition root (``runtime``), never by the execution service,
and a port implementation never constructs itself from a registry or imports a
framework into a public output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contextmap.point_representation.models import (
    EncoderIdentity,
    PreparedSupport,
    RepresentationSpace,
)


class UnencodableSupportError(ValueError):
    """Raised by an encoder when one support cannot be represented.

    The service records the support as failed and continues. Any other
    exception is a backend failure and stops the run: it is never converted into
    a default representation.
    """


@dataclass(frozen=True, kw_only=True)
class EncodedVector:
    """What an encoder returns for one support.

    Attributes:
        values: The vector, one float per component of the encoder's
            representation space; plain Python floats, never a framework tensor.
        undefined_components: Ascending indexes of components the encoder could
            not define for this support. Their values are placeholders that the
            service replaces with zero and never interprets.
    """

    values: tuple[float, ...]
    undefined_components: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Require a vector.

        Raises:
            ValueError: If ``values`` is empty.
        """
        if not self.values:
            raise ValueError("an encoded vector needs values")


@runtime_checkable
class PointEncoder(Protocol):
    """Capability port: turn a prepared 3D support into a vector."""

    def encoder_identity(self) -> EncoderIdentity:
        """Report this encoder's identity and effective configuration.

        Returns:
            The identity attached to every representation this encoder produces.
        """
        ...

    def representation_space(self) -> RepresentationSpace:
        """Report the space the produced vectors live in.

        Returns:
            The space, including the support policy the service must extract
            supports under: that policy is part of what a vector describes.
        """
        ...

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        """Represent one prepared support.

        Args:
            prepared: The support and its prepared coordinates; the encoder
                must not mutate it or read geometry any other way.

        Returns:
            One value per component of :meth:`representation_space`.

        Raises:
            UnencodableSupportError: If this support cannot be represented. The
                encoder must raise instead of returning a zero or default vector.
        """
        ...
