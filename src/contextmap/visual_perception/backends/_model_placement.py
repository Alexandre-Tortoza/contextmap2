"""Check that an injected, already loaded torch model runs where its configuration says."""

from __future__ import annotations


def verify_model_placement(
    model: object, *, device: str, precision: str | None, backend: str
) -> object:
    """Compare the model's first parameter with the configured device and precision.

    The composition root hands some runtimes a model it loaded itself; its device and
    dtype enter provenance through the configuration, so a model placed elsewhere would
    be recorded as something it is not.

    Args:
        model: Loaded torch module exposing ``parameters()``.
        device: Configured torch device, such as ``"cuda"`` or ``"cuda:0"``. Without an
            index, any index of that device type matches, as ``model.to("cuda")`` does.
        precision: Configured dtype name, such as ``"float32"``, or ``None`` when the
            runtime realizes the precision some other way and the dtype is not compared.
        backend: Backend name used in the error message.

    Returns:
        The dtype of the verified parameter, to cast inputs to without importing torch.

    Raises:
        ValueError: If the model has no parameter to inspect, or if it sits on another
            device or in another dtype than the configuration declares.
    """
    parameters = getattr(model, "parameters", None)
    parameter = next(iter(parameters()), None) if callable(parameters) else None
    if parameter is None:
        raise ValueError(f"{backend} model exposes no parameter to verify its device and dtype")
    actual_device = parameter.device
    device_type, _, device_index = device.partition(":")
    if actual_device.type != device_type or (
        device_index and str(actual_device.index) != device_index
    ):
        raise ValueError(
            f"{backend} model parameters are on {actual_device}, but the configuration "
            f"declares device {device!r}"
        )
    dtype = parameter.dtype
    if precision is not None and str(dtype).removeprefix("torch.") != precision:
        raise ValueError(
            f"{backend} model parameters are {dtype}, but the configuration declares "
            f"precision {precision!r}"
        )
    return dtype
