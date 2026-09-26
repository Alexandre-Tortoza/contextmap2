"""Check that an injected, already loaded torch model runs where its configuration says."""

from __future__ import annotations


def verify_model_placement(
    model: object,
    *,
    device: str,
    precision: str,
    backend: str,
    accepted_dtypes: frozenset[str] | None = None,
) -> object:
    """Compare the model's first parameter with the configured device and precision.

    The composition root hands some runtimes a model it loaded itself; its device and
    dtype enter provenance through the configuration, so a model placed elsewhere would
    be recorded as something it is not.

    Args:
        model: Loaded torch module exposing ``parameters()``.
        device: Configured torch device, such as ``"cuda"`` or ``"cuda:0"``. Without an
            index, any index of that device type matches, as ``model.to("cuda")`` does.
        precision: Configured precision name, such as ``"float32"``.
        backend: Backend name used in the error message.
        accepted_dtypes: Weight dtype names under which the runtime really infers in
            ``precision``. Defaults to ``{precision}``, right for a runtime without
            autocast, where the weight dtype is the inference precision.

    Returns:
        The dtype of the verified parameter, to cast inputs to without importing torch.

    Raises:
        ValueError: If the model has no parameter to inspect, or if it sits on another
            device or holds weights in a dtype the configured precision does not accept.
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
    accepted = accepted_dtypes or frozenset({precision})
    if str(dtype).removeprefix("torch.") not in accepted:
        raise ValueError(
            f"{backend} model parameters are {dtype}, but the configuration declares "
            f"precision {precision!r}, which accepts weights in {', '.join(sorted(accepted))}"
        )
    return dtype
