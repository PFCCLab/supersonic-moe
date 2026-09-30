# ********************************************************************************
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ********************************************************************************

"""Invocation-scoped compatibility carriers for native SM90 FP8 kernels.

The public carrier remains a BF16 tensor so existing PyLayer signatures do not
change.  Native E4M3 data and raw UE8M0 scales live in this registry and are
resolved by storage identity, including permute/T views.  This module is kept
free of QuACK/CUTLASS imports so it is safe to exercise on CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import itertools
import threading
from typing import Any, Callable, Iterator, Mapping
import weakref


PayloadRecipe = Callable[[], Any]


@dataclass(frozen=True)
class WeakTensorDescriptor:
    """Weak tensor metadata that never keeps a registry owner alive."""

    reference: weakref.ReferenceType
    description: str

    def resolve(self, *, required: bool = True) -> Any:
        value = self.reference()
        if value is None and required:
            raise RuntimeError(f"SM90 {self.description} is no longer alive")
        return value


@dataclass
class PendingSM90Payload:
    data: Any
    scales: Any
    group_size: int
    role: str
    metadata: dict[str, Any]
    invocation: "SM90Invocation"
    owner_id: int
    owner_data: bool = False
    owner_scales: bool = False
    finalizer: weakref.finalize | None = field(default=None, repr=False)

    def as_tuple(self):
        source = resolve_sm90_metadata_tensor(
            self.metadata, "source", required=False
        )
        data = source if self.owner_data else self.data
        scales = source if self.owner_scales else self.scales
        return data, scales, self.group_size, self.role, self.metadata


@dataclass(frozen=True)
class SM90FP8ConfigSnapshot:
    """Fields that determine native Hopper kernel and carrier semantics."""

    enabled: bool
    fused_gated: bool
    save_z_fp8: bool
    recompute_z: bool
    fuse_y1_quant: bool
    epilogue_quant: bool
    fp8_wgrad: bool
    fuse_y1_bf16_trunc: bool
    swiglu_clamp_value: float
    activation_group_size: int
    weight_block_size: int
    fp8_backend: Any


@dataclass(frozen=True)
class SM90Invocation:
    """Immutable snapshot for one exact-SM90 expert invocation."""

    invocation_id: int
    fp8_config: SM90FP8ConfigSnapshot
    activation_group_size: int
    weight_block_size: int
    activation: Any
    backend: Any

    @property
    def profile(self) -> tuple[int, int]:
        return self.activation_group_size, self.weight_block_size


@dataclass
class SM90Carrier:
    """Native payload associated with a BF16 compatibility tensor."""

    invocation_id: int
    role: str
    orientation: str
    data: Any = None
    scales: Any = None
    recipe: PayloadRecipe | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retain_count: int = 0
    deferred_wgrad_count: int = 0
    _keys: set[tuple[Any, ...]] = field(default_factory=set, repr=False)
    _owners: set[int] = field(default_factory=set, repr=False)
    _finalizers: dict[int, weakref.finalize] = field(default_factory=dict, repr=False)

    @property
    def is_materialized(self) -> bool:
        return self.data is not None

    @property
    def is_deferred(self) -> bool:
        return self.recipe is not None and self.data is None

    def materialize_colwise(self) -> "SM90Carrier":
        """Materialize the independent Hopper wgrad K-major payload once."""
        if self.metadata.get("colwise_data") is not None:
            return self
        recipe = self.metadata.get("colwise_recipe")
        if recipe is None:
            return self
        result = recipe()
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise RuntimeError(
                "SM90 colwise recipe must return (data, scales[, offsets])"
            )
        self.metadata["colwise_data"] = result[0]
        self.metadata["colwise_scales"] = result[1]
        if len(result) == 3:
            self.metadata["colwise_packed_offsets"] = result[2]
        self.metadata["colwise_recipe"] = None
        return self

    def materialize(self) -> "SM90Carrier":
        """Run a recompute recipe once and install its native payload."""
        if self.data is not None or self.recipe is None:
            return self
        result = self.recipe()
        if isinstance(result, SM90Carrier):
            self.data, self.scales = result.data, result.scales
            self.metadata.update(result.metadata)
        elif isinstance(result, tuple):
            if not result:
                raise RuntimeError("SM90 carrier recipe returned an empty tuple")
            self.data = result[0]
            self.scales = result[1] if len(result) > 1 else None
        else:
            self.data = result
        self.recipe = None
        return self


_CURRENT_INVOCATION: ContextVar[SM90Invocation | None] = ContextVar(
    "sonicmoe_sm90_fp8_invocation", default=None
)
_INVOCATION_IDS = itertools.count(1)
_REGISTRY: dict[tuple[Any, ...], SM90Carrier] = {}
_INVOCATION_KEYS: dict[int, set[tuple[Any, ...]]] = {}
_PENDING_PAYLOADS: dict[tuple[Any, ...], PendingSM90Payload] = {}
_RETAINED_INVOCATIONS: dict[int, SM90Invocation] = {}
_INVOCATION_OWNERS: dict[int, dict[int, weakref.finalize]] = {}
_LOCK = threading.RLock()
_UNSET = object()


def _device_key(tensor: Any) -> tuple[str, int | None]:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", None)
    if device_type is None:
        text = str(device).lower()
        device_type = "cuda" if ("cuda" in text or "gpu" in text) else text
    index = getattr(device, "index", None)
    if index is None:
        index = getattr(device, "id", None)
    return str(device_type), index


def _storage_offset(tensor: Any) -> int:
    for name in ("_offset", "storage_offset"):
        value = getattr(tensor, name, None)
        if value is not None:
            try:
                return int(value() if callable(value) else value)
            except (TypeError, RuntimeError, ValueError):
                pass
    return 0


def _storage_pointer(tensor: Any) -> int | None:
    """Best-effort base pointer without requiring native Torch storage APIs."""
    storage_fn = getattr(tensor, "untyped_storage", None)
    if callable(storage_fn):
        try:
            storage = storage_fn()
            pointer_fn = getattr(storage, "data_ptr", None)
            if callable(pointer_fn):
                return int(pointer_fn())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    pointer_fn = getattr(tensor, "data_ptr", None)
    if callable(pointer_fn):
        try:
            pointer = int(pointer_fn())
            element_size_fn = getattr(tensor, "element_size", None)
            if callable(element_size_fn):
                pointer -= _storage_offset(tensor) * int(element_size_fn())
            return pointer
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return None


def storage_identity(tensor: Any) -> tuple[Any, ...]:
    """Return a proxy-safe storage identity shared by permute/T views."""
    pointer = _storage_pointer(tensor)
    if pointer is None:
        return ("object", id(tensor))
    return ("storage", *_device_key(tensor), pointer)


def _shape_stride(tensor: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    stride_value = getattr(tensor, "stride", ())
    try:
        stride = tuple(int(value) for value in (
            stride_value() if callable(stride_value) else stride_value
        ))
    except (TypeError, RuntimeError, ValueError):
        stride = ()
    return shape, stride


def view_identity(tensor: Any) -> tuple[Any, ...]:
    """Identify one logical view while retaining its base-storage relation."""
    shape, stride = _shape_stride(tensor)
    return ("view", storage_identity(tensor), _storage_offset(tensor), shape, stride)


def _registry_key(invocation_id: int, tensor: Any) -> tuple[Any, ...]:
    return invocation_id, *view_identity(tensor)


def _object_key(invocation_id: int, tensor: Any) -> tuple[Any, ...]:
    return invocation_id, "object", id(tensor)


def _invocation_id(invocation: SM90Invocation | SM90Carrier | int) -> int:
    return int(
        invocation
        if isinstance(invocation, int)
        else invocation.invocation_id
    )


def current_sm90_invocation() -> SM90Invocation | None:
    return _CURRENT_INVOCATION.get()


def _weak_tensor_descriptor(tensor: Any, description: str) -> WeakTensorDescriptor:
    try:
        return WeakTensorDescriptor(weakref.ref(tensor), description)
    except TypeError as exc:
        raise TypeError(f"SM90 {description} must support weak references") from exc


def resolve_sm90_metadata_tensor(
    metadata: Mapping[str, Any], name: str, *, required: bool = True
) -> Any:
    value = metadata.get(name)
    if isinstance(value, WeakTensorDescriptor):
        return value.resolve(required=required)
    if isinstance(value, weakref.ReferenceType):
        value = value()
    if value is None and required:
        raise RuntimeError(f"SM90 metadata tensor {name!r} is no longer alive")
    return value


def _detach_pending(payload: PendingSM90Payload) -> None:
    if payload.finalizer is not None and payload.finalizer.alive:
        payload.finalizer.detach()
    payload.finalizer = None


def _pending_owner_finalized(key: tuple[Any, ...], owner_id: int) -> None:
    with _LOCK:
        payload = _PENDING_PAYLOADS.get(key)
        if payload is not None and payload.owner_id == owner_id:
            _PENDING_PAYLOADS.pop(key, None)
    if payload is not None:
        _collect_invocation(payload.invocation.invocation_id, force=False)


def bind_sm90_payload(
    tensor: Any,
    data: Any,
    scales: Any,
    group_size: int,
    *,
    role: str = "communication",
    metadata: Mapping[str, Any] | None = None,
    invocation: SM90Invocation | None = None,
) -> None:
    """Bind bytes to one explicit invocation before backward re-enters it."""
    invocation = invocation or current_sm90_invocation()
    if invocation is None:
        raise RuntimeError(
            "bind_sm90_payload requires an active or explicit SM90 invocation"
        )
    key = view_identity(tensor)
    owner_id = id(tensor)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("source", _weak_tensor_descriptor(tensor, "payload source"))
    payload_metadata["invocation_id"] = invocation.invocation_id
    payload = PendingSM90Payload(
        None if data is tensor else data,
        None if scales is tensor else scales,
        int(group_size),
        str(role),
        payload_metadata,
        invocation,
        owner_id,
        owner_data=data is tensor,
        owner_scales=scales is tensor,
    )
    try:
        payload.finalizer = weakref.finalize(
            tensor, _pending_owner_finalized, key, owner_id
        )
    except (TypeError, RuntimeError) as exc:
        raise TypeError("SM90 pending payload owner must support finalization") from exc
    with _LOCK:
        previous = _PENDING_PAYLOADS.get(key)
        if previous is not None:
            _detach_pending(previous)
        _PENDING_PAYLOADS[key] = payload
        _RETAINED_INVOCATIONS[invocation.invocation_id] = invocation


def _payload_tuple(payload: PendingSM90Payload | None):
    return None if payload is None else payload.as_tuple()


def consume_sm90_payload(tensor: Any, *, invocation: SM90Invocation | None = None):
    payload = _pop_pending_payload(tensor, invocation=invocation)
    return _payload_tuple(payload)


def _matching_pending_keys(
    tensor: Any, invocation: SM90Invocation | None
) -> list[tuple[Any, ...]]:
    exact = view_identity(tensor)
    storage = storage_identity(tensor)
    invocation_id = None if invocation is None else invocation.invocation_id
    exact_payload = _PENDING_PAYLOADS.get(exact)
    if exact_payload is not None and (
        invocation_id is None or exact_payload.invocation.invocation_id == invocation_id
    ):
        return [exact]
    return [
        key for key, payload in _PENDING_PAYLOADS.items()
        if len(key) > 2 and key[0] == "view" and key[1] == storage
        and (invocation_id is None or payload.invocation.invocation_id == invocation_id)
    ]


def find_pending_sm90_invocations(tensor: Any) -> tuple[SM90Invocation, ...]:
    """Return live invocation owners for pending payloads on one storage."""
    exact = view_identity(tensor)
    storage = storage_identity(tensor)
    with _LOCK:
        matches = {
            payload.invocation.invocation_id: payload.invocation
            for key, payload in _PENDING_PAYLOADS.items()
            if key == exact
            or (len(key) > 2 and key[0] == "view" and key[1] == storage)
        }
    return tuple(matches.values())


def find_sm90_operand_invocations(tensor: Any) -> tuple[SM90Invocation, ...]:
    """Resolve invocation candidates from live carriers and pending payloads."""
    matches = {}
    for carrier in find_sm90_storage_carriers(tensor):
        invocation = carrier.metadata.get("invocation")
        if invocation is not None:
            matches[carrier.invocation_id] = invocation
    for invocation in find_pending_sm90_invocations(tensor):
        matches[invocation.invocation_id] = invocation
    return tuple(matches.values())


def retained_sm90_invocations() -> tuple[SM90Invocation, ...]:
    """Return retained native invocations for explicit backward ownership."""
    with _LOCK:
        return tuple(
            invocation
            for invocation in _RETAINED_INVOCATIONS.values()
            if _is_native_invocation(invocation)
        )


def find_sm90_invocation_payload(
    invocation: SM90Invocation, shape: tuple[int, ...], roles: tuple[str, ...]
) -> SM90Carrier | None:
    """Find a role-prioritized native payload matching a compatibility view."""
    with _LOCK:
        carriers = tuple(
            {
                id(carrier): carrier
                for carrier in _REGISTRY.values()
                if carrier.invocation_id == invocation.invocation_id
            }.values()
        )
    role_priority = {role: index for index, role in enumerate(roles)}
    matches = [
        carrier
        for carrier in carriers
        if carrier.data is not None
        and tuple(carrier.data.shape) == tuple(shape)
        and carrier.role in role_priority
    ]
    if not matches:
        return None
    return min(matches, key=lambda carrier: role_priority[carrier.role])


def get_pending_sm90_payload(
    tensor: Any, *, invocation: SM90Invocation | None = None, required: bool = False
):
    """Resolve pending bytes only when tensor/storage ownership is unique."""
    with _LOCK:
        keys = _matching_pending_keys(tensor, invocation)
        if len(keys) == 1:
            return _payload_tuple(_PENDING_PAYLOADS[keys[0]])
    if len(keys) > 1 and required:
        raise RuntimeError(
            "tensor has pending SM90 payloads from multiple invocations; "
            "pass invocation explicitly"
        )
    return None


def _pop_pending_payload(
    tensor: Any, *, invocation: SM90Invocation | None = None, required: bool = False
) -> PendingSM90Payload | None:
    with _LOCK:
        keys = _matching_pending_keys(tensor, invocation)
        if len(keys) == 1:
            payload = _PENDING_PAYLOADS.pop(keys[0])
            _detach_pending(payload)
            return payload
    if len(keys) > 1 and required:
        raise RuntimeError(
            "tensor has pending SM90 payloads from multiple invocations; "
            "pass invocation explicitly"
        )
    return None


def pop_pending_sm90_payload(
    tensor: Any, *, invocation: SM90Invocation | None = None, required: bool = False
):
    """Pop a uniquely owned pending payload through a storage alias."""
    return _payload_tuple(
        _pop_pending_payload(tensor, invocation=invocation, required=required)
    )


def has_sm90_native_carriers() -> bool:
    """Whether any retained/live Hopper-native invocation can claim new bytes."""
    with _LOCK:
        return any(is_sm90_native_carrier(carrier) for carrier in _REGISTRY.values())


def claim_pending_sm90_carrier(
    tensor: Any,
    invocation: SM90Invocation,
) -> SM90Carrier | None:
    """Promote an unscoped communication payload into its operand invocation."""
    pending = consume_sm90_payload(tensor, invocation=invocation)
    if pending is not None:
        data, scales, group_size, role, metadata = pending
        return register_sm90_carrier(
            tensor,
            data=data,
            scales=scales,
            role=role,
            metadata={"group_size": group_size, **metadata},
            invocation=invocation,
            replace=True,
        )
    return get_sm90_carrier(tensor, invocation=invocation)


def _is_native_invocation(invocation: SM90Invocation | None) -> bool:
    if invocation is None:
        return False
    value = getattr(invocation.backend, "value", invocation.backend)
    return value == "hopper_native"


def is_sm90_native_active() -> bool:
    return _is_native_invocation(current_sm90_invocation())


def is_sm90_native_carrier(carrier: SM90Carrier | None) -> bool:
    """Accept retained backward carriers after the forward scope has exited."""
    if carrier is None:
        return False
    invocation = carrier.metadata.get("invocation")
    return (
        isinstance(invocation, SM90Invocation)
        and invocation.invocation_id == carrier.invocation_id
        and _is_native_invocation(invocation)
    )


def _exact_sm90(device: Any = None, *, capability: tuple[int, int] | None = None) -> bool:
    if capability is not None:
        return int(capability[0]) == 9
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device_type = getattr(device, "type", None)
        if device_type is not None and device_type != "cuda":
            return False
        return int(torch.cuda.get_device_capability(device)[0]) == 9
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


@contextmanager
def use_sm90_invocation(invocation: SM90Invocation | SM90Carrier):
    """Re-enter one retained invocation while a deferred callback executes."""
    invocation_id = _invocation_id(invocation)
    if isinstance(invocation, SM90Invocation):
        active = invocation
    else:
        active = invocation.metadata.get("invocation")
    if active is None or active.invocation_id != invocation_id:
        raise RuntimeError("SM90 invocation metadata is no longer available")
    token = _CURRENT_INVOCATION.set(active)
    try:
        yield active
    finally:
        _CURRENT_INVOCATION.reset(token)
        _collect_invocation(invocation_id, force=False)


@contextmanager
def sm90_invocation_scope(
    fp8_config: Any,
    activation: Any,
    backend: Any = _UNSET,
    *,
    device: Any = None,
    capability: tuple[int, int] | None = None,
) -> Iterator[SM90Invocation | None]:
    """Activate one isolated runtime only when the device is exact SM90."""
    if backend is _UNSET:
        backend = getattr(fp8_config, "fp8_backend", None) or "hopper_native"
    backend_value = getattr(backend, "value", backend)
    if backend_value != "hopper_native" or not _exact_sm90(
        device, capability=capability
    ):
        yield None
        return
    snapshot = SM90FP8ConfigSnapshot(
        enabled=bool(getattr(fp8_config, "enabled", True)),
        fused_gated=bool(getattr(fp8_config, "fused_gated", True)),
        save_z_fp8=bool(getattr(fp8_config, "save_z_fp8", False)),
        recompute_z=bool(getattr(fp8_config, "recompute_z", False)),
        fuse_y1_quant=bool(getattr(fp8_config, "fuse_y1_quant", False)),
        epilogue_quant=bool(getattr(fp8_config, "epilogue_quant", True)),
        fp8_wgrad=bool(getattr(fp8_config, "fp8_wgrad", False)),
        fuse_y1_bf16_trunc=bool(getattr(fp8_config, "fuse_y1_bf16_trunc", False)),
        swiglu_clamp_value=float(getattr(fp8_config, "swiglu_clamp_value", 0.0)),
        activation_group_size=int(fp8_config.activation_group_size),
        weight_block_size=int(fp8_config.weight_block_size),
        fp8_backend=backend,
    )
    invocation = SM90Invocation(
        invocation_id=next(_INVOCATION_IDS),
        fp8_config=snapshot,
        activation_group_size=snapshot.activation_group_size,
        weight_block_size=snapshot.weight_block_size,
        activation=activation,
        backend=backend,
    )
    token: Token[SM90Invocation | None] = _CURRENT_INVOCATION.set(invocation)
    failed = False
    try:
        yield invocation
    except BaseException:
        failed = True
        raise
    finally:
        _CURRENT_INVOCATION.reset(token)
        with _LOCK:
            _RETAINED_INVOCATIONS[invocation.invocation_id] = invocation
        if failed:
            release_sm90_invocation(invocation)
        else:
            _collect_invocation(invocation.invocation_id, force=False)


def _remove_key(key: tuple[Any, ...], carrier: SM90Carrier) -> None:
    if _REGISTRY.get(key) is carrier:
        _REGISTRY.pop(key, None)
    carrier._keys.discard(key)
    keys = _INVOCATION_KEYS.get(carrier.invocation_id)
    if keys is not None:
        keys.discard(key)
        if not keys:
            _INVOCATION_KEYS.pop(carrier.invocation_id, None)


def _owner_finalized(key: tuple[Any, ...], owner_id: int) -> None:
    invocation_id = int(key[0])
    with _LOCK:
        carrier = _REGISTRY.get(key)
        if carrier is None:
            return
        carrier._owners.discard(owner_id)
        carrier._finalizers.pop(owner_id, None)
        object_key = (carrier.invocation_id, "object", owner_id)
        if object_key in carrier._keys:
            _remove_key(object_key, carrier)
        if not carrier._owners and carrier.retain_count == 0:
            for carrier_key in tuple(carrier._keys):
                _remove_key(carrier_key, carrier)
    _collect_invocation(invocation_id, force=False)


def _bind_owner(key: tuple[Any, ...], tensor: Any, carrier: SM90Carrier) -> None:
    owner_id = id(tensor)
    if owner_id in carrier._owners:
        return
    try:
        finalizer = weakref.finalize(tensor, _owner_finalized, key, owner_id)
    except (TypeError, RuntimeError):
        return
    carrier._owners.add(owner_id)
    carrier._finalizers[owner_id] = finalizer


def register_sm90_carrier(
    tensor: Any,
    *,
    data: Any = None,
    scales: Any = None,
    role: str,
    orientation: str = "identity",
    recipe: PayloadRecipe | None = None,
    metadata: Mapping[str, Any] | None = None,
    invocation: SM90Invocation | None = None,
    replace: bool = False,
) -> SM90Carrier:
    """Register native payload for a BF16 tensor in the active invocation."""
    invocation = invocation or current_sm90_invocation()
    if invocation is None:
        raise RuntimeError("register_sm90_carrier requires an active SM90 invocation")
    key = _registry_key(invocation.invocation_id, tensor)
    with _LOCK:
        carrier = _REGISTRY.get(key)
        if carrier is not None and not replace:
            if data is not None and carrier.data is not None and data is not carrier.data:
                raise RuntimeError("SM90 carrier storage is already registered")
            if data is not None:
                carrier.data = data
            if scales is not None:
                carrier.scales = scales
            if recipe is not None:
                carrier.recipe = recipe
            if metadata:
                carrier.metadata.update(metadata)
        else:
            if carrier is not None:
                _remove_key(key, carrier)
            carrier_metadata = {"invocation": invocation, **dict(metadata or {})}
            source = carrier_metadata.get("source")
            if source is not None and not isinstance(
                source, (WeakTensorDescriptor, weakref.ReferenceType)
            ):
                carrier_metadata["source"] = _weak_tensor_descriptor(
                    source, f"{role} source"
                )
            if role in {"z", "y1"}:
                carrier_metadata.setdefault(
                    "source", _weak_tensor_descriptor(tensor, f"{role} source")
                )
            carrier = SM90Carrier(
                invocation_id=invocation.invocation_id,
                role=str(role),
                orientation=str(orientation),
                data=data,
                scales=scales,
                recipe=recipe,
                metadata=carrier_metadata,
            )
            _REGISTRY[key] = carrier
            carrier._keys.add(key)
            _INVOCATION_KEYS.setdefault(invocation.invocation_id, set()).add(key)
        _bind_owner(key, tensor, carrier)
        return carrier


def alias_sm90_carrier(
    source: Any,
    alias: Any,
    *,
    orientation: str | None = None,
    invocation: SM90Invocation | None = None,
) -> SM90Carrier:
    """Associate a copied/rebased tensor object with an existing carrier."""
    carrier = get_sm90_carrier(source, invocation=invocation, required=True)
    assert carrier is not None
    invocation_id = carrier.invocation_id
    key = _registry_key(invocation_id, alias)
    with _LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not carrier:
            raise RuntimeError("SM90 carrier alias collides with another payload")
        _REGISTRY[key] = carrier
        carrier._keys.add(key)
        _INVOCATION_KEYS.setdefault(invocation_id, set()).add(key)
        _bind_owner(key, alias, carrier)
        if orientation is not None:
            carrier.orientation = str(orientation)
        return carrier


def get_sm90_carrier(
    tensor: Any,
    *,
    invocation: SM90Invocation | None = None,
    required: bool = False,
    materialize: bool = False,
) -> SM90Carrier | None:
    """Resolve a tensor only when its invocation association is unambiguous."""
    invocation = invocation or current_sm90_invocation()
    carrier = None
    if invocation is not None:
        invocation_id = invocation.invocation_id
        with _LOCK:
            carrier = _REGISTRY.get(_object_key(invocation_id, tensor))
            if carrier is None:
                carrier = _REGISTRY.get(_registry_key(invocation_id, tensor))
    else:
        suffixes = {view_identity(tensor), ("object", id(tensor))}
        with _LOCK:
            matches = {
                id(candidate): candidate
                for key, candidate in _REGISTRY.items()
                if key[1:] in suffixes
            }
        if len(matches) == 1:
            carrier = next(iter(matches.values()))
        elif len(matches) > 1 and required:
            raise RuntimeError(
                "tensor belongs to multiple SM90 invocations; pass invocation explicitly"
            )
    if carrier is None and required:
        raise RuntimeError("tensor has no SM90 carrier in the current invocation")
    if carrier is not None and materialize:
        carrier.materialize()
    return carrier


def get_sm90_storage_carriers(
    tensor: Any, *, invocation: SM90Invocation | None = None
) -> tuple[SM90Carrier, ...]:
    """Return all carriers on a base storage for explicit view/orientation handling."""
    invocation = invocation or current_sm90_invocation()
    if invocation is None:
        return ()
    storage_key = storage_identity(tensor)
    with _LOCK:
        return tuple({
            id(carrier): carrier
            for key, carrier in _REGISTRY.items()
            if key[0] == invocation.invocation_id
            and len(key) > 3
            and key[1] == "view"
            and key[2] == storage_key
        }.values())


def find_sm90_storage_carriers(tensor: Any) -> tuple[SM90Carrier, ...]:
    """Return uniquely live carriers for a storage, independent of active scope."""
    storage_key = storage_identity(tensor)
    with _LOCK:
        return tuple({
            id(carrier): carrier
            for key, carrier in _REGISTRY.items()
            if len(key) > 3
            and key[1] == "view"
            and key[2] == storage_key
        }.values())


def is_sm90_carrier(tensor: Any) -> bool:
    return get_sm90_carrier(tensor) is not None


def retain_sm90_carrier(tensor_or_carrier: Any, *, deferred_wgrad: bool = False) -> SM90Carrier:
    """Keep a payload alive after its compatibility tensor is finalized."""
    carrier = (
        tensor_or_carrier
        if isinstance(tensor_or_carrier, SM90Carrier)
        else get_sm90_carrier(tensor_or_carrier, required=True)
    )
    assert carrier is not None
    with _LOCK:
        carrier.retain_count += 1
        if deferred_wgrad:
            carrier.deferred_wgrad_count += 1
    return carrier


def release_sm90_carrier(
    tensor_or_carrier: Any,
    *,
    deferred_wgrad: bool = False,
    discard: bool = False,
) -> None:
    """Release actgrad/deferred-wgrad ownership and collect when unused."""
    carrier = (
        tensor_or_carrier
        if isinstance(tensor_or_carrier, SM90Carrier)
        else get_sm90_carrier(tensor_or_carrier, required=True)
    )
    assert carrier is not None
    with _LOCK:
        if carrier.retain_count <= 0:
            raise RuntimeError("SM90 carrier release without a matching retain")
        if deferred_wgrad:
            if carrier.deferred_wgrad_count <= 0:
                raise RuntimeError("SM90 deferred-wgrad release without retain")
            carrier.deferred_wgrad_count -= 1
        carrier.retain_count -= 1
        if discard and (carrier.retain_count != 0 or carrier._owners):
            raise RuntimeError("cannot discard an SM90 carrier while it is still referenced")
        if discard or (carrier.retain_count == 0 and not carrier._owners):
            for key in tuple(carrier._keys):
                _remove_key(key, carrier)
        invocation_id = carrier.invocation_id
    _collect_invocation(invocation_id, force=False)


@contextmanager
def retained_sm90_carrier(tensor_or_carrier: Any, *, deferred_wgrad: bool = False):
    carrier = retain_sm90_carrier(tensor_or_carrier, deferred_wgrad=deferred_wgrad)
    try:
        yield carrier
    finally:
        release_sm90_carrier(carrier, deferred_wgrad=deferred_wgrad)


def retain_for_deferred_wgrad(*tensors: Any) -> tuple[SM90Carrier, ...]:
    return tuple(retain_sm90_carrier(tensor, deferred_wgrad=True) for tensor in tensors)


def release_deferred_wgrad(*carriers: SM90Carrier) -> None:
    for carrier in carriers:
        release_sm90_carrier(carrier, deferred_wgrad=True)


def retain_sm90_invocation(
    invocation: SM90Invocation, owner: Any | None = None
) -> SM90Invocation:
    """Retain one invocation until an explicit release or owner finalization."""
    invocation_id = invocation.invocation_id
    with _LOCK:
        _RETAINED_INVOCATIONS[invocation_id] = invocation
    if owner is not None:
        owner_id = id(owner)
        try:
            finalizer = weakref.finalize(
                owner, release_sm90_invocation_owner, invocation_id, owner_id
            )
        except (TypeError, RuntimeError) as exc:
            raise TypeError("SM90 invocation owner must support finalization") from exc
        with _LOCK:
            owners = _INVOCATION_OWNERS.setdefault(invocation_id, {})
            previous = owners.get(owner_id)
            if previous is not None and previous.alive:
                previous.detach()
            owners[owner_id] = finalizer
    return invocation


def complete_sm90_invocation(
    invocation: SM90Invocation | int, owner: Any | None = None
) -> None:
    """Mark backward complete; deferred carrier retains may finish later."""
    invocation_id = _invocation_id(invocation)
    if owner is not None:
        release_sm90_invocation_owner(invocation_id, id(owner))
    _collect_invocation(invocation_id, force=False)


def release_sm90_invocation_owner(invocation: SM90Invocation | int, owner_id: int) -> None:
    invocation_id = _invocation_id(invocation)
    with _LOCK:
        owners = _INVOCATION_OWNERS.get(invocation_id)
        if owners is not None:
            finalizer = owners.pop(owner_id, None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()
            if not owners:
                _INVOCATION_OWNERS.pop(invocation_id, None)
    _collect_invocation(invocation_id, force=False)


def wrap_deferred_sm90_call(callback, *tensors: Any):
    """Retain carriers and restore their invocation for one deferred launch."""
    resolved = tuple(
        tensor if isinstance(tensor, SM90Carrier) else get_sm90_carrier(tensor)
        for tensor in tensors
    )
    resolved = tuple(carrier for carrier in resolved if carrier is not None)
    if not resolved:
        return callback
    carriers = tuple(
        retain_sm90_carrier(carrier, deferred_wgrad=True) for carrier in resolved
    )
    invocation_ids = {carrier.invocation_id for carrier in carriers}
    if len(invocation_ids) != 1:
        release_deferred_wgrad(*carriers)
        raise RuntimeError("deferred SM90 operands span multiple invocations")
    completed = False

    def release_once():
        nonlocal completed
        if not completed:
            completed = True
            release_deferred_wgrad(*carriers)

    def run():
        try:
            with use_sm90_invocation(carriers[0]):
                return callback()
        finally:
            release_once()

    weakref.finalize(run, release_once)
    return run


def unique_retained_sm90_invocation() -> SM90Invocation | None:
    """Return the retained invocation only when exactly one can own new bytes."""
    with _LOCK:
        live_ids = {
            carrier.invocation_id
            for carrier in _REGISTRY.values()
            if is_sm90_native_carrier(carrier)
        }
        matches = [
            invocation
            for invocation_id, invocation in _RETAINED_INVOCATIONS.items()
            if invocation_id in live_ids and _is_native_invocation(invocation)
        ]
    return matches[0] if len(matches) == 1 else None


def _collect_invocation(invocation_id: int, *, force: bool) -> None:
    with _LOCK:
        for key in tuple(_INVOCATION_KEYS.get(invocation_id, ())):
            carrier = _REGISTRY.get(key)
            if carrier is None:
                continue
            if force or (carrier.retain_count == 0 and not carrier._owners):
                _remove_key(key, carrier)
        has_pending = any(
            payload.invocation.invocation_id == invocation_id
            for payload in _PENDING_PAYLOADS.values()
        )
        has_owners = bool(_INVOCATION_OWNERS.get(invocation_id))
        if not _INVOCATION_KEYS.get(invocation_id) and not has_pending and not has_owners:
            _RETAINED_INVOCATIONS.pop(invocation_id, None)


def clear_sm90_invocation(invocation: SM90Invocation | int, *, force: bool = False) -> None:
    invocation_id = invocation if isinstance(invocation, int) else invocation.invocation_id
    _collect_invocation(int(invocation_id), force=force)


def release_sm90_invocation(invocation: SM90Invocation | int) -> None:
    """Drop all runtime state after an invocation's backward has completed."""
    invocation_id = invocation if isinstance(invocation, int) else invocation.invocation_id
    invocation_id = int(invocation_id)
    with _LOCK:
        for key in tuple(_INVOCATION_KEYS.get(invocation_id, ())):
            carrier = _REGISTRY.get(key)
            if carrier is None:
                continue
            for finalizer in carrier._finalizers.values():
                finalizer.detach()
            carrier._owners.clear()
            carrier._finalizers.clear()
            carrier.retain_count = 0
            carrier.deferred_wgrad_count = 0
        pending_keys = [
            key for key, payload in _PENDING_PAYLOADS.items()
            if payload.invocation.invocation_id == invocation_id
        ]
        for key in pending_keys:
            payload = _PENDING_PAYLOADS.pop(key)
            _detach_pending(payload)
        owner_finalizers = _INVOCATION_OWNERS.pop(invocation_id, {})
        for finalizer in owner_finalizers.values():
            if finalizer.alive:
                finalizer.detach()
    _collect_invocation(invocation_id, force=True)


def sm90_registry_stats() -> dict[str, int]:
    """Return deduplicated live state counts for lifecycle assertions."""
    with _LOCK:
        return {
            "carriers": len({id(carrier) for carrier in _REGISTRY.values()}),
            "keys": len(_REGISTRY),
            "invocations": len(_INVOCATION_KEYS),
            "retained_invocations": len(_RETAINED_INVOCATIONS),
            "invocation_owners": sum(len(owners) for owners in _INVOCATION_OWNERS.values()),
            "pending_payloads": len(_PENDING_PAYLOADS),
        }


def replace_with_sm90_anchor(tensor: Any) -> Any:
    """Rebase a compatibility tensor onto one element when the proxy supports it.

    Paddle's underline-tensor sharing preserves the destination tensor object
    while installing a one-element zero-stride view. Native Torch has no
    equivalent safe public operation, so it is intentionally left unchanged.
    Kernels must consume the registry payload rather than dereference the
    compatibility tensor after rebasing.
    """
    share = getattr(tensor, "_share_underline_tensor_to", None)
    if not callable(share):
        return tensor
    try:
        import paddle
        anchor = paddle.zeros([1], dtype=tensor.dtype)
        copy_to = getattr(anchor, "_copy_to", None)
        if callable(copy_to) and str(anchor.place) != str(tensor.place):
            anchor = copy_to(tensor.place, False)
        carrier = get_sm90_carrier(tensor)
        if carrier is not None:
            carrier.metadata.setdefault(
                "pre_anchor_source",
                _weak_tensor_descriptor(tensor, f"{carrier.role} pre-anchor source"),
            )
        anchor_view = paddle.as_strided(
            anchor, tensor.shape, [0] * len(tensor.shape), 0
        )
        object_key = _object_key(carrier.invocation_id, tensor)
        if carrier is not None:
            with _LOCK:
                _REGISTRY[object_key] = carrier
                carrier._keys.add(object_key)
                _INVOCATION_KEYS.setdefault(carrier.invocation_id, set()).add(object_key)
        anchor_view._share_underline_tensor_to(tensor)
        if carrier is not None:
            carrier.metadata["anchor"] = anchor
            carrier.metadata["anchor_view"] = anchor_view
        return tensor
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return tensor


def attach_sm90_carrier(
    source: Any,
    destination: Any,
    *,
    orientation: str | None = None,
    anchor: bool = False,
) -> SM90Carrier:
    """Move carrier identity to a PyLayer output/gradient object in place."""
    carrier = alias_sm90_carrier(source, destination, orientation=orientation)
    if anchor:
        replace_with_sm90_anchor(destination)
    return carrier


def attach_sm90_colwise_recipe(
    tensor: Any,
    recipe: PayloadRecipe,
    *,
    expert_offsets: Any,
    gathered: bool = False,
) -> SM90Carrier | None:
    """Install a lazy 1x32 wgrad payload recipe on a registered tensor."""
    carrier = get_sm90_carrier(tensor)
    if carrier is None:
        return None
    carrier.metadata.update({
        "colwise_recipe": recipe,
        "colwise_data": None,
        "colwise_scales": None,
        "colwise_group_size": 32,
        "colwise_layout": "feature_total_k_k_major",
        "colwise_expert_offsets": expert_offsets,
        "colwise_gathered": bool(gathered),
    })
    return carrier


def get_sm90_colwise_payload(tensor: Any, *, required: bool = False):
    """Resolve and lazily materialize ``(E4M3 [feature,K], raw UE8M0)``."""
    carrier = get_sm90_carrier(tensor, required=required)
    if carrier is None:
        return None
    carrier.materialize_colwise()
    data = carrier.metadata.get("colwise_data")
    scales = carrier.metadata.get("colwise_scales")
    if required and (data is None or scales is None):
        raise RuntimeError("tensor carrier has no SM90 wgrad colwise payload")
    return None if data is None or scales is None else (data, scales)


# Short aliases used by reference, expert, and dispatch call sites.
register_carrier = register_sm90_carrier
get_carrier = get_sm90_carrier
alias_carrier = alias_sm90_carrier
retain_carrier = retain_sm90_carrier
release_carrier = release_sm90_carrier
invocation_scope = sm90_invocation_scope
