"""ERNIE integration layer for SonicMoE."""

from sonicmoe.ernie_compat.deepep_metadata import (  # noqa: F401
    deepep_to_sonic_metadata,
    deepep_topk_to_sonic_metadata,
)
from sonicmoe.ernie_compat.mlp_node_v2 import (  # noqa: F401
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
    stack_ernie_w1,
    stack_ernie_w2,
)
