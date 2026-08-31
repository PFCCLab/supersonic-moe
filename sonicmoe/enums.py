# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from enum import Enum


LIBRARY_NAME = "sonicmoe"
TENSORMAP = "tensormap"


class KernelBackendMoE(Enum):
    scattermoe = "scattermoe"
    torch = "torch"
    sonicmoe = "sonicmoe"


class ActivationType(Enum):
    SWIGLU = "swiglu"
    SWIGLU_PRECISE = "swiglu_precise"
    GEGLU = "geglu"
    REGLU = "reglu"
    # SiTU-GLU: situ(g, beta) * up_act(u, linear_beta).  The two beta scalars are
    # NOT part of the enum; they live in SonicMoEConfig / _FP8Config and are
    # encoded into the GEMM-level activation string (see quack_utils/
    # activation_situ.py::encode_situ_activation).
    SITU_GLU = "situ_glu"

    RELU_SQ = "relu_sq"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


def is_glu(activation_type: ActivationType):
    if activation_type in [
        ActivationType.SWIGLU,
        ActivationType.REGLU,
        ActivationType.GEGLU,
        ActivationType.SWIGLU_PRECISE,
        ActivationType.SITU_GLU,
    ]:
        return True
    # Encoded SiTU activation strings ("situ_glu:b=4.0:lb=25.0") also reach
    # here: they carry the beta constants that must survive down to the GEMM
    # epilogue, so they are not collapsed to the bare enum member.  Prefix-match
    # rather than importing quack_utils.activation_situ, which pulls in cutlass.
    if isinstance(activation_type, str):
        return activation_type.split(":", 1)[0] == ActivationType.SITU_GLU.value
    return False
