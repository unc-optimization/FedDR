from .l1_norm import L1_Norm
from .l2_norm_squared import L2_Norm_Squared
from .l2_norm import L2_Norm
from .linf_norm import LInf_Norm
from .empty_func import Empty_Function

REGISTRY = {}

REGISTRY["l1_norm"] = L1_Norm
REGISTRY["l2_norm_squared"] = L2_Norm_Squared
REGISTRY["l2_norm"] = L2_Norm
REGISTRY["linf_norm"] = LInf_Norm
REGISTRY["none"] = Empty_Function