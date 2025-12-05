# numpy_compat.py
import numpy as np

# Fix NumPy 2.0 deprecated aliases — MUST be done ONCE at import time
# This prevents AttributeError: `np.float_` was removed in NumPy 2.0
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool_"):
    np.bool_ = np.bool_
if not hasattr(np, "object_"):
    np.object_ = object

# Optional: silence warning if you want
import warnings
warnings.filterwarnings("ignore", message="`np.float_` is a deprecated alias")