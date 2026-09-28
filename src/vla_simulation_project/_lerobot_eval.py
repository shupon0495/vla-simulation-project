"""NumPy 2.0 compatibility wrapper for lerobot-eval.

LIBERO-plus (4976dc3) and robosuite use np.float_ and np.fromstring() which
were removed or changed in NumPy 2.0. This module patches them before lerobot
imports so downstream code continues to work.
"""
from __future__ import annotations
import numpy as np
np.float_ = np.float64

_np_frombuffer = np.frombuffer

def _np_fromstring(s, dtype=float, offset=0, count=-1, sep=b'', **kwargs):
    """Re-implement np.fromstring on NumPy 2.

    NumPy 1.x np.fromstring has two modes controlled by the *sep* argument:

    1. Text mode (sep=' ' or sep='\\t'): splits *s* by the separator and
       converts each token to the given dtype.  Used by robosuite to read
       whitespace-separated numbers from XML attributes.

    2. Binary mode (sep= b''): treats *s* as raw bytes and interprets them
       as an array.  Used by LIBERO env_wrapper to decode image blobs.

    NumPy 2 removed np.fromstring.  In binary mode the correct replacement is
    np.frombuffer which accepts the same positional arguments (s, dtype,
    offset, count).  In text mode we must re-implement the split+convert
    logic ourselves.
    """
    buf_kwargs = dict(offset=offset, count=count, **kwargs)

    # Normalise sep to bytes for comparison
    sep_normalized = sep.encode('utf-8') if isinstance(sep, str) else bytes(sep) if sep else b''

    if sep_normalized in (b' ', b'\t'):
        # Text mode: split string by whitespace/delimiter and convert
        if isinstance(s, (bytes, bytearray)):
            parts = s.split(sep_normalized)
        elif isinstance(s, str):
            parts = s.split(sep_normalized.decode('ascii'))
        else:
            parts = str(s).split(sep_normalized.decode('ascii', errors='replace'))
        arr = np.array([float(x) for x in parts], dtype=dtype)
        if count >= 0:
            n = min(count, len(arr))
            return arr[:n]
        return arr
    else:
        # Binary mode: delegate to np.frombuffer (the NumPy 2 replacement)
        return _np_frombuffer(s, dtype=dtype, **buf_kwargs)

np.fromstring = _np_fromstring

from lerobot.scripts.lerobot_eval import main
main()
