"""NumPy 2.0 compatibility wrapper for lerobot-eval.

LIBERO-plus (4976dc3) uses np.float_ and np.fromstring() which were
removed in NumPy 2.0. This module patches them before lerobot imports.
"""
from __future__ import annotations
import numpy as np
np.float_ = np.float64
np.fromstring = np.frombuffer  # type: ignore[assignment]

from lerobot.scripts.lerobot_eval import main
main()
