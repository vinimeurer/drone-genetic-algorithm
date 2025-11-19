# src/v_eff.py
import math
import numpy as np
from typing import Any
from numba import njit

from .constants import AZIMUTE_BIN, AZIMUTE_BINS

@njit(cache=True)
def _build_v_eff_table_numba(vels, az_bins, vento_array):
    n_v = vels.shape[0]
    n_az = az_bins.shape[0]
    n_w = vento_array.shape[0] * vento_array.shape[1]
    table = np.zeros((n_v, n_az, n_w), dtype=np.float32)
    idx = 0
    for d in range(vento_array.shape[0]):
        for h in range(vento_array.shape[1]):
            w_vel = vento_array[d, h, 0]
            w_dir = vento_array[d, h, 1]
            dir_to = (w_dir + 180.0) % 360.0
            for iv in range(n_v):
                v = vels[iv]
                for ia in range(n_az):
                    az = az_bins[ia]
                    ang_d = az * (np.pi / 180.0)
                    ang_w = dir_to * (np.pi / 180.0)
                    vx = v * np.cos(ang_d) + w_vel * np.cos(ang_w)
                    vy = v * np.sin(ang_d) + w_vel * np.sin(ang_w)
                    val = math.hypot(vx, vy)
                    if val < 0.1:
                        val = 0.1
                    table[iv, ia, idx] = val
            idx += 1
    return table

def build_v_eff_table(vels: np.ndarray, vento_array: np.ndarray) -> np.ndarray:
    return _build_v_eff_table_numba(vels.astype(np.float32), AZIMUTE_BINS.astype(np.float32), vento_array)
