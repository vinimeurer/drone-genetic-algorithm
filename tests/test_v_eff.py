# tests/test_v_eff.py
import numpy as np
from src import v_eff

def test_build_v_eff_table_basic(drone):
    # criar vento_array pequeno (2 dias x 24h x 2)
    vento_array = np.zeros((2, 24, 2), dtype=np.float32)
    # vento leve com direção 0
    vento_array[:, :, 0] = 3.0
    vento_array[:, :, 1] = 0.0
    vels = np.array(drone.velocidades, dtype=np.float32)
    table = v_eff.build_v_eff_table(vels, vento_array)
    # dimensões: n_v x n_az x (n_dias*24)
    assert table.shape[0] == len(vels)
    assert table.shape[1] == v_eff.AZIMUTE_BINS.shape[0]
    assert table.shape[2] == vento_array.shape[0] * vento_array.shape[1]
    # valores positivos e não inferiores a 0.1
    assert (table >= 0.1).all()
