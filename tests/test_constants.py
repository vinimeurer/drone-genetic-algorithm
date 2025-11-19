# tests/test_constants.py
from src import constants

def test_constants_defaults():
    assert isinstance(constants.DEFAULT_N_POP, int)
    assert constants.DEFAULT_N_POP > 0
    assert isinstance(constants.DRONE_VELOCIDADES, list)
    assert len(constants.AZIMUTE_BINS) > 0
    # AZIMUTE_BINS deve estar em [0, 350] com passo AZIMUTE_BIN
    assert constants.AZIMUTE_BINS[0] == 0.0
    assert constants.AZIMUTE_BINS[-1] < 360
