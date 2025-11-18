# tests/test_drone.py
def test_drone_basic(drone):
    assert hasattr(drone, "autonomia_base")
    assert drone.autonomia_real == drone.autonomia_base * drone.fator_curitiba
    assert hasattr(drone, "velocidades")
    assert isinstance(drone.vel_to_idx, dict)
    # verificar que todas velocidades têm índice
    for v in drone.velocidades:
        assert v in drone.vel_to_idx
