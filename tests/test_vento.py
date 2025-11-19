# tests/test_vento.py
def test_vento_parsing_and_get(vento):
    # teste dia válido
    v = vento.get_vento(1, 0)
    assert "vel_kmh" in v and "direcao_deg" in v
    # hora wrap-around
    v2 = vento.get_vento(1, 24)
    assert v2 == vento.get_vento(1, 0)
    # dia fora do range retorna zeros
    v3 = vento.get_vento(1000, 5)
    assert v3["vel_kmh"] == 0.0 and v3["direcao_deg"] == 0.0
