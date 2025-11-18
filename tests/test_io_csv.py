# tests/test_io_csv.py
import os
import csv
from src.io_csv import reavaliar_preciso, gerar_csv_final

def test_reavaliar_preciso_and_gerar_csv(tmp_path, coordenadas, vento, drone):
    # montar info compatível com expectativa: (rota, velocidades, dist, custo, recs, recargas)
    rota = [1, 2, 3, 4, 1]
    vels = [drone.velocidades[0]] * (len(rota)-1)
    # usar reavaliar_preciso (preciso) para obter formato correto
    info = (rota, vels, 0.0, 0.0, 0, [])
    fit, result = reavaliar_preciso(info, coordenadas, vento, drone, max_dias=31)
    # fitness deve estar no intervalo
    assert 0.01 <= fit <= 0.99
    assert isinstance(result, tuple)
    # gerar csv
    out = tmp_path / "rota_test.csv"
    gerar_csv_final(result, coordenadas, vento, str(out))
    assert out.exists()
    # checar cabeçalho
    with open(out, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r)
        assert "CEP_inicial" in header
