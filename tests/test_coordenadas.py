# tests/test_coordenadas.py
import math
import numpy as np

def test_coordenadas_build_and_distance_azimute(coordenadas):
    # ids devem começar em 1 e ter tamanho 4 (conforme fixture)
    assert len(coordenadas.ids) == 4
    # distância entre um ponto e ele mesmo = 0
    d = coordenadas.distancia(1, 1)
    assert abs(d - 0.0) < 1e-6
    # distancia simétrica
    d12 = coordenadas.distancia(1, 2)
    d21 = coordenadas.distancia(2, 1)
    assert abs(d12 - d21) < 1e-6
    # azimute complementar: az[i,j] + 180 == az[j,i] (mod 360)
    a12 = coordenadas.azimute(1, 2)
    a21 = coordenadas.azimute(2, 1)
    assert abs(((a12 + 180) % 360) - a21) < 1e-6

def test_haversine_values():
    # comparar haversine manual com aproximado conhecido
    from src.coordenadas import Coordenadas
    import tempfile, os
    # criar csv com 2 pontos próximos e instanciar
    path = os.path.join(tempfile.gettempdir(), "coords_tmp.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("cep,latitude,longitude\n")
        f.write("a,0.0,0.0\n")
        f.write("b,0.0,1.0\n")
    c = Coordenadas(path)
    d = c.distancia(1, 2)
    # distância entre (0,0) e (0,1) ~ 111.319 km
    assert abs(d - 111.319) < 0.5
    os.remove(path)
