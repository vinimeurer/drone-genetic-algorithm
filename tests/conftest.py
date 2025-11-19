# tests/conftest.py
import sys
import os
import types
import tempfile
import shutil
import pytest

# --- MOCK NUMBA (njit) antes de importar os módulos src ---
# Criamos um módulo "numba" simples que apenas retorna a função original
# quando usado como decorator njit(...).
dummy_numba = types.SimpleNamespace()
def njit(*args, **kwargs):
    # se usado como @njit(...) -> retorna decorator
    if args and callable(args[0]):
        # usado sem parâmetros: @njit
        return args[0]
    def _decorator(fn):
        return fn
    return _decorator
dummy_numba.njit = njit
# Inserir no sys.modules apenas se já não existir.
# Mesmo se existir numba real, substituímos para garantir comportamento consistente.
sys.modules['numba'] = dummy_numba

# --- garantir src no path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Fixtures: arquivos temporários CSV, instâncias de Coordenadas, Vento, Drone
@pytest.fixture(scope="session")
def tmp_dir():
    d = tempfile.mkdtemp(prefix="tests_")
    yield d
    shutil.rmtree(d)

@pytest.fixture(scope="session")
def coord_csv(tmp_dir):
    path = os.path.join(tmp_dir, "coordenadas.csv")
    # criamos 4 pontos simples
    with open(path, "w", encoding="utf-8") as f:
        f.write("cep,latitude,longitude\n")
        f.write("00000-000, -23.550520, -46.633308\n")  # São Paulo (1)
        f.write("11111-111, -22.906847, -43.172896\n")  # Rio (2)
        f.write("22222-222, -25.427, -49.273\n")        # Curitiba (3)
        f.write("33333-333, -15.794229, -47.882166\n")  # Brasília (4)
    return path

@pytest.fixture(scope="session")
def vento_csv(tmp_dir):
    path = os.path.join(tmp_dir, "vento.csv")
    # simples: dia,hora,vel_kmh,direcao_deg
    with open(path, "w", encoding="utf-8") as f:
        f.write("dia,hora,vel_kmh,direcao_deg\n")
        # Preencher 2 dias x 24h úteis
        for d in range(1, 3):
            for h in range(24):
                f.write(f"{d},{h},{5.0 + h % 3},{(h*15) % 360}\n")
    return path

@pytest.fixture
def coordenadas(coord_csv):
    from src.coordenadas import Coordenadas
    return Coordenadas(coord_csv)

@pytest.fixture
def vento(vento_csv):
    from src.vento import Vento
    return Vento(vento_csv, max_dias=31)

@pytest.fixture
def drone():
    from src.drone import Drone
    return Drone()

# Provide a fast GA instance for testing with small sizes
@pytest.fixture
def ga_instance(coordenadas, vento, drone):
    from src.ga import GeneticAlgorithm
    # criar GA pequeno para executar rapidamente
    ga = GeneticAlgorithm(
        coord=coordenadas, vento=vento, drone=drone,
        n_pop=4, n_gen=3, elitismo=0.25, taxa_mut_inicial=0.5,
        taxa_mut_final=0.5, seed=1, n_workers=1, max_stagnation=10
    )
    return ga
