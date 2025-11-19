# tests/test_evaluator.py
import numpy as np

def _make_simple_scenario(coordenadas, vento, drone):
    # 4 pontos -> L = base + 2 = 3+2 = 5
    from src.ga import GeneticAlgorithm
    ga = GeneticAlgorithm(coord=coordenadas, vento=vento, drone=drone, n_pop=2, n_gen=1, seed=1, n_workers=1)
    # montar 1 indivíduo simples: rota idx 1,2,3,4,1
    rota = [1, 2, 3, 4, 1]
    # velocidades: usar índices de drone.velocidades
    vels = [drone.velocidades[0]] * (len(rota)-1)
    return rota, vels, ga

def test_avaliar_lote_numba_basic(coordenadas, vento, drone):
    rota, vels, ga = _make_simple_scenario(coordenadas, vento, drone)
    from src.evaluator import avaliar_lote_numba
    # converter para batches de índices
    import numpy as np
    # construir arrays conforme esperado pela função numba (com índices internos)
    rotas_batch = np.zeros((1, ga.L), dtype=np.uint16)
    vels_batch = np.zeros((1, ga.L-1), dtype=np.uint8)
    for i, rid in enumerate(rota):
        rotas_batch[0, i] = ga.coord.idx_map[rid]
    for i, vv in enumerate(vels):
        vels_batch[0, i] = ga.drone.vel_to_idx[int(vv)]
    fits, dists, custos, recs = avaliar_lote_numba(
        rotas_batch, vels_batch, ga.dist_matrix, ga.az_matrix, ga.v_eff_table,
        ga.vento_idx_map, float(ga.drone.autonomia_real), int(ga.drone.tempo_pouso_seg),
        int(ga.vento_array.shape[0])
    )
    assert fits.shape[0] == 1
    assert dists[0] >= 0.0
    assert custos[0] >= 0.0
    assert recs.dtype == np.int32

def test_avaliar_branch_bateria_excedida(coordenadas, vento, drone):
    # modificar drone para autonomia muito pequena para forçar pouso
    drone.autonomia_real = 1.0  # segundos muito pequenos
    rota, vels, ga = _make_simple_scenario(coordenadas, vento, drone)
    from src.evaluator import avaliar_lote_numba
    import numpy as np
    rotas_batch = np.zeros((1, ga.L), dtype=np.uint16)
    vels_batch = np.zeros((1, ga.L-1), dtype=np.uint8)
    for i, rid in enumerate(rota):
        rotas_batch[0, i] = ga.coord.idx_map[rid]
    for i, vv in enumerate(vels):
        vels_batch[0, i] = ga.drone.vel_to_idx[int(vv)]
    fits, dists, custos, recs = avaliar_lote_numba(
        rotas_batch, vels_batch, ga.dist_matrix, ga.az_matrix, ga.v_eff_table,
        ga.vento_idx_map, float(ga.drone.autonomia_real), int(ga.drone.tempo_pouso_seg),
        int(ga.vento_array.shape[0])
    )
    # pousos_forcados deve ser >= 1 por causa de autonomia reduzida
    assert recs[0] >= 1
    assert custos[0] > 0.0
