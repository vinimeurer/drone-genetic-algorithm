# src/evaluator.py
import math
import numpy as np
from typing import Tuple
from numba import njit
from .constants import AZIMUTE_BIN

# mantemos a mesma assinatura; Numba accelerate se disponível
@njit(cache=True)
def avaliar_lote_numba(
    rotas_batch, vels_batch, dist_matrix, az_matrix, v_eff_table,
    vento_idx_map, autonomia_real, tempo_pouso_seg, max_dias
):
    B, L = rotas_batch.shape
    fits = np.zeros(B, dtype=np.float32)
    dists = np.zeros(B, dtype=np.float32)
    custos = np.zeros(B, dtype=np.float32)
    recs = np.zeros(B, dtype=np.int32)

    for b in range(B):
        custo_total = 0.0
        distancia_total = 0.0
        bateria = autonomia_real
        dia = 1
        hora_atual_seg = 6 * 3600
        pousos_forcados = 0

        for i in range(L - 1):
            i1 = int(rotas_batch[b, i])
            i2 = int(rotas_batch[b, i + 1])
            dist = float(dist_matrix[i1, i2])
            distancia_total += dist
            az = float(az_matrix[i1, i2])
            az_bin = int(az // AZIMUTE_BIN)
            if az_bin < 0:
                az_bin = 0
            if az_bin >= v_eff_table.shape[1]:
                az_bin = v_eff_table.shape[1] - 1
            v_idx = int(vels_batch[b, i])
            w_idx = vento_idx_map[(dia - 1) * 24 + (hora_atual_seg // 3600) % 24]
            if v_idx < 0:
                v_idx = 0
            if v_idx >= v_eff_table.shape[0]:
                v_idx = v_eff_table.shape[0] - 1
            if w_idx < 0:
                w_idx = 0
            if w_idx >= v_eff_table.shape[2]:
                w_idx = v_eff_table.shape[2] - 1

            v_eff = v_eff_table[v_idx, az_bin, w_idx]
            tempo_seg = int(math.ceil(dist * 3600.0 / v_eff))

            if tempo_seg > bateria:
                excesso_min = (tempo_seg - bateria) / 60.0
                custo_total += 15.0 + 3.0 * excesso_min
                pousos_forcados += 1
                bateria = autonomia_real
                hora_atual_seg += tempo_pouso_seg
            else:
                bateria -= tempo_seg

            custo_total += (tempo_seg / 60.0) + (dist * 40.0)
            hora_atual_seg += tempo_seg + tempo_pouso_seg

            if hora_atual_seg >= 19 * 3600:
                dia += 1
                hora_atual_seg = 6 * 3600
                if dia > max_dias:
                    custo_total += 1e7
                    break

        custo_total += 5.0 * pousos_forcados
        fitness = 1.0 / (1.0 + (custo_total / 60000.0) ** 1.8)
        if fitness < 0.01:
            fitness = 0.01
        if fitness > 0.99:
            fitness = 0.99
        fits[b] = round(fitness, 5)
        dists[b] = distancia_total
        custos[b] = custo_total
        recs[b] = pousos_forcados

    return fits, dists, custos, recs