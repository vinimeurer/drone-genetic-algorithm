# ag_puro_v3_otimizado.py
# -*- coding: utf-8 -*-
"""
Versão otimizada do AG PURO V3 para roteamento de drone.
Principais mudanças:
- pré-computação trigonométrica
- tempo representado em segundos (evita datetime)
- acesso direto a matrizes numpy
- avaliação em blocos (chunked) usando ThreadPoolExecutor por padrão
- opção de usar numba se estiver disponível (set USE_NUMBA = True)
"""

import csv
import itertools
import math
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import os
import time
import sys

# If you have numba installed and want extreme speedups, set USE_NUMBA = True
USE_NUMBA = True
try:
    if USE_NUMBA:
        from numba import njit
except Exception:
    USE_NUMBA = False

# =============================
# CLASSES PRINCIPAIS (OTIMIZADAS)
# =============================

class Coordenadas:
    def __init__(self, arquivo_csv: str):
        self.df = pd.read_csv(arquivo_csv).reset_index(drop=True).copy()
        self.df["ID"] = list(range(1, len(self.df) + 1))
        # Expect columns 'latitude' and 'longitude', 'cep' optional
        self.coordenadas: Dict[int, Dict[str, float]] = {
            int(row["ID"]): {
                "cep": row.get("cep", ""),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
            }
            for _, row in self.df.iterrows()
        }
        self.ids = sorted(self.coordenadas.keys())
        self.n = len(self.ids)
        self.idx_map = {id_: i for i, id_ in enumerate(self.ids)}
        # arrays
        self.lats = np.array([self.coordenadas[i]["lat"] for i in self.ids], dtype=np.float64)
        self.lons = np.array([self.coordenadas[i]["lon"] for i in self.ids], dtype=np.float64)
        self.ceps = [self.coordenadas[i].get("cep", "") for i in self.ids]
        # precompute radians, sin/cos of lat
        self.lat_rad = np.radians(self.lats)
        self.lon_rad = np.radians(self.lons)
        self.cos_lat = np.cos(self.lat_rad)
        self.sin_lat = np.sin(self.lat_rad)
        # distance and azimuth matrices
        self._build_matrices()

    def _haversine_np(self, i, j):
        # uses radians arrays
        R = 6371.0
        phi1 = self.lat_rad[i]; phi2 = self.lat_rad[j]
        dphi = phi2 - phi1
        dlambda = self.lon_rad[j] - self.lon_rad[i]
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def _azimute_np(self, i, j):
        dlon = self.lon_rad[j] - self.lon_rad[i]
        lat1_r = self.lat_rad[i]; lat2_r = self.lat_rad[j]
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        az = math.degrees(math.atan2(x, y))
        return (az + 360.0) % 360.0

    def _build_matrices(self):
        n = self.n
        self.dist_matrix = np.zeros((n, n), dtype=np.float32)
        self.az_matrix = np.zeros((n, n), dtype=np.float32)
        # compute upper triangle
        for i, j in itertools.combinations(range(n), 2):
            d = self._haversine_np(i, j)
            az = self._azimute_np(i, j)
            self.dist_matrix[i, j] = d
            self.dist_matrix[j, i] = d
            self.az_matrix[i, j] = az
            self.az_matrix[j, i] = (az + 180.0) % 360.0

    def distancia_idx(self, idx1: int, idx2: int) -> float:
        return float(self.dist_matrix[idx1, idx2])

    def azimute_idx(self, idx1: int, idx2: int) -> float:
        return float(self.az_matrix[idx1, idx2])


class Vento:
    def __init__(self, arquivo_csv: str):
        df = pd.read_csv(arquivo_csv)
        self.vento: Dict[int, Dict[int, Dict[str, float]]] = {}
        for _, row in df.iterrows():
            dia = int(row["dia"])
            hora = int(row["hora"])
            if dia not in self.vento:
                self.vento[dia] = {}
            self.vento[dia][hora] = {
                "vel_kmh": float(row.get("vel_kmh", 0.0)),
                "direcao_deg": float(row.get("direcao_deg", 0.0)),
            }

    def get_vento(self, dia: int, hora: int) -> Dict[str, float]:
        return self.vento.get(dia, {}).get(hora, {"vel_kmh": 0.0, "direcao_deg": 0.0})


class Drone:
    def __init__(self, autonomia_base: float = 5000.0, fator_curitiba: float = 0.93,
                 velocidades: List[int] = None):
        self.autonomia_base = autonomia_base
        self.fator_curitiba = fator_curitiba
        self.autonomia_real = self.autonomia_base * self.fator_curitiba
        if velocidades is None:
            self.base_vels = list(range(36, 100, 4))
            self.velocidades = self.base_vels
        else:
            self.velocidades = velocidades
        self.tempo_pouso_seg = 72

    def tempo_voo_seg(self, distancia_km: float, v_kmh: float) -> int:
        if v_kmh <= 0:
            return int(1e6)
        tempo = distancia_km * 3600.0 / v_kmh
        return int(math.ceil(tempo))


# =============================
# UTILIDADES
# =============================

def calcular_v_efetiva(v_drone_kmh: float, direcao_voo_deg: float, vento_info: Dict[str, float]) -> float:
    # vetores simples
    v_wind = vento_info.get("vel_kmh", 0.0)
    # vento informado é direção de onde vem; converter para direção para onde vai
    dir_wind_to = (vento_info.get("direcao_deg", 0.0) + 180.0) % 360.0
    ang_drone = math.radians(direcao_voo_deg)
    ang_wind = math.radians(dir_wind_to)
    vx = v_drone_kmh * math.cos(ang_drone) + v_wind * math.cos(ang_wind)
    vy = v_drone_kmh * math.sin(ang_drone) + v_wind * math.sin(ang_wind)
    v_eff = math.hypot(vx, vy)
    return max(v_eff, 0.1)


# =============================
# AVALIAÇÃO COM DISTÂNCIA + TEMPO (OTIMIZADA)
# =============================

def avaliar_rota_individual_fast(individual, coord: Coordenadas, vento: Vento, drone: Drone, max_dias=7):
    rota_ids, velocidades = individual
    # transform ids -> indices (integers 0..n-1) for fast matrix access
    idxs = [coord.idx_map[i] for i in rota_ids]
    custo_total = 0.0
    distancia_total = 0.0
    bateria = drone.autonomia_real
    dia = 1
    hora_atual_seg = 6 * 3600  # 06:00:00 em segundos desde 00:00
    pousos_forcados = 0
    recargas = []

    for i in range(len(idxs) - 1):
        a_idx = idxs[i]; b_idx = idxs[i+1]
        dist = float(coord.dist_matrix[a_idx, b_idx])  # km
        distancia_total += dist
        v_chosen = float(velocidades[i])
        az = float(coord.az_matrix[a_idx, b_idx])
        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        v_efetiva = calcular_v_efetiva(v_chosen, az, vento_info)
        tempo_seg = drone.tempo_voo_seg(dist, v_efetiva)

        # Bateria: se tempo maior que bateria => pouso forçado (recarga) antes de partir
        if tempo_seg > bateria:
            excesso_min = (tempo_seg - bateria) / 60.0
            custo_total += 15.0 + 3.0 * excesso_min
            pousos_forcados += 1
            recargas.append((rota_ids[i], "RECARGA FORÇADA"))
            bateria = drone.autonomia_real
            # gastar o tempo do pouso
            hora_atual_seg += drone.tempo_pouso_seg

        # agora voa
        bateria -= tempo_seg
        # custo: tempo (min) + distância ponderada
        custo_total += (tempo_seg / 60.0) + (dist * 40.0)
        hora_atual_seg += tempo_seg + drone.tempo_pouso_seg

        # checar fim do dia (>=19:00)
        if hora_atual_seg >= 19 * 3600:
            dia += 1
            hora_atual_seg = 6 * 3600
            if dia > max_dias:
                custo_total += 1e7
                break

    custo_total += 5.0 * pousos_forcados

    custo_ref = 60000.0
    fitness = 1.0 / (1.0 + (custo_total / custo_ref) ** 1.8)
    fitness = max(0.01, min(0.99, fitness))
    fitness = round(fitness, 5)

    return fitness, (rota_ids, velocidades, distancia_total, custo_total, pousos_forcados, recargas)


# Optional: wrapper to allow numba compilation later if desired
if USE_NUMBA:
    # Numba implementation would require rewriting with numpy arrays and no python objects
    pass

# =============================
# CROSSOVER & MUTAÇÃO (mesmos conceitos, leves otimizações)
# =============================

def ox_crossover(p1, p2, base_id=1):
    size = len(p1)
    if size < 4:
        return p1[:], p2[:]
    a, b = sorted(random.sample(range(1, size - 1), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    pos = b
    for gene in p2[1:-1]:
        if gene not in child[a:b]:
            if pos >= size - 1:
                pos = 1
            child[pos] = gene
            pos += 1
    child[0] = base_id
    child[-1] = base_id
    return child


def crossover_velocidades(v1, v2):
    size = len(v1)
    if size == 0:
        return []
    a, b = sorted(random.sample(range(size), 2))
    child_v = v1[a:b+1]
    rest = [x for x in v2 if x not in child_v]
    random.shuffle(rest)
    out = child_v + rest[:size - len(child_v)]
    # if still short, pad sampling from combined
    while len(out) < size:
        out.append(random.choice(v1 + v2))
    return out


def mutacao_inversao(rota, taxa):
    if random.random() > taxa:
        return rota
    i, j = sorted(random.sample(range(1, len(rota)-1), 2))
    return rota[:i] + rota[i:j+1][::-1] + rota[j+1:]


def mutacao_velocidades(vels, taxa_mut, drone):
    vel_n = vels[:]
    high_vels = drone.velocidades[-6:]
    weights = [0.25] * (len(drone.velocidades) - len(high_vels)) + [0.75] * len(high_vels)
    for i in range(len(vel_n)):
        if random.random() < taxa_mut:
            vel_n[i] = random.choices(drone.velocidades, weights=weights, k=1)[0]
    return vel_n


# =============================
# ALGORITMO GENÉTICO OTIMIZADO
# =============================

class GeneticAlgorithmFast:
    def __init__(self, coord: Coordenadas, vento: Vento, drone: Drone,
                 n_pop=200, n_gen=800, elitismo=0.12,
                 taxa_mut_inicial=0.07, taxa_mut_final=0.35,
                 seed=42, n_workers=None, chunk_size=16, use_process=False):
        self.coord = coord
        self.vento = vento
        self.drone = drone
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.elitismo = elitismo
        self.taxa_mut_inicial = taxa_mut_inicial
        self.taxa_mut_final = taxa_mut_final
        self.seed = seed
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        self.chunk_size = chunk_size
        self.use_process = use_process
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.base = [i for i in self.coord.ids if i != 1]

    def inicializar_populacao(self):
        populacao = []
        for _ in range(self.n_pop):
            perm = random.sample(self.base, len(self.base))
            rota = [1] + perm + [1]
            velocidades = [random.choices(
                self.drone.velocidades,
                weights=[0.25]* (len(self.drone.velocidades)-6) + [0.75]*6,
                k=1)[0] for _ in range(len(rota) - 1)]
            populacao.append((rota, velocidades))
        # add some heuristic seeds (vizinho mais próximo) to help converge faster
        try:
            nn = self.greedy_nearest_neighbor()
            populacao[0] = nn
        except Exception:
            pass
        return populacao

    def greedy_nearest_neighbor(self):
        # return one route using nearest neighbor heuristic (fast)
        remaining = set(self.base)
        cur = 1
        rota = [1]
        while remaining:
            i_cur = self.coord.idx_map[cur]
            # find nearest index among remaining
            best = None; best_d = 1e12
            for r in remaining:
                d = self.coord.dist_matrix[i_cur, self.coord.idx_map[r]]
                if d < best_d:
                    best = r; best_d = d
            rota.append(best)
            remaining.remove(best)
            cur = best
        rota.append(1)
        velocidades = [random.choice(self.drone.velocidades) for _ in range(len(rota)-1)]
        return (rota, velocidades)

    def avaliar_populacao_parallel(self, populacao, executor):
        # chunk populacao to reduce number of futures
        futures = []
        results = []
        pop_len = len(populacao)
        for k in range(0, pop_len, self.chunk_size):
            chunk = populacao[k:k+self.chunk_size]
            futures.append(executor.submit(self._avaliar_bloco, chunk))
        for f in futures:
            results.extend(f.result())
        return results

    def _avaliar_bloco(self, bloco):
        out = []
        for ind in bloco:
            out.append(avaliar_rota_individual_fast(ind, self.coord, self.vento, self.drone))
        return out

    def selecionar_pais(self, avaliacoes, gen):
        k = 5 if gen < self.n_gen * 0.25 else 3
        contestants = random.sample(avaliacoes, min(k, len(avaliacoes)))
        return max(contestants, key=lambda x: x[0])[1]

    def executar(self, verbose=True):
        populacao = self.inicializar_populacao()

        ExecutorClass = ProcessPoolExecutor if self.use_process else ThreadPoolExecutor
        with ExecutorClass(max_workers=self.n_workers) as ex:
            avaliacoes = self.avaliar_populacao_parallel(populacao, ex)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])
            melhor_global = avaliacoes[0]
            estagnado = 0
            if verbose:
                print("=== INICIANDO AG PURO V3 (OTIMIZADO) ===")
            start_time = time.time()
            for gen in range(self.n_gen):
                t = gen / max(1, (self.n_gen - 1))
                taxa_mut_atual = self.taxa_mut_inicial + t * (self.taxa_mut_final - self.taxa_mut_inicial)
                elite_n = max(5, int(self.elitismo * self.n_pop))
                nova_pop = [avaliacoes[i][1][:2] for i in range(elite_n)]

                # diversidade
                for _ in range(4):
                    perm = random.sample(self.base, len(self.base))
                    rota = [1] + perm + [1]
                    rota = mutacao_inversao(rota, taxa_mut_atual * 1.8)
                    velocidades = [random.choices(
                        self.drone.velocidades,
                        weights=[0.2]* (len(self.drone.velocidades)-6) + [0.8]*6,
                        k=1)[0] for _ in range(len(rota)-1)]
                    nova_pop.append((rota, velocidades))

                while len(nova_pop) < self.n_pop:
                    p1 = self.selecionar_pais(avaliacoes, gen)
                    p2 = self.selecionar_pais(avaliacoes, gen)
                    filho_rota = ox_crossover(p1[0], p2[0], base_id=1)
                    filho_vels = crossover_velocidades(p1[1], p2[1])
                    filho_rota = mutacao_inversao(filho_rota, taxa_mut_atual)
                    filho_vels = mutacao_velocidades(filho_vels, taxa_mut_atual, self.drone)
                    nova_pop.append((filho_rota, filho_vels))

                populacao = nova_pop
                avaliacoes = self.avaliar_populacao_parallel(populacao, ex)
                avaliacoes.sort(reverse=True, key=lambda x: x[0])

                if avaliacoes[0][0] > melhor_global[0]:
                    melhor_global = avaliacoes[0]
                    estagnado = 0
                else:
                    estagnado += 1

                # reinício catastrófico (ajustado)
                if estagnado > 80:
                    if verbose:
                        print(f"\n>>> REINÍCIO INTELIGENTE na G{gen+1} <<<")
                    nova_pop = [melhor_global[1][:2]]
                    for _ in range(self.n_pop - 1):
                        perm = random.sample(self.base, len(self.base))
                        rota = [1] + perm + [1]
                        rota = mutacao_inversao(rota, 0.75)
                        velocidades = [random.choices(
                            self.drone.velocidades,
                            weights=[0.15]* (len(self.drone.velocidades)-6) + [0.85]*6,
                            k=1)[0] for _ in range(len(rota)-1)]
                        nova_pop.append((rota, velocidades))
                    populacao = nova_pop
                    avaliacoes = self.avaliar_populacao_parallel(populacao, ex)
                    avaliacoes.sort(reverse=True, key=lambda x: x[0])
                    estagnado = 0

                if verbose and (gen % max(1, int(self.n_gen/50)) == 0 or gen < 10 or gen == self.n_gen-1):
                    best = avaliacoes[0]
                    dist = best[1][2]
                    tempo = best[1][3] / 40.0
                    print(f"G{gen+1:4d} | Fit: {best[0]:.5f} | Dist: {dist:.1f}km | Tempo: ~{tempo:.0f}min | Rec: {best[1][4]} | Stag: {estagnado}")

                # critério de parada antecipado (convergência)
                if estagnado > 400:
                    if verbose:
                        print("Convergência detectada -> interrompendo cedo.")
                    break

            total_time = time.time() - start_time
            if verbose:
                print(f"=== FIM (tempo: {total_time:.1f}s) ===")
        return melhor_global[1], melhor_global[0]


# =============================
# CSV FINAL (corrigido para receber 'vento' como argumento)
# =============================

def gerar_csv_final(info, coord: Coordenadas, vento: Vento, arquivo_saida="melhor_rota_ag_puro_v3_otimizado.csv"):
    rota, velocidades, distancia_total, _, _, recargas = info
    linhas = []
    dia = 1
    hora_atual = 6 * 3600
    drone_tempo_pouso = 72
    recarga_set = {(id_, msg) for id_, msg in recargas}

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        c1 = coord.coordenadas[id1]
        c2 = coord.coordenadas[id2]
        velocidade = velocidades[i]
        a_idx = coord.idx_map[id1]; b_idx = coord.idx_map[id2]
        dist = float(coord.dist_matrix[a_idx, b_idx])
        az = float(coord.az_matrix[a_idx, b_idx])
        vento_info = vento.get_vento(dia, int(hora_atual // 3600))
        v_eff = calcular_v_efetiva(velocidade, az, vento_info)
        tempo_voo = int(math.ceil(dist * 3600.0 / max(v_eff, 0.1)))
        hora_final_seg = hora_atual + tempo_voo
        pouso = "SIM" if (id1, "RECARGA FORÇADA") in recarga_set else "NÃO"
        linhas.append([
            c1["cep"], c1["lat"], c1["lon"],
            dia, _sec_to_hhmmss(hora_atual),
            velocidade, c2["cep"], c2["lat"], c2["lon"],
            pouso,
            _sec_to_hhmmss(hora_final_seg),
        ])
        hora_atual = hora_final_seg + drone_tempo_pouso
        if hora_atual >= 19 * 3600:
            dia += 1
            hora_atual = 6 * 3600

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "CEP_inicial", "Latitude_inicial", "Longitude_inicial", "Dia_do_voo",
            "Hora_inicial", "Velocidade", "CEP_final", "Latitude_final",
            "Longitude_final", "Pouso", "Hora_final"
        ])
        writer.writerows(linhas)
    print(f"\nArquivo CSV gerado: {arquivo_saida}")
    print(f"Distância total: {distancia_total:.2f} km")


def _sec_to_hhmmss(segundos):
    segundos = int(segundos % (24*3600))
    h = segundos // 3600
    m = (segundos % 3600) // 60
    s = segundos % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================
# BLOCO PRINCIPAL (exemplo de execução)
# =============================

if __name__ == "__main__":
    arquivo_coordenadas = "coordenadas.csv"
    arquivo_vento = "vento.csv"
    seed = 42
    arquivo_saida = "melhor_rota_ag_puro_v3_otimizado.csv"

    random.seed(seed)
    np.random.seed(seed)

    print("Carregando dados...")
    coord = Coordenadas(arquivo_coordenadas)
    vento = Vento(arquivo_vento)
    drone = Drone()

    print("Iniciando AG PURO V3 OTIMIZADO...")
    ga = GeneticAlgorithmFast(
        coord, vento, drone,
        n_pop=200, n_gen=300,
        elitismo=0.12,
        taxa_mut_inicial=0.07, taxa_mut_final=0.35,
        seed=seed,
        n_workers=min(8, max(1, (os.cpu_count() or 2) - 1)),
        chunk_size=16,
        use_process=False  # True se quiser usar ProcessPoolExecutor
    )

    melhor_info, melhor_fit = ga.executar(verbose=True)

    print("\n" + "="*70)
    print("MELHOR SOLUÇÃO ENCONTRADA (OTIMIZADO)")
    print("="*70)
    print(f"Fitness: {melhor_fit:.5f}")
    print(f"Distância Total: {melhor_info[2]:.2f} km")
    print(f"Tempo Estimado: ~{melhor_info[3]/40:.0f} min")
    print(f"Recargas Forçadas: {melhor_info[4]}")
    print(f"Rota (IDs): {melhor_info[0][:8]}... → {melhor_info[0][-1]}")
    print("="*70)

    gerar_csv_final(melhor_info, coord, vento, arquivo_saida)
