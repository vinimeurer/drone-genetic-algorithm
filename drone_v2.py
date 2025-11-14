# -*- coding: utf-8 -*-
"""
ga_drone_v5_1.py
AG PURO v5.1 — Versão corrigida/robusta de v5.0 TURBO
Autor: Você + Assistente (v5.1)
Data: 2025-11-14 (v5.1)
Notas:
 - Usa Numba quando disponível (caminho acelerado).
 - Usa ThreadPoolExecutor (seguro com Numba — evita cópias pesadas).
 - Corrige hash do cache (um hash por indivíduo).
 - Valida entradas antes de avaliações Numba.
 - Mantém a tabela v_eff pré-calculada.
"""

import csv
import math
import random
import time
import hashlib
import threading
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional: numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------------------
# CONFIGURAÇÃO v5.1
# ---------------------
DEFAULT_N_POP = 300
DEFAULT_N_GEN = 2000
DEFAULT_ELITISMO = 0.04
DEFAULT_TAXA_MUT_INI = 0.07
DEFAULT_TAXA_MUT_FIN = 0.35
LOG_INTERVAL = 10
WORKERS = max(1, (os.cpu_count() or 2) - 1)  # threads
FIT_CACHE_MAX_ENTRIES = 200000
FIT_CACHE_DIGEST_BYTES = 8
SEED = 42
MAX_STAGNATION = 300

# Velocidades discretas (mantive conjunto reduzido pro desempenho)
DRONE_VELOCIDADES = [36, 44, 52, 60, 68, 76, 84, 92]

# Discretização para tabela v_eff: reduzir bin para maior precisão
AZIMUTE_BIN = 10   # 360 / 10 = 36 bins
AZIMUTE_BINS = np.arange(0, 360, AZIMUTE_BIN, dtype=np.float32)

# =============================
# UTIL: LRU CACHE THREAD-SAFE
# =============================
class LRUCache:
    def __init__(self, maxsize=200000):
        self.maxsize = int(maxsize)
        self._od = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            try:
                val = self._od.pop(key)
                self._od[key] = val
                return val
            except KeyError:
                return None

    def set(self, key, value):
        with self._lock:
            if key in self._od:
                self._od.pop(key)
            self._od[key] = value
            if len(self._od) > self.maxsize:
                self._od.popitem(last=False)

    def clear(self):
        with self._lock:
            self._od.clear()

    def __len__(self):
        with self._lock:
            return len(self._od)


# =============================
# CLASSES (COORD, VENTO, DRONE)
# =============================
class Coordenadas:
    def __init__(self, arquivo_csv: str):
        df = pd.read_csv(arquivo_csv).reset_index(drop=True)
        df["ID"] = list(range(1, len(df) + 1))
        self.coordenadas = {
            int(row["ID"]): {"cep": row.get("cep", ""), "lat": float(row["latitude"]), "lon": float(row["longitude"])}
            for _, row in df.iterrows()
        }
        self.ids = sorted(self.coordenadas.keys())
        self.idx_map = {id_: i for i, id_ in enumerate(self.ids)}
        self._build_matrices()

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _azimute(self, lat1, lon1, lat2, lon2):
        dlon = math.radians(lon2 - lon1)
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        az = math.degrees(math.atan2(x, y))
        return (az + 360) % 360

    def _build_matrices(self):
        n = len(self.ids)
        self.dist_matrix = np.zeros((n, n), dtype=np.float32)
        self.az_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            id_i = self.ids[i]
            p_i = self.coordenadas[id_i]
            for j in range(i + 1, n):
                id_j = self.ids[j]
                p_j = self.coordenadas[id_j]
                d = self._haversine(p_i["lat"], p_i["lon"], p_j["lat"], p_j["lon"])
                az = self._azimute(p_i["lat"], p_i["lon"], p_j["lat"], p_j["lon"])
                self.dist_matrix[i, j] = self.dist_matrix[j, i] = d
                self.az_matrix[i, j] = az
                self.az_matrix[j, i] = (az + 180) % 360

    def distancia(self, id1: int, id2: int) -> float:
        return float(self.dist_matrix[self.idx_map[id1], self.idx_map[id2]])

    def azimute(self, id1: int, id2: int) -> float:
        return float(self.az_matrix[self.idx_map[id1], self.idx_map[id2]])


class Vento:
    def __init__(self, arquivo_csv: str, max_dias=31):
        df = pd.read_csv(arquivo_csv)
        self.max_dias = max_dias
        self.vento_array = np.zeros((max_dias, 24, 2), dtype=np.float32)
        for _, row in df.iterrows():
            dia = int(row["dia"])
            hora = int(row["hora"])
            if 1 <= dia <= max_dias and 0 <= hora <= 23:
                vel = float(row.get("vel_kmh", 0.0))
                direc = float(row.get("direcao_deg", 0.0))
                self.vento_array[dia - 1, hora, 0] = vel
                self.vento_array[dia - 1, hora, 1] = direc

    def get_vento(self, dia: int, hora: int):
        if dia < 1 or dia > self.max_dias:
            return {"vel_kmh": 0.0, "direcao_deg": 0.0}
        hora = hora % 24
        return {
            "vel_kmh": float(self.vento_array[dia - 1, hora, 0]),
            "direcao_deg": float(self.vento_array[dia - 1, hora, 1])
        }


class Drone:
    def __init__(self):
        self.autonomia_base = 5000.0
        self.fator_curitiba = 0.93
        self.autonomia_real = self.autonomia_base * self.fator_curitiba
        self.velocidades = DRONE_VELOCIDADES
        self.tempo_pouso_seg = 72
        self.vel_to_idx = {v: i for i, v in enumerate(self.velocidades)}


# =============================
# TABELA v_eff (pré-calculada)
# =============================
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _build_v_eff_table_numba(vels, az_bins, vento_array):
        n_v = vels.shape[0]
        n_az = az_bins.shape[0]
        n_w = vento_array.shape[0] * vento_array.shape[1]
        table = np.zeros((n_v, n_az, n_w), dtype=np.float32)
        idx = 0
        for d in range(vento_array.shape[0]):
            for h in range(vento_array.shape[1]):
                w_vel = vento_array[d, h, 0]
                w_dir = vento_array[d, h, 1]
                dir_to = (w_dir + 180.0) % 360.0
                for iv in range(n_v):
                    v = vels[iv]
                    for ia in range(n_az):
                        az = az_bins[ia]
                        ang_d = az * (np.pi / 180.0)
                        ang_w = dir_to * (np.pi / 180.0)
                        vx = v * np.cos(ang_d) + w_vel * np.cos(ang_w)
                        vy = v * np.sin(ang_d) + w_vel * np.sin(ang_w)
                        val = math.hypot(vx, vy)
                        if val < 0.1:
                            val = 0.1
                        table[iv, ia, idx] = val
                idx += 1
        return table
else:
    def _build_v_eff_table_numba(vels, az_bins, vento_array):
        # fallback python implementation (same semantics)
        n_v = len(vels)
        n_az = len(az_bins)
        n_w = vento_array.shape[0] * vento_array.shape[1]
        table = np.zeros((n_v, n_az, n_w), dtype=np.float32)
        idx = 0
        for d in range(vento_array.shape[0]):
            for h in range(vento_array.shape[1]):
                w_vel = vento_array[d, h, 0]
                w_dir = vento_array[d, h, 1]
                dir_to = (w_dir + 180.0) % 360.0
                for iv in range(n_v):
                    v = float(vels[iv])
                    for ia in range(n_az):
                        az = float(az_bins[ia])
                        ang_d = math.radians(az)
                        ang_w = math.radians(dir_to)
                        vx = v * math.cos(ang_d) + w_vel * math.cos(ang_w)
                        vy = v * math.sin(ang_d) + w_vel * math.sin(ang_w)
                        val = math.hypot(vx, vy)
                        if val < 0.1:
                            val = 0.1
                        table[iv, ia, idx] = val
                idx += 1
        return table


# =============================
# AVALIAÇÃO EM LOTE (NUMBA) — mesma assinatura do v5.0
# =============================
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def avaliar_lote_numba(
        rotas_batch,      # (B, L) uint16 (índices 0..N-1)
        vels_batch,       # (B, L-1) uint8 (índices de velocidade)
        dist_matrix,      # (N, N) float32
        az_matrix,        # (N, N) float32
        v_eff_table,      # (V, A, W) float32
        vento_idx_map,    # (max_dias*24,) int32 (mapping simples)
        autonomia_real,   # float32
        tempo_pouso_seg,  # int32
        max_dias          # int32
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
                # defensive: indices inválidos serão interpretados como 0 (base)
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
                # clamp indices
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
else:
    # fallback python implementation with identical semantics (vectorized where possible)
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
                az_bin = max(0, min(az_bin, v_eff_table.shape[1] - 1))
                v_idx = int(vels_batch[b, i])
                w_idx = int(vento_idx_map[(dia - 1) * 24 + (hora_atual_seg // 3600) % 24])
                v_idx = max(0, min(v_idx, v_eff_table.shape[0] - 1))
                w_idx = max(0, min(w_idx, v_eff_table.shape[2] - 1))
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
            fitness = float(round(max(0.01, min(0.99, fitness)), 5))
            fits[b] = fitness
            dists[b] = distancia_total
            custos[b] = custo_total
            recs[b] = pousos_forcados
        return fits, dists, custos, recs


# =============================
# ALGORITMO GENÉTICO TURBO (v5.1)
# =============================
class GeneticAlgorithm:
    def __init__(self, coord, vento, drone,
                 n_pop=DEFAULT_N_POP, n_gen=DEFAULT_N_GEN, elitismo=DEFAULT_ELITISMO,
                 taxa_mut_inicial=DEFAULT_TAXA_MUT_INI, taxa_mut_final=DEFAULT_TAXA_MUT_FIN,
                 seed=SEED, n_workers=WORKERS, max_stagnation=MAX_STAGNATION):
        self.coord = coord
        self.vento = vento
        self.drone = drone
        self.n_pop = int(n_pop)
        self.n_gen = int(n_gen)
        self.elitismo = float(elitismo)
        self.taxa_mut_inicial = taxa_mut_inicial
        self.taxa_mut_final = taxa_mut_final
        self.seed = seed
        self.n_workers = n_workers
        self.max_stagnation = max_stagnation
        self.log_interval = LOG_INTERVAL

        random.seed(seed)
        np.random.seed(seed)

        self.base = [i for i in coord.ids if i != 1]
        self.N = len(coord.ids)
        self.L = len(self.base) + 2  # +2 para base

        # Matrizes otimizadas (contiguous float32)
        self.dist_matrix = np.ascontiguousarray(coord.dist_matrix.astype(np.float32))
        self.az_matrix = np.ascontiguousarray(coord.az_matrix.astype(np.float32))
        self.vento_array = np.ascontiguousarray(vento.vento_array.astype(np.float32))

        # Tabela v_eff (pré-calculada)
        self.v_eff_table = _build_v_eff_table_numba(
            np.array(drone.velocidades, dtype=np.float32),
            AZIMUTE_BINS.astype(np.float32),
            self.vento_array
        )

        # vento_idx_map simples (mapa 0..(dias*24-1) -> index no 3º eixo da tabela)
        self.vento_idx_map = np.arange(self.vento_array.shape[0] * 24, dtype=np.int32)

        # Cache
        self.fitness_cache = LRUCache(FIT_CACHE_MAX_ENTRIES)

    def inicializar_populacao(self):
        pop = []
        base = self.base
        vels = self.drone.velocidades
        weights = [0.25] * (len(vels) - 2) + [0.75] * 2
        for _ in range(self.n_pop):
            perm = random.sample(base, len(base))
            rota = [1] + perm + [1]
            velocidades = random.choices(vels, weights=weights, k=len(rota) - 1)
            pop.append((rota, velocidades))
        return pop

    def _avaliar_lote(self, individuos):
        """
        Avalia um lote de indivíduos (lista de (rota, vels)).
        Retorna lista de tuplas: (fitness, (rota, vels, dist, custo, recs, []))
        """
        B = len(individuos)
        rotas_batch = np.zeros((B, self.L), dtype=np.uint16)
        vels_batch = np.zeros((B, self.L - 1), dtype=np.uint8)
        hashes = [None] * B

        # preenchimento de arrays e criação de hashes (um hash por indivíduo)
        for b, (rota, vels) in enumerate(individuos):
            # defensive fill: cap length
            if len(rota) != self.L:
                # tentar sanitizar: preencher/trim
                rota = (rota + [1] * self.L)[:self.L]
            for i, rid in enumerate(rota):
                rotas_batch[b, i] = self.coord.idx_map.get(int(rid), 0)
            for i, v in enumerate(vels):
                # se velocidade não está no mapeamento, escolher idx 0
                vels_batch[b, i] = self.drone.vel_to_idx.get(int(v), 0)

            # hash **novo** para cada indivíduo (corrige bug v5.0)
            h = hashlib.blake2b(digest_size=FIT_CACHE_DIGEST_BYTES)
            h.update(rotas_batch[b].tobytes())
            h.update(b'|')
            h.update(vels_batch[b].tobytes())
            key = h.digest()
            hashes[b] = key

        # verificar cache
        cached_results = [None] * B
        uncached_idx = []
        for i, key in enumerate(hashes):
            cached = self.fitness_cache.get(key)
            if cached is not None:
                cached_results[i] = cached
            else:
                uncached_idx.append(i)

        # avaliação dos não-cacheados em lotes (uma chamada Numba)
        if uncached_idx:
            sub_rotas = rotas_batch[uncached_idx]
            sub_vels = vels_batch[uncached_idx]
            # chamada para função Numba / fallback
            fits, dists, custos, recs = avaliar_lote_numba(
                sub_rotas, sub_vels,
                self.dist_matrix, self.az_matrix,
                self.v_eff_table, self.vento_idx_map,
                np.float32(self.drone.autonomia_real),
                np.int32(self.drone.tempo_pouso_seg),
                np.int32(self.vento_array.shape[0])
            )
            for i_local, idx in enumerate(uncached_idx):
                fit = float(fits[i_local])
                res = (fit, (individuos[idx][0], individuos[idx][1], float(dists[i_local]), float(custos[i_local]), int(recs[i_local]), []))
                # armazena no cache
                self.fitness_cache.set(hashes[idx], res)
                cached_results[idx] = res

        # todos os resultados agora em cached_results (preserva ordem)
        return cached_results

    def avaliar_populacao(self, populacao):
        # usar ThreadPoolExecutor: seguro com Numba e evita pickle pesado
        batch_size = max(1, len(populacao) // max(1, self.n_workers))
        resultados = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = []
            for i in range(0, len(populacao), batch_size):
                batch = populacao[i:i + batch_size]
                futures.append(ex.submit(self._avaliar_lote, batch))
            for f in futures:
                resultados.extend(f.result())
        return resultados

    def selecionar_pais(self, avaliacoes, gen):
        k = 5 if gen < self.n_gen * 0.25 else 3
        contestants = random.sample(avaliacoes, min(k, len(avaliacoes)))
        return max(contestants, key=lambda x: x[0])[1]

    def executar(self, verbose=True):
        populacao = self.inicializar_populacao()
        start_time = time.time()
        avaliacoes = self.avaliar_populacao(populacao)
        avaliacoes.sort(reverse=True, key=lambda x: x[0])
        melhor_global = avaliacoes[0]
        estagnado = 0

        if verbose:
            print("=== AG PURO v5.1 INICIADO ===")
            print(f"Numba disponível: {NUMBA_AVAILABLE} | Threads: {self.n_workers} | Cache entries: {FIT_CACHE_MAX_ENTRIES}")

        for gen in range(self.n_gen):
            t = gen / (self.n_gen - 1) if self.n_gen > 1 else 1.0
            taxa_mut = self.taxa_mut_inicial + t * (self.taxa_mut_final - self.taxa_mut_inicial)
            elite_n = max(1, int(self.elitismo * self.n_pop))
            nova_pop = [avaliacoes[i][1][:2] for i in range(elite_n)]

            # reprodução
            while len(nova_pop) < self.n_pop:
                p1 = self.selecionar_pais(avaliacoes, gen)
                p2 = self.selecionar_pais(avaliacoes, gen)
                child_rota = self.pmx_crossover(p1[0], p2[0])
                child_vels = self.crossover_velocidades(p1[1], p2[1])
                child_rota = self.mutacao_inversao(child_rota, taxa_mut)
                child_vels = self.mutacao_velocidades(child_vels, taxa_mut)
                nova_pop.append((child_rota, child_vels))

            populacao = nova_pop
            avaliacoes = self.avaliar_populacao(populacao)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])

            if avaliacoes[0][0] > melhor_global[0]:
                melhor_global = avaliacoes[0]
                estagnado = 0
            else:
                estagnado += 1

            if estagnado > 40:
                if verbose:
                    print(f"\n>>> REINÍCIO RÁPIDO (G{gen+1}) <<<")
                elite = [x[1][:2] for x in avaliacoes[:max(1, int(0.05 * self.n_pop))]]
                nova_pop = elite[:]
                while len(nova_pop) < self.n_pop:
                    perm = random.sample(self.base, len(self.base))
                    rota = [1] + perm + [1]
                    rota = self.mutacao_inversao(rota, 0.75)
                    vels = random.choices(self.drone.velocidades, k=len(rota) - 1)
                    nova_pop.append((rota, vels))
                populacao = nova_pop
                avaliacoes = self.avaliar_populacao(populacao)
                avaliacoes.sort(reverse=True, key=lambda x: x[0])
                estagnado = 0

            if estagnado > self.max_stagnation:
                if verbose:
                    print(f"\nEstagnação crítica (G{gen+1}) — interrompendo.")
                break

            if verbose and (gen % self.log_interval == 0 or gen == self.n_gen - 1):
                best = avaliacoes[0]
                elapsed = time.time() - start_time
                print(f"G{gen+1:4d} | Fit: {best[0]:.5f} | Dist: {best[1][2]:.1f}km | T:~{best[1][3]/40:.0f}min | Rec: {best[1][4]} | Stag: {estagnado} | Elap: {elapsed:.1f}s")

        if verbose:
            print("=== FIM TURBO v5.1 ===")
        return melhor_global[1], melhor_global[0]

    # Operadores genéticos (mantidos)
    def pmx_crossover(self, p1, p2):
        size = len(p1)
        if size < 4:
            return p1[:]
        a, b = sorted(random.sample(range(1, size - 1), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        mapping = {p2[i]: p1[i] for i in range(a, b) if p2[i] != p1[i]}
        used = set(child[a:b])
        for i in range(1, size - 1):
            if a <= i < b:
                continue
            gene = p2[i]
            # resolve chain
            while gene in mapping and gene not in used:
                gene = mapping[gene]
            if gene not in used:
                child[i] = gene
                used.add(gene)
        for i in range(1, size - 1):
            if child[i] is None:
                for g in p1[1:-1]:
                    if g not in used:
                        child[i] = g
                        used.add(g)
                        break
        child[0] = child[-1] = 1
        return child

    def crossover_velocidades(self, v1, v2):
        if len(v1) != len(v2):
            return v1[:]
        a, b = sorted(random.sample(range(len(v1)), 2))
        return v1[:a] + v2[a:b] + v1[b:]

    def mutacao_inversao(self, rota, taxa):
        if random.random() > taxa:
            return rota
        i, j = sorted(random.sample(range(1, len(rota) - 1), 2))
        return rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]

    def mutacao_velocidades(self, vels, taxa):
        v = vels[:]
        for i in range(len(v)):
            if random.random() < taxa:
                v[i] = random.choice(self.drone.velocidades)
        return v

# =============================
# CSV FINAL
# =============================
def reavaliar_preciso(info, coord, vento, drone, max_dias=31):
    """
    Reavalia o indivíduo com precisão total:
    - Usa vento real por dia/hora
    - Calcula v_eff exato (sem discretização de azimute)
    - Rastreia bateria, recargas e horários com precisão
    """
    rota, velocidades, _, _, _, _ = info
    custo_total = 0.0
    distancia_total = 0.0
    bateria = drone.autonomia_real
    dia = 1
    hora_atual_seg = 6 * 3600
    pousos_forcados = 0
    recargas = []

    idx_map = coord.idx_map
    dist_matrix = coord.dist_matrix
    az_matrix = coord.az_matrix

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        i1, i2 = idx_map[id1], idx_map[id2]
        dist = float(dist_matrix[i1, i2])
        distancia_total += dist
        az = float(az_matrix[i1, i2])
        v = float(velocidades[i])

        # Vento real (sem tabela pré-calculada)
        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        w_vel = vento_info["vel_kmh"]
        w_dir = vento_info["direcao_deg"]
        dir_to = (w_dir + 180.0) % 360.0

        # v_eff exato
        ang_d = math.radians(az)
        ang_w = math.radians(dir_to)
        vx = v * math.cos(ang_d) + w_vel * math.cos(ang_w)
        vy = v * math.sin(ang_d) + w_vel * math.sin(ang_w)
        v_eff = math.hypot(vx, vy)
        if v_eff < 0.1:
            v_eff = 0.1

        tempo_seg = int(math.ceil(dist * 3600.0 / v_eff))

        if tempo_seg > bateria:
            excesso_min = (tempo_seg - bateria) / 60.0
            custo_total += 15.0 + 3.0 * excesso_min
            pousos_forcados += 1
            recargas.append((id1, "RECARGA FORÇADA"))
            bateria = drone.autonomia_real
            hora_atual_seg += drone.tempo_pouso_seg
        else:
            bateria -= tempo_seg

        custo_total += (tempo_seg / 60.0) + (dist * 40.0)
        hora_atual_seg += tempo_seg + drone.tempo_pouso_seg

        if hora_atual_seg >= 19 * 3600:
            dia += 1
            hora_atual_seg = 6 * 3600
            if dia > max_dias:
                custo_total += 1e7
                break

    custo_total += 5.0 * pousos_forcados
    fitness = 1.0 / (1.0 + (custo_total / 60000.0) ** 1.8)
    fitness = max(0.01, min(0.99, fitness))
    fitness = round(fitness, 5)

    return fitness, (rota, velocidades, distancia_total, custo_total, pousos_forcados, recargas)

# =============================
# CSV FINAL
# =============================
def gerar_csv_final(info, coord, vento, arquivo_saida="rota.csv"):
    rota, velocidades, distancia_total, _, _, recargas = info
    recarga_set = {(id_, msg) for id_, msg in recargas}
    linhas = []
    dia = 1
    hora_atual_seg = 6 * 3600
    idx_map = coord.idx_map
    dist_matrix = coord.dist_matrix
    az_matrix = coord.az_matrix

    def s2hms(s):
        h = int(s // 3600) % 24
        m = int((s % 3600) // 60)
        return f"{h:02d}:{m:02d}"

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        c1, c2 = coord.coordenadas[id1], coord.coordenadas[id2]
        v = velocidades[i]
        i1, i2 = idx_map[id1], idx_map[id2]
        dist = dist_matrix[i1, i2]
        az = az_matrix[i1, i2]
        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        v_eff = max(0.1, math.hypot(
            v * math.cos(math.radians(az)) + vento_info["vel_kmh"] * math.cos(math.radians(vento_info["direcao_deg"] + 180)),
            v * math.sin(math.radians(az)) + vento_info["vel_kmh"] * math.sin(math.radians(vento_info["direcao_deg"] + 180))
        ))
        tempo_voo = int(math.ceil(dist * 3600.0 / v_eff))
        hora_final = hora_atual_seg + tempo_voo
        pouso = "SIM" if (id1, "RECARGA FORÇADA") in recarga_set else "NÃO"
        linhas.append([
            c1["cep"], c1["lat"], c1["lon"], dia, s2hms(hora_atual_seg),
            v, c2["cep"], c2["lat"], c2["lon"], pouso, s2hms(hora_final)
        ])
        hora_atual_seg = hora_final + 72
        if hora_atual_seg >= 19 * 3600:
            dia += 1
            hora_atual_seg = 6 * 3600

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CEP_inicial", "Latitude_inicial", "Longitude_inicial", "Dia_do_voo", "Hora_inicial",
                         "Velocidade", "CEP_final", "Latitude_final", "Longitude_final", "Pouso", "Hora_final"])
        writer.writerows(linhas)
    print(f"\nArquivo gerado: {arquivo_saida}")
    print(f"Distância total: {distancia_total:.2f} km")


# =============================
# MAIN (execução)
# =============================
if __name__ == "__main__":
    print("=== AG DRONE v5.1 ===")
    arquivo_coord = "coordenadas.csv"
    arquivo_vento = "vento.csv"

    if not os.path.exists(arquivo_coord):
        raise FileNotFoundError(f"Arquivo de coordenadas não encontrado: {arquivo_coord}")
    if not os.path.exists(arquivo_vento):
        raise FileNotFoundError(f"Arquivo de vento não encontrado: {arquivo_vento}")

    coord = Coordenadas(arquivo_coord)
    vento = Vento(arquivo_vento)
    drone = Drone()

    ga = GeneticAlgorithm(coord, vento, drone,
                          n_pop=DEFAULT_N_POP, n_gen=DEFAULT_N_GEN,
                          n_workers=WORKERS, max_stagnation=MAX_STAGNATION)

    melhor_info, melhor_fit = ga.executar()

    print("\n" + "=" * 70)
    print("MELHOR SOLUÇÃO (v5.1)")
    print("=" * 70)
    print(f"Fitness: {melhor_fit:.5f}")
    print(f"Distância: {melhor_info[2]:.2f} km")
    print(f"Tempo: ~{melhor_info[3] / 40:.0f} min")
    print(f"Recargas: {melhor_info[4]}")
    print("=" * 70)

    # =========================
    # REAVALIAÇÃO PRECISA (sem tabela v_eff)
    # =========================
    print("Reavaliando com precisão total (vento real, sem discretização)...")
    fit_preciso, info_precisa = reavaliar_preciso(melhor_info, coord, vento, drone)

    print("\n" + "=" * 70)
    print("RESULTADO FINAL (PRECISO)")
    print("=" * 70)
    print(f"Fitness (preciso): {fit_preciso:.5f}")
    print(f"Distância: {info_precisa[2]:.2f} km")
    print(f"Tempo: ~{info_precisa[3] / 40:.0f} min")
    print(f"Recargas: {info_precisa[4]}")
    print("=" * 70)

    # Gerar CSV com dados precisos
    gerar_csv_final(info_precisa, coord, vento, "rota_precisa.csv")

    print(f"\nNumba disponível: {NUMBA_AVAILABLE} | Threads: {WORKERS} | Cache máximo: {FIT_CACHE_MAX_ENTRIES} entradas")
    print("CSV final gerado com avaliação precisa: rota_precisa.csv")
