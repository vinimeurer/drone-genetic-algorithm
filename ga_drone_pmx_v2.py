# -*- coding: utf-8 -*-
"""
ga_drone_v4.1.py
AG PURO v4.1 — Seguro e otimizado (LRU cache + chaves compactas + crossovers robustos)
Autor: Você + Assistente
Data: 2025-11-13
Notas:
 - Ajuste rápido no topo: FIT_CACHE_ENABLED, FIT_CACHE_MAX_ENTRIES, WORKERS, N_POP/N_GEN, etc.
 - Cache usa blake2b(digest_size=16) sobre arrays numpy (rota: uint16, velocidades: uint8).
 - LRUCache thread-safe evita crescimento descontrolado e thrash de disco.
 - Crossovers PMX/OX reescritos para nunca retornar None e sempre produzir permutações válidas.
"""

import csv
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from collections import OrderedDict
import threading
import hashlib

import numpy as np
import pandas as pd
import os

# Optional: psutil for memory monitoring (not required)
try:
    import psutil
except Exception:
    psutil = None

# ---------------------
# CONFIGURÁVEIS RÁPIDOS
# ---------------------
DEFAULT_N_POP = 500
DEFAULT_N_GEN = 1500
DEFAULT_ELITISMO = 0.04
DEFAULT_TAXA_MUT_INI = 0.07
DEFAULT_TAXA_MUT_FIN = 0.35
LOG_INTERVAL = 10
WORKERS = max(2, (os.cpu_count() or 2) - 1)
FIT_CACHE_ENABLED = True
FIT_CACHE_MAX_ENTRIES = 60000   # Ajuste este valor conforme RAM (ex.: 60000 é conservador para 16 GB)
FIT_CACHE_DIGEST_BYTES = 16     # tamanho do digest blake2b (8 ou 16)
SEED = 42
MAX_STAGNATION = 300

# =============================
# UTIL: LRU CACHE THREAD-SAFE
# =============================
class LRUCache:
    """LRU cache simples e thread-safe usando OrderedDict."""
    def __init__(self, maxsize=60000):
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
                return
            self._od[key] = value
            if len(self._od) > self.maxsize:
                try:
                    self._od.popitem(last=False)
                except Exception:
                    pass

    def clear(self):
        with self._lock:
            self._od.clear()

    def __len__(self):
        with self._lock:
            return len(self._od)


# =============================
# CLASSES (COORDENADAS, VENTO, DRONE)
# =============================
class Coordenadas:
    def __init__(self, arquivo_csv: str):
        self.df = pd.read_csv(arquivo_csv).reset_index(drop=True).copy()
        self.df["ID"] = list(range(1, len(self.df) + 1))
        self.coordenadas: Dict[int, Dict[str, float]] = {
            int(row["ID"]): {
                "cep": row.get("cep", ""),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
            }
            for _, row in self.df.iterrows()
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
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

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
            for j in range(i+1, n):
                id_j = self.ids[j]
                p_j = self.coordenadas[id_j]
                d = self._haversine(p_i["lat"], p_i["lon"], p_j["lat"], p_j["lon"])
                az = self._azimute(p_i["lat"], p_i["lon"], p_j["lat"], p_j["lon"])
                self.dist_matrix[i, j] = d
                self.dist_matrix[j, i] = d
                self.az_matrix[i, j] = az
                self.az_matrix[j, i] = (az + 180) % 360

    def distancia(self, id1: int, id2: int) -> float:
        return float(self.dist_matrix[self.idx_map[id1], self.idx_map[id2]])

    def azimute(self, id1: int, id2: int) -> float:
        return float(self.az_matrix[self.idx_map[id1], self.idx_map[id2]])


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
            self.velocidades = list(range(36, 100, 4))
        else:
            self.velocidades = velocidades
        self.tempo_pouso_seg = 72

    def tempo_voo_seg(self, distancia_km: float, v_kmh: float) -> int:
        if v_kmh <= 0:
            return int(1e6)
        tempo = distancia_km * 3600.0 / v_kmh
        return int(math.ceil(tempo))


def calcular_v_efetiva(v_drone_kmh: float, direcao_voo_deg: float, vento_info: Dict[str, float]) -> float:
    v_wind = vento_info.get("vel_kmh", 0.0)
    dir_wind_to = (vento_info.get("direcao_deg", 0.0) + 180.0) % 360.0
    ang_drone = math.radians(direcao_voo_deg)
    ang_wind = math.radians(dir_wind_to)
    vx = v_drone_kmh * math.cos(ang_drone) + v_wind * math.cos(ang_wind)
    vy = v_drone_kmh * math.sin(ang_drone) + v_wind * math.sin(ang_wind)
    v_eff = math.hypot(vx, vy)
    return max(v_eff, 0.1)


# =============================
# AVALIAÇÃO (ENXUTA E COM CACHE OPCIONAL)
# =============================
def avaliar_rota_individual(individual, coord: Coordenadas, vento: Vento, drone: Drone, max_dias=7):
    rota, velocidades = individual
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

    v_eff_cache = {}

    # iterate edges
    for i in range(len(rota) - 1):
        id1 = rota[i]
        id2 = rota[i + 1]
        # defensive: ids must be present
        if id1 not in idx_map or id2 not in idx_map:
            # penaliza muito rota inválida
            return 0.01, (rota, velocidades, 0.0, 1e9, 9999, [("INVALID", "ID")])
        i1 = idx_map[id1]
        i2 = idx_map[id2]
        dist = float(dist_matrix[i1, i2])
        distancia_total += dist

        v_chosen = float(velocidades[i])
        az = float(az_matrix[i1, i2])

        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        key = (int(round(v_chosen)), int(round(az)) % 360,
               int(round(vento_info["vel_kmh"])), int(round(vento_info["direcao_deg"])))
        v_efetiva = v_eff_cache.get(key)
        if v_efetiva is None:
            v_efetiva = calcular_v_efetiva(v_chosen, az, vento_info)
            v_eff_cache[key] = v_efetiva

        tempo_seg = drone.tempo_voo_seg(dist, v_efetiva)

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
    custo_ref = 60000.0
    fitness = 1.0 / (1.0 + (custo_total / custo_ref) ** 1.8)
    fitness = max(0.01, min(0.99, fitness))
    fitness = round(fitness, 5)

    return fitness, (rota, velocidades, distancia_total, custo_total, pousos_forcados, recargas)


# =============================
# OPERADORES GENÉTICOS (ROBUSTOS)
# =============================
def ox_crossover(p1, p2, base_id=1):
    size = len(p1)
    if size < 4:
        return p1[:]
    a, b = sorted(random.sample(range(1, size - 1), 2))
    child = [None] * size
    # copy slice
    child[a:b] = p1[a:b]
    pos = b
    used = set(child[a:b])
    for gene in p2[1:-1]:
        if gene not in used:
            if pos >= size - 1:
                pos = 1
            child[pos] = gene
            used.add(gene)
            pos += 1
    # fallback: fill remaining with genes from p1 that are not used
    faltando = [g for g in p1[1:-1] if g not in used]
    for i in range(1, size - 1):
        if child[i] is None:
            if faltando:
                child[i] = faltando.pop(0)
            else:
                # as last resort, pick any id from p1 that isn't present
                for g in p1[1:-1]:
                    if g not in child:
                        child[i] = g
                        break
    child[0] = base_id
    child[-1] = base_id
    return child


def pmx_crossover(p1, p2, base_id=1):
    size = len(p1)
    if size < 4:
        return p1[:]
    a, b = sorted(random.sample(range(1, size - 1), 2))
    child = [None] * size
    child[a:b] = p1[a:b]

    # build mapping p2 -> p1 for slice
    mapping = {}
    for i in range(a, b):
        if p2[i] != p1[i]:
            mapping[p2[i]] = p1[i]

    used = set(x for x in child[a:b] if x is not None)

    # fill positions outside [a,b) by following mapping resolution
    for i in range(1, size - 1):
        if a <= i < b:
            continue
        gene = p2[i]
        visited = set()
        # resolve through mapping until a gene not in used or loop
        while gene in mapping and gene not in visited:
            visited.add(gene)
            gene = mapping[gene]
        if gene not in used:
            child[i] = gene
            used.add(gene)

    # fill remaining slots with leftover genes from p1 (in order)
    faltando = [g for g in p1[1:-1] if g not in used]
    for i in range(1, size - 1):
        if child[i] is None:
            if faltando:
                child[i] = faltando.pop(0)
                used.add(child[i])
            else:
                # final fallback: choose any not-yet-used id from p1
                for g in p1[1:-1]:
                    if g not in used:
                        child[i] = g
                        used.add(g)
                        break

    child[0] = base_id
    child[-1] = base_id
    return child


def crossover_velocidades(v1, v2):
    if len(v1) != len(v2):
        # safety: return copy of v1 if lengths differ (shouldn't happen)
        return v1[:]
    a, b = sorted(random.sample(range(len(v1)), 2))
    child = v1[:a] + v2[a:b] + v1[b:]
    return child


def mutacao_inversao(rota, taxa):
    if random.random() > taxa:
        return rota
    i, j = sorted(random.sample(range(1, len(rota)-1), 2))
    return rota[:i] + rota[i:j+1][::-1] + rota[j+1:]


def mutacao_velocidades(vels, taxa_mut, drone):
    vel_n = vels[:]
    high_vels = drone.velocidades[-6:]
    base_len = max(1, len(drone.velocidades) - len(high_vels))
    weights = [0.25] * base_len + [0.75] * len(high_vels)
    for i in range(len(vel_n)):
        if random.random() < taxa_mut:
            vel_n[i] = random.choices(drone.velocidades, weights=weights, k=1)[0]
    return vel_n


# =============================
# ALGORITMO GENÉTICO (PURO) - IMPLEMENTAÇÃO
# =============================
class GeneticAlgorithm:
    def __init__(self, coord, vento, drone,
                 n_pop=DEFAULT_N_POP, n_gen=DEFAULT_N_GEN, elitismo=DEFAULT_ELITISMO,
                 taxa_mut_inicial=DEFAULT_TAXA_MUT_INI, taxa_mut_final=DEFAULT_TAXA_MUT_FIN,
                 seed=SEED, n_workers=WORKERS, max_stagnation=MAX_STAGNATION, log_interval=LOG_INTERVAL):
        self.coord = coord
        self.vento = vento
        self.drone = drone
        self.n_pop = int(n_pop)
        self.n_gen = int(n_gen)
        self.elitismo = float(elitismo)
        self.taxa_mut_inicial = taxa_mut_inicial
        self.taxa_mut_final = taxa_mut_final
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.base = [i for i in self.coord.ids if i != 1]
        self.n_workers = n_workers
        self.max_stagnation = max_stagnation
        self.log_interval = log_interval

        # inicializa cache LRU se habilitado
        if FIT_CACHE_ENABLED:
            self.fitness_cache = LRUCache(maxsize=FIT_CACHE_MAX_ENTRIES)
        else:
            self.fitness_cache = None

    def inicializar_populacao(self):
        populacao = []
        base = self.base
        dv = self.drone.velocidades
        weights = [0.25] * (max(1, len(dv) - 6)) + [0.75] * 6
        for _ in range(self.n_pop):
            perm = random.sample(base, len(base))
            rota = [1] + perm + [1]
            velocidades = [random.choices(dv, weights=weights, k=1)[0] for _ in range(len(rota)-1)]
            populacao.append((rota, velocidades))
        return populacao

    def _avaliar_com_cache(self, individual):
        if not FIT_CACHE_ENABLED or self.fitness_cache is None:
            return avaliar_rota_individual(individual, self.coord, self.vento, self.drone)

        rota, vels = individual

        # sanitize None values (defensive)
        if any(r is None for r in rota):
            rota = [r if r is not None else 1 for r in rota]
        if any(v is None for v in vels):
            vels = [int(v if v is not None else self.drone.velocidades[0]) for v in vels]

        # compact arrays: uint16 for rota (suficiente p/ <=65535 pontos), uint8 for velocidades
        rota_arr = np.array(rota, dtype=np.uint16)
        vels_arr = np.array(vels, dtype=np.uint8)

        h = hashlib.blake2b(digest_size=FIT_CACHE_DIGEST_BYTES)
        h.update(rota_arr.tobytes())
        h.update(b'|')
        h.update(vels_arr.tobytes())
        key = h.digest()

        cached = self.fitness_cache.get(key)
        if cached is not None:
            return cached

        res = avaliar_rota_individual((rota, vels), self.coord, self.vento, self.drone)
        self.fitness_cache.set(key, res)
        return res

    def avaliar_populacao_parallel(self, populacao):
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            results = list(ex.map(self._avaliar_com_cache, populacao))
        return results

    def selecionar_pais(self, avaliacoes, gen):
        k = 5 if gen < self.n_gen * 0.25 else 3
        contestants = random.sample(avaliacoes, min(k, len(avaliacoes)))
        return max(contestants, key=lambda x: x[0])[1]

    def executar(self, verbose=True):
        populacao = self.inicializar_populacao()
        start_time = time.time()

        avaliacoes = self.avaliar_populacao_parallel(populacao)
        avaliacoes.sort(reverse=True, key=lambda x: x[0])
        melhor_global = avaliacoes[0]
        estagnado = 0

        if verbose:
            print("=== INICIANDO AG PURO V4.1 (OTIMIZADO E SEGURO) ===")

        for gen in range(self.n_gen):
            t = gen / (self.n_gen - 1) if self.n_gen > 1 else 1.0
            taxa_mut_atual = self.taxa_mut_inicial + t * (self.taxa_mut_final - self.taxa_mut_inicial)

            elite_n = max(1, int(self.elitismo * self.n_pop))
            nova_pop = [avaliacoes[i][1][:2] for i in range(elite_n)]

            # pequena injeção de diversidade
            for _ in range(3):
                perm = random.sample(self.base, len(self.base))
                rota = [1] + perm + [1]
                rota = mutacao_inversao(rota, taxa_mut_atual * 1.2)
                velocidades = [random.choice(self.drone.velocidades) for _ in range(len(rota)-1)]
                nova_pop.append((rota, velocidades))

            # reprodução
            while len(nova_pop) < self.n_pop:
                p1 = self.selecionar_pais(avaliacoes, gen)
                p2 = self.selecionar_pais(avaliacoes, gen)
                # escolha híbrida: PMX + vel crossover
                filho_rota = pmx_crossover(p1[0], p2[0], base_id=1)
                filho_vels = crossover_velocidades(p1[1], p2[1])
                filho_rota = mutacao_inversao(filho_rota, taxa_mut_atual)
                filho_vels = mutacao_velocidades(filho_vels, taxa_mut_atual, self.drone)
                nova_pop.append((filho_rota, filho_vels))

            populacao = nova_pop
            avaliacoes = self.avaliar_populacao_parallel(populacao)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])

            # atualização de melhor
            if avaliacoes and avaliacoes[0][0] > melhor_global[0]:
                melhor_global = avaliacoes[0]
                estagnado = 0
            else:
                estagnado += 1

            # reinício se estagnar muito
            if estagnado > 40:
                if verbose:
                    print(f"\n>>> REINÍCIO NA G{gen+1} (estagnado={estagnado}) <<<")
                elite_keep = avaliacoes[:max(1, int(0.05*self.n_pop))]
                nova_pop = [x[1][:2] for x in elite_keep]
                for _ in range(self.n_pop - len(nova_pop)):
                    perm = random.sample(self.base, len(self.base))
                    rota = [1] + perm + [1]
                    rota = mutacao_inversao(rota, 0.75)
                    velocidades = [random.choice(self.drone.velocidades) for _ in range(len(rota)-1)]
                    nova_pop.append((rota, velocidades))
                populacao = nova_pop
                avaliacoes = self.avaliar_populacao_parallel(populacao)
                avaliacoes.sort(reverse=True, key=lambda x: x[0])
                estagnado = 0

            if estagnado > self.max_stagnation:
                if verbose:
                    print(f"\nEncerrando por estagnação longa (G{gen+1})")
                break

            if verbose and (gen % self.log_interval == 0 or gen == self.n_gen - 1):
                best = avaliacoes[0]
                dist = best[1][2]
                tempo = best[1][3] / 40.0
                elapsed = time.time() - start_time
                # optional memory print
                mem_str = ""
                if psutil:
                    pinfo = psutil.Process(os.getpid()).memory_info()
                    mem_str = f" | RSS:{pinfo.rss//1024//1024}MB"
                print(f"G{gen+1:4d} | Fit: {best[0]:.5f} | Dist: {dist:.1f}km | Tempo: ~{tempo:.0f}min | Rec: {best[1][4]} | Stag: {estagnado} | T:{elapsed:.1f}s{mem_str}")

        if verbose:
            print("=== FIM DO AG PURO V4.1 ===")
        return melhor_global[1], melhor_global[0]


# =============================
# CSV FINAL (mesmo formato)
# =============================
def gerar_csv_final(info, coord: Coordenadas, vento: Vento, arquivo_saida="melhor_rota_ag_puro_v4_1.csv"):
    rota, velocidades, distancia_total, _, _, recargas = info
    linhas = []
    dia = 1
    hora_atual_seg = 6 * 3600
    drone_tempo_pouso = 72
    recarga_set = {(id_, msg) for id_, msg in recargas}

    idx_map = coord.idx_map
    dist_matrix = coord.dist_matrix
    az_matrix = coord.az_matrix

    def s2hms(s):
        h = int(s // 3600) % 24
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        c1, c2 = coord.coordenadas[id1], coord.coordenadas[id2]
        velocidade = velocidades[i]
        i1, i2 = idx_map[id1], idx_map[id2]
        dist = float(dist_matrix[i1, i2])
        az = float(az_matrix[i1, i2])
        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        v_eff = calcular_v_efetiva(velocidade, az, vento_info)
        tempo_voo = int(math.ceil(dist * 3600.0 / max(v_eff, 0.1)))
        hora_final_seg = hora_atual_seg + tempo_voo
        pouso = "SIM" if (id1, "RECARGA FORÇADA") in recarga_set else "NÃO"
        linhas.append([
            c1["cep"], c1["lat"], c1["lon"],
            dia, s2hms(hora_atual_seg),
            velocidade, c2["cep"], c2["lat"], c2["lon"],
            pouso, s2hms(hora_final_seg),
        ])
        hora_atual_seg = hora_final_seg + drone_tempo_pouso
        if hora_atual_seg >= 19 * 3600:
            dia += 1
            hora_atual_seg = 6 * 3600

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


# =============================
# BLOCO PRINCIPAL
# =============================
if __name__ == "__main__":
    print("=== OTIMIZAÇÃO DE ROTA DE DRONE (AG PURO V4.1 OTIMIZADO E SEGURO) ===\n")

    arquivo_coordenadas = "coordenadas.csv"
    arquivo_vento = "vento.csv"
    seed = SEED
    arquivo_saida = "rota_pmx_v4_1.csv"

    random.seed(seed)
    np.random.seed(seed)

    print("Carregando dados...")
    coord = Coordenadas(arquivo_coordenadas)
    vento = Vento(arquivo_vento)
    drone = Drone()

    print("Iniciando AG PURO V4.1 (otimizado)...")
    ga = GeneticAlgorithm(
        coord, vento, drone,
        n_pop=DEFAULT_N_POP, n_gen=DEFAULT_N_GEN,
        elitismo=DEFAULT_ELITISMO,
        taxa_mut_inicial=DEFAULT_TAXA_MUT_INI, taxa_mut_final=DEFAULT_TAXA_MUT_FIN,
        seed=seed, n_workers=WORKERS, max_stagnation=MAX_STAGNATION, log_interval=LOG_INTERVAL
    )

    melhor_info, melhor_fit = ga.executar()

    print("\n" + "="*70)
    print("MELHOR SOLUÇÃO ENCONTRADA (AG PURO V4.1)")
    print("="*70)
    print(f"Fitness: {melhor_fit:.5f}")
    print(f"Distância Total: {melhor_info[2]:.2f} km")
    print(f"Tempo Estimado: ~{melhor_info[3]/40:.0f} min")
    print(f"Recargas Forçadas: {melhor_info[4]}")
    print(f"Rota (IDs): {melhor_info[0][:8]}... → {melhor_info[0][-1]}")
    print("="*70)

    gerar_csv_final(melhor_info, coord, vento, arquivo_saida)
