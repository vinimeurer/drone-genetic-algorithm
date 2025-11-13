# -*- coding: utf-8 -*-
"""
AG PURO V4.0 — OTIMIZADO AO MÁXIMO (100% PURO, SEM HEURÍSTICAS)
Autor: Você + Grok (xAI) — PATCHED (LRU cache + chaves compactas)
Data: 2025-11-13
Objetivo: Chegar perto de ~340–400 km, mantendo AG PURO.
Observações das alterações:
 - Substituí o dict de cache por uma LRUCache limitada e thread-safe.
 - As chaves do cache agora são hashes blake2b (16 bytes) de arrays numpy compactos:
   rota -> uint16 (suficiente para 400 pontos), velocidades -> uint8.
 - Pequenas notas em locais modificados.
"""

import csv
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import os

# --------------------- ADICIONADOS ---------------------
import hashlib
from collections import OrderedDict
import threading
# -------------------------------------------------------

# ---------------------
# CONFIGURÁVEIS RÁPIDOS
# ---------------------
N_POP = 500
N_GEN = 1500
ELITISMO_INI = 0.02
TAXA_MUT_INI = 0.07
TAXA_MUT_PICO = 0.45
WORKERS = max(2, (os.cpu_count() or 2) - 1)
LOG_INTERVAL = 10
SEED = 42
FIT_CACHE_ENABLED = True
REINJECAO_INTERVAL = 50
REINJECAO_TAXA = 0.15
MAX_STAGNATION = 300

# =============================
# LRU CACHE (THREAD-SAFE) - NOVO
# =============================
class LRUCache:
    """LRU cache simples, thread-safe com lock. Limita número de entradas para evitar OOM / swap."""
    def __init__(self, maxsize=60000):
        self.maxsize = int(maxsize)
        self._od = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            try:
                val = self._od.pop(key)
                # re-inserir no final (mais recente)
                self._od[key] = val
                return val
            except KeyError:
                return None

    def set(self, key, value):
        with self._lock:
            if key in self._od:
                # atualizar posição
                self._od.pop(key)
                self._od[key] = value
                return
            self._od[key] = value
            # ejetar itens antigos se exceder maxsize
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
# CLASSES (OTIMIZADAS)
# =============================

class Coordenadas:
    def __init__(self, arquivo_csv: str):
        df = pd.read_csv(arquivo_csv).reset_index(drop=True)
        df["ID"] = list(range(1, len(df) + 1))
        self.coordenadas = {
            int(row["ID"]): {
                "cep": row.get("cep", ""),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
            }
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
    def __init__(self, arquivo_csv: str):
        df = pd.read_csv(arquivo_csv)
        self.vento = {}
        for _, row in df.iterrows():
            dia = int(row["dia"])
            hora = int(row["hora"])
            self.vento.setdefault(dia, {})[hora] = {
                "vel_kmh": float(row.get("vel_kmh", 0.0)),
                "direcao_deg": float(row.get("direcao_deg", 0.0)),
            }

    def get_vento(self, dia: int, hora: int) -> Dict[str, float]:
        return self.vento.get(dia, {}).get(hora, {"vel_kmh": 0.0, "direcao_deg": 0.0})


class Drone:
    def __init__(self, autonomia_base: float = 5000.0, fator_curitiba: float = 0.93):
        self.autonomia_real = autonomia_base * fator_curitiba
        self.velocidades = list(range(36, 100, 4))
        self.tempo_pouso_seg = 72

    def tempo_voo_seg(self, distancia_km: float, v_kmh: float) -> int:
        if v_kmh <= 0: return int(1e6)
        return int(math.ceil(distancia_km * 3600.0 / v_kmh))


def calcular_v_efetiva(v_drone_kmh: float, direcao_voo_deg: float, vento_info: Dict[str, float]) -> float:
    v_wind = vento_info.get("vel_kmh", 0.0)
    dir_wind_to = (vento_info.get("direcao_deg", 0.0) + 180.0) % 360.0
    ang_drone = math.radians(direcao_voo_deg)
    ang_wind = math.radians(dir_wind_to)
    vx = v_drone_kmh * math.cos(ang_drone) + v_wind * math.cos(ang_wind)
    vy = v_drone_kmh * math.sin(ang_drone) + v_wind * math.sin(ang_wind)
    return max(math.hypot(vx, vy), 0.1)


# =============================
# AVALIAÇÃO (COM CACHE LOCAL + FITNESS RANK)
# =============================

def avaliar_rota_individual(individual, coord: Coordenadas, vento: Vento, drone: Drone):
    rota, velocidades = individual
    custo_total = distancia_total = 0.0
    bateria = drone.autonomia_real
    dia = 1
    hora_atual_seg = 6 * 3600
    pousos_forcados = 0
    recargas = []

    v_eff_cache = {}
    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        i1, i2 = coord.idx_map[id1], coord.idx_map[id2]
        dist = float(coord.dist_matrix[i1, i2])
        distancia_total += dist
        v_chosen = float(velocidades[i])
        az = float(coord.az_matrix[i1, i2])
        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))

        key = (int(round(v_chosen)), int(round(az)) % 360, int(round(vento_info["vel_kmh"])), int(round(vento_info["direcao_deg"])))
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
            if dia > 7:
                custo_total += 1e7
                break

    custo_total += 5.0 * pousos_forcados
    custo_ref = 60000.0
    fitness = 1.0 / (1.0 + (custo_total / custo_ref) ** 1.8)
    fitness = max(0.01, min(0.99, fitness))
    return fitness, (rota, velocidades, distancia_total, custo_total, pousos_forcados, recargas)


# =============================
# OPERADORES GENÉTICOS (HÍBRIDOS)
# =============================

def pmx_crossover(p1, p2, base_id=1):
    size = len(p1)
    if size < 4:
        return p1[:]
    a, b = sorted(random.sample(range(1, size - 1), 2))
    child = [None] * size
    child[a:b] = p1[a:b]

    mapping = {p2[i]: p1[i] for i in range(a, b)}
    used = set(child[a:b])

    # Preenche o resto garantindo que não haja None
    for i in range(1, size - 1):
        if a <= i < b:
            continue
        gene = p2[i]
        # resolve conflitos de mapeamento
        while gene in mapping and gene in used:
            gene = mapping[gene]
        if gene in used or gene is None:
            # procura o próximo gene não usado
            for g in p1[1:-1]:
                if g not in used:
                    gene = g
                    break
        child[i] = gene
        used.add(gene)

    child[0] = child[-1] = base_id

    # Fallback final: garante que todos os IDs existam
    faltando = [g for g in p1[1:-1] if g not in child]
    for i in range(1, size - 1):
        if child[i] is None and faltando:
            child[i] = faltando.pop(0)
    return child



def ox_crossover(p1, p2, base_id=1):
    size = len(p1)
    if size < 4:
        return p1[:]
    a, b = sorted(random.sample(range(1, size - 1), 2))
    child = [None] * size
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

    # Fallback: garante que não há None
    faltando = [g for g in p1[1:-1] if g not in child]
    for i in range(1, size - 1):
        if child[i] is None and faltando:
            child[i] = faltando.pop(0)

    child[0] = child[-1] = base_id
    return child



def mutacao_inversao(rota, taxa):
    if random.random() > taxa: return rota
    i, j = sorted(random.sample(range(1, len(rota)-1), 2))
    return rota[:i] + rota[i:j+1][::-1] + rota[j+1:]


def mutacao_swap(rota, taxa):
    if random.random() > taxa: return rota
    i, j = random.sample(range(1, len(rota)-1), 2)
    nova = rota[:]
    nova[i], nova[j] = nova[j], nova[i]
    return nova


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
# ALGORITMO GENÉTICO PURO V4.0
# =============================

class GeneticAlgorithmPuro:
    def __init__(self, coord, vento, drone, seed=SEED):
        self.coord = coord
        self.vento = vento
        self.drone = drone
        self.n_pop = N_POP
        self.n_gen = N_GEN
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.base = [i for i in self.coord.ids if i != 1]
        self.n_workers = WORKERS

        # Substitui dict por LRUCache para limitar uso de memória (modificação).
        if FIT_CACHE_ENABLED:
            # Ajuste max_entries conforme RAM disponível. Para 400 pontos, 60k-120k é conservador.
            max_entries = 60000
            self.fitness_cache = LRUCache(maxsize=max_entries)
        else:
            self.fitness_cache = None

    # Substituído: versão que gera chave compacta usando blake2b sobre arrays numpy.
    def _avaliar_com_cache(self, ind):
        if not FIT_CACHE_ENABLED:
            return avaliar_rota_individual(ind, self.coord, self.vento, self.drone)

        rota, vels = ind

        # --- compactação para bytes (uint16 para rota, uint8 para velocidades)
        # Para 400 pontos, uint16 é mais que suficiente; se tiver >65535 troque para uint32.
        # --- SANITIZAÇÃO DE ROTA (caso algum crossover tenha gerado None) ---
        if any(r is None for r in rota):
            # substitui None por 1 (base) temporariamente para evitar crash no hash
            rota = [r if r is not None else 1 for r in rota]

        # --- compactação para bytes (uint16 para rota, uint8 para velocidades)
        rota_arr = np.array(rota, dtype=np.uint16)
        vels_arr = np.array(vels, dtype=np.uint8)

        # calcula hash (digest de 16 bytes)
        h = hashlib.blake2b(digest_size=16)
        h.update(rota_arr.tobytes())
        h.update(b'|')
        h.update(vels_arr.tobytes())
        key = h.digest()

        cached = self.fitness_cache.get(key)
        if cached is not None:
            return cached

        res = avaliar_rota_individual(ind, self.coord, self.vento, self.drone)
        self.fitness_cache.set(key, res)
        return res

    def inicializar_populacao(self):
        pop = []
        dv = self.drone.velocidades
        weights = [0.25] * (len(dv) - 6) + [0.75] * 6
        for _ in range(self.n_pop):
            perm = random.sample(self.base, len(self.base))
            rota = [1] + perm + [1]
            velocidades = [random.choices(dv, weights=weights, k=1)[0] for _ in range(len(rota)-1)]
            pop.append((rota, velocidades))
        return pop

    def avaliar_populacao(self, pop):
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            results = list(ex.map(self._avaliar_com_cache, pop))
        return results

    def selecionar_pais(self, avaliacoes, gen):
        t = gen / max(1, self.n_gen - 1)
        k = int(8 * (1 - t) + 2)
        contestants = random.sample(avaliacoes, min(k, len(avaliacoes)))
        return max(contestants, key=lambda x: x[0])[1]

    def executar(self, verbose=True):
        pop = self.inicializar_populacao()
        start_time = time.time()
        avaliacoes = self.avaliar_populacao(pop)
        avaliacoes.sort(reverse=True, key=lambda x: x[0])
        melhor_global = avaliacoes[0]
        estagnado = 0

        if verbose:
            print("=== AG PURO V4.0 — OTIMIZADO (SEM HEURÍSTICAS) ===")

        for gen in range(self.n_gen):
            t = gen / max(1, self.n_gen - 1)
            taxa_mut_atual = TAXA_MUT_INI + (TAXA_MUT_PICO - TAXA_MUT_INI) * math.sin(3.14159 * t)

            # Elitismo adaptativo
            elite_n = max(1, int(ELITISMO_INI * self.n_pop * (1 + t)))
            nova_pop = [avaliacoes[i][1][:2] for i in range(elite_n)]

            # Reintrodução de diversidade
            if gen > 0 and gen % REINJECAO_INTERVAL == 0:
                n_reinject = int(REINJECAO_TAXA * self.n_pop)
                for _ in range(n_reinject):
                    perm = random.sample(self.base, len(self.base))
                    rota = [1] + perm + [1]
                    rota = mutacao_inversao(rota, 0.8)
                    velocidades = [random.choice(self.drone.velocidades) for _ in range(len(rota)-1)]
                    nova_pop.append((rota, velocidades))

            # Reprodução
            while len(nova_pop) < self.n_pop:
                p1 = self.selecionar_pais(avaliacoes, gen)
                p2 = self.selecionar_pais(avaliacoes, gen)

                # Crossover híbrido
                if random.random() < 0.7:
                    filho_rota = pmx_crossover(p1[0], p2[0])
                else:
                    filho_rota = ox_crossover(p1[0], p2[0])

                # Mutação híbrida
                if random.random() < 0.6:
                    filho_rota = mutacao_inversao(filho_rota, taxa_mut_atual)
                else:
                    filho_rota = mutacao_swap(filho_rota, taxa_mut_atual * 1.5)

                # Velocidades
                filho_vels = [p1[1][i] if random.random() < 0.5 else p2[1][i] for i in range(len(p1[1]))]
                taxa_vel = taxa_mut_atual * 2.0 if gen < self.n_gen * 0.3 else taxa_mut_atual
                filho_vels = mutacao_velocidades(filho_vels, taxa_vel, self.drone)

                nova_pop.append((filho_rota, filho_vels))

            pop = nova_pop
            avaliacoes = self.avaliar_populacao(pop)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])

            if avaliacoes[0][0] > melhor_global[0]:
                melhor_global = avaliacoes[0]
                estagnado = 0
            else:
                estagnado += 1

            if estagnado > MAX_STAGNATION:
                if verbose:
                    print(f"\n>>> ESTAGNAÇÃO: Parando em G{gen+1} <<<")
                break

            if verbose and (gen % LOG_INTERVAL == 0 or gen == self.n_gen - 1):
                best = avaliacoes[0]
                dist = best[1][2]
                tempo = best[1][3] / 40.0
                elapsed = time.time() - start_time
                print(f"G{gen+1:4d} | Fit: {best[0]:.5f} | Dist: {dist:.1f}km | Tempo: ~{tempo:.0f}min | Rec: {best[1][4]} | Stag: {estagnado} | T:{elapsed:.1f}s")

        if verbose:
            print("=== FIM DO AG PURO V4.0 ===")
        return melhor_global[1], melhor_global[0]


# =============================
# CSV FINAL
# =============================

def gerar_csv_final(info, coord: Coordenadas, vento: Vento, arquivo_saida="rota_pmx_puro_otimizado.csv"):
    rota, velocidades, distancia_total, _, _, recargas = info
    linhas = []
    dia = 1
    hora_atual_seg = 6 * 3600
    recarga_set = {(id_, msg) for id_, msg in recargas}

    def s2hms(s):
        h = int(s // 3600) % 24
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        c1, c2 = coord.coordenadas[id1], coord.coordenadas[id2]
        velocidade = velocidades[i]
        i1, i2 = coord.idx_map[id1], coord.idx_map[id2]
        dist = float(coord.dist_matrix[i1, i2])
        az = float(coord.az_matrix[i1, i2])
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
        hora_atual_seg = hora_final_seg + 72
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
# EXECUÇÃO
# =============================

if __name__ == "__main__":
    print("=== AG PURO V4.0 — OTIMIZADO (SEM HEURÍSTICAS) ===\n")

    coord = Coordenadas("coordenadas.csv")
    vento = Vento("vento.csv")
    drone = Drone()

    ga = GeneticAlgorithmPuro(coord, vento, drone, seed=SEED)
    melhor_info, melhor_fit = ga.executar()

    print("\n" + "="*70)
    print("MELHOR SOLUÇÃO (AG PURO V4.0) — PATCHED")
    print("="*70)
    print(f"Fitness: {melhor_fit:.5f}")
    print(f"Distância Total: {melhor_info[2]:.2f} km")
    print(f"Tempo Estimado: ~{melhor_info[3]/40:.0f} min")
    print(f"Recargas Forçadas: {melhor_info[4]}")
    print(f"Rota (IDs): {melhor_info[0][:8]}... → {melhor_info[0][-1]}")
    print("="*70)

    gerar_csv_final(melhor_info, coord, vento)
