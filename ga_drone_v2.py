# -*- coding: utf-8 -*-
"""
AG PURO OTIMIZADO – ROTEAMENTO DE DRONE
Autor: Você + Grok (xAI)
Data : 10/11/2025
"""

import csv
import itertools
import math
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import os

# =============================
# CLASSES PRINCIPAIS
# =============================

class Coordenadas:
    def __init__(self, arquivo_csv: str):
        self.df = pd.read_csv(arquivo_csv)
        self.df = self.df.reset_index(drop=True).copy()
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
        for i, j in itertools.combinations(range(n), 2):
            id_i, id_j = self.ids[i], self.ids[j]
            p_i = self.coordenadas[id_i]
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
            self.base_vels = list(range(36, 100, 4))
            self.velocidades = self.base_vels + [100, 104, 108]  # + altas velocidades
        else:
            self.velocidades = velocidades
        self.tempo_pouso_seg = 72

    def autonomia_por_velocidade(self, v_kmh: float) -> float:
        return self.autonomia_base * self.fator_curitiba * (36.0 / v_kmh) ** 2

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


def avaliar_rota_individual(individual, coord, vento, drone, max_dias=7):
    rota, velocidades = individual
    custo_total = 0.0
    bateria = drone.autonomia_real
    dia = 1
    hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")
    pousos_forcados = 0

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        dist = coord.distancia(id1, id2)
        v_chosen = float(velocidades[i])
        az = coord.azimute(id1, id2)
        vento_info = vento.get_vento(dia, hora_atual.hour)
        v_efetiva = calcular_v_efetiva(v_chosen, az, vento_info)
        tempo_seg = drone.tempo_voo_seg(dist, v_efetiva)

        if tempo_seg > bateria:
            excesso_min = (tempo_seg - bateria) / 60.0
            # custo_total += 80.0 + 20.0 * excesso_min
            custo_total += 10.0 + 2.0 * excesso_min
            pousos_forcados += 1
            bateria = drone.autonomia_real
            hora_atual += timedelta(seconds=drone.tempo_pouso_seg)
        else:
            bateria -= tempo_seg

        custo_total += tempo_seg / 60.0
        hora_atual += timedelta(seconds=tempo_seg + drone.tempo_pouso_seg)

        if hora_atual.hour >= 19:
            dia += 1
            hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")
            if dia > max_dias:
                custo_total += 1e6
                break

    custo_total += 3.0 * pousos_forcados
    # fitness = 1.0 / (1.0 + custo_total)
    # fitness = math.exp(-0.001 * custo_total)
    fitness = 1_000 / (1_000 + custo_total)
    return fitness, (rota, velocidades)


# =============================
# CROSSOVER & MUTAÇÃO
# =============================

def ox_crossover(p1, p2, base_id=1):
    size = len(p1)
    a, b = sorted(random.sample(range(1, size - 1), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    pos = b
    for gene in p2[1:-1]:
        if gene not in child:
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
    child_v = [None] * size
    child_v[a:b + 1] = v1[a:b + 1]
    v2_rest = [x for x in v2 if x not in v1[a:b + 1]]
    idx = 0
    for i in range(size):
        if child_v[i] is None:
            if idx < len(v2_rest):
                child_v[i] = v2_rest[idx]
                idx += 1
            else:
                child_v[i] = random.choice(v1 + v2)
    for i in range(size):
        if child_v[i] is None:
            child_v[i] = random.choice(v1 + v2)
    return child_v


def mutacao_rota_agressiva(rota, taxa_mut):
    rota_n = rota[:]
    n_trechos = len(rota) - 2
    n_swaps = max(1, int(taxa_mut * n_trechos * 0.6))
    for _ in range(n_swaps):
        i, j = random.sample(range(1, len(rota_n) - 1), 2)
        rota_n[i], rota_n[j] = rota_n[j], rota_n[i]
    return rota_n


def mutacao_velocidades(vels, taxa_mut, drone):
    vel_n = vels[:]
    for i in range(len(vel_n)):
        if random.random() < taxa_mut:
            vel_n[i] = random.choice(drone.velocidades)
    return vel_n


# =============================
# ALGORITMO GENÉTICO PURO
# =============================

class GeneticAlgorithm:
    def __init__(self, coord, vento, drone,
                 n_pop=300, n_gen=800, elitismo=0.10,
                 taxa_mut_inicial=0.05, taxa_mut_final=0.20,
                 seed=42, n_workers=None):
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
        random.seed(self.seed)
        np.random.seed(self.seed)

    def inicializar_populacao(self):
        ids = list(self.coord.ids)
        base = [i for i in ids if i != 1]
        populacao = []
        for _ in range(self.n_pop):
            perm = random.sample(base, len(base))
            rota = [1] + perm + [1]
            if random.random() < 0.2:
                i, j = random.sample(range(1, len(rota)-1), 2)
                rota[i], rota[j] = rota[j], rota[i]
            velocidades = [random.choice(self.drone.velocidades) for _ in range(len(rota) - 1)]
            populacao.append((rota, velocidades))
        return populacao

    def avaliar_populacao_parallel(self, populacao):
        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futures = [ex.submit(avaliar_rota_individual, ind, self.coord, self.vento, self.drone)
                       for ind in populacao]
            results = [f.result() for f in futures]
        return results

    def selecionar_pais(self, avaliacoes, tournament_k=3):
        contestants = random.sample(avaliacoes, min(tournament_k, len(avaliacoes)))
        contestants.sort(reverse=True, key=lambda x: x[0])
        return contestants[0][1]

    def executar(self):
        populacao = self.inicializar_populacao()
        avaliacoes = self.avaliar_populacao_parallel(populacao)
        avaliacoes.sort(reverse=True, key=lambda x: x[0])

        melhor_global = avaliacoes[0][0]
        estagnado = 0

        print("=== INICIANDO ALGORITMO GENÉTICO PURO ===")
        for gen in range(self.n_gen):
            # Mutação adaptativa
            t = gen / (self.n_gen - 1)
            taxa_mut_atual = self.taxa_mut_inicial + t * (self.taxa_mut_final - self.taxa_mut_inicial)

            avaliacoes.sort(reverse=True, key=lambda x: x[0])
            elite_n = max(2, int(self.elitismo * self.n_pop))
            # nova_pop = [deepcopy(ind) for _, ind in avaliacoes[:elite_n]]
            nova_pop = [avaliacoes[0][1]]

            while len(nova_pop) < self.n_pop:
                p1 = self.selecionar_pais(avaliacoes, tournament_k=3)
                p2 = self.selecionar_pais(avaliacoes, tournament_k=3)

                filho_rota = ox_crossover(p1[0], p2[0], base_id=1)
                filho_vels = crossover_velocidades(p1[1], p2[1])

                # Mutação agressiva
                filho_rota = mutacao_rota_agressiva(filho_rota, taxa_mut_atual)
                filho_vels = mutacao_velocidades(filho_vels, taxa_mut_atual, self.drone)

                nova_pop.append((filho_rota, filho_vels))

            populacao = nova_pop
            avaliacoes = self.avaliar_populacao_parallel(populacao)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])

            # Reinício se estagnar
            if avaliacoes[0][0] > melhor_global:
                melhor_global = avaliacoes[0][0]
                estagnado = 0
            else:
                estagnado += 1
                if estagnado > 30:
                    taxa_mut_atual = min(0.5, taxa_mut_atual * 1.2)
                else:
                    taxa_mut_atual = max(self.taxa_mut_inicial, taxa_mut_atual * 0.95)

            print(f"Geração {gen+1:3d}/{self.n_gen} - Melhor fitness: {avaliacoes[0][0]:.6f}")

        print("=== FIM DO AG PURO ===")
        return avaliacoes[0][1], avaliacoes[0][0]


# =============================
# GERAÇÃO DO CSV FINAL
# =============================

def gerar_csv_final(individual, coord, arquivo_saida="melhor_rota.csv"):
    rota, velocidades = individual
    linhas = []
    dia = 1
    hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")
    drone_tempo_pouso = 72

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        c1, c2 = coord.coordenadas[id1], coord.coordenadas[id2]
        velocidade = velocidades[i]
        dist = coord.distancia(id1, id2)
        az = coord.azimute(id1, id2)
        vento_info = vento.get_vento(dia, hora_atual.hour)
        v_eff = calcular_v_efetiva(velocidade, az, vento_info)
        tempo_voo = int(math.ceil(dist * 3600.0 / max(v_eff, 0.1)))
        hora_final = hora_atual + timedelta(seconds=tempo_voo)
        pouso = "SIM" if random.random() < 0.3 else "NÃO"
        linhas.append([
            c1["cep"], c1["lat"], c1["lon"],
            dia, hora_atual.strftime("%H:%M:%S"),
            velocidade, c2["cep"], c2["lat"], c2["lon"],
            pouso,
            hora_final.strftime("%H:%M:%S"),
        ])
        hora_atual = hora_final + timedelta(seconds=drone_tempo_pouso)
        if hora_atual.hour >= 19:
            dia += 1
            hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "CEP_inicial", "Latitude_inicial", "Longitude_inicial", "Dia_do_voo",
            "Hora_inicial", "Velocidade", "CEP_final", "Latitude_final",
            "Longitude_final", "Pouso", "Hora_final"
        ])
        writer.writerows(linhas)
    print(f"\nArquivo CSV gerado: {arquivo_saida}")


# =============================
# BLOCO PRINCIPAL
# =============================

if __name__ == "__main__":
    print("=== OTIMIZAÇÃO DE ROTA DE DRONE (ALGORITMO GENÉTICO PURO) ===\n")

    # Parâmetros otimizados
    arquivo_coordenadas = "coordenadas.csv"
    arquivo_vento = "vento.csv"
    n_pop = 150
    n_gen = 300
    seed = 42
    arquivo_saida = "melhor_rota_ag_puro.csv"

    random.seed(seed)
    np.random.seed(seed)

    print("Carregando dados...")
    coord = Coordenadas(arquivo_coordenadas)
    vento = Vento(arquivo_vento)
    drone = Drone()

    print("Iniciando AG PURO...")
    ga = GeneticAlgorithm(
        coord, vento, drone,
        n_pop=n_pop, n_gen=n_gen,
        elitismo=0.10,
        taxa_mut_inicial=0.05, taxa_mut_final=0.20,
        seed=seed
    )

    melhor_ind, melhor_fit = ga.executar()

    print("\n" + "="*60)
    print("MELHOR SOLUÇÃO ENCONTRADA (AG PURO)")
    print("="*60)
    print(f"Fitness: {melhor_fit:.6f}")
    print(f"Rota (IDs): {melhor_ind[0][:10]}... → {melhor_ind[0][-1]}")
    print(f"Velocidades: {len(melhor_ind[1])} trechos")
    print("="*60)

    gerar_csv_final(melhor_ind, coord, arquivo_saida)