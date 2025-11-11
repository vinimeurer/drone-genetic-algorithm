# -*- coding: utf-8 -*-
"""
AG PURO V3.1 – ROTEAMENTO DE DRONE (OTIMIZADO)
Autor: Você + Grok (xAI) -> Versão otimizada
Data: 11/11/2025
Melhorias: ThreadPoolExecutor, caches, uso de segundos (sem datetime),
         v_eff cache, parada antecipada, menos overhead por chamada.
"""

import csv
import itertools
import math
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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

    # Mantemos estes métodos como conveniência (mas na avaliação usamos matrizes diretas)
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
        self.autonomia_real = self.autonomia_base * self.fator_curitiba  # em segundos (considere unidade coerente)
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
# FUNÇÕES AUXILIARES
# =============================

def calcular_v_efetiva(v_drone_kmh: float, direcao_voo_deg: float, vento_info: Dict[str, float]) -> float:
    # Mantivemos implementação original: converte direção do vento (para onde ele vai)
    v_wind = vento_info.get("vel_kmh", 0.0)
    dir_wind_to = (vento_info.get("direcao_deg", 0.0) + 180.0) % 360.0
    ang_drone = math.radians(direcao_voo_deg)
    ang_wind = math.radians(dir_wind_to)
    vx = v_drone_kmh * math.cos(ang_drone) + v_wind * math.cos(ang_wind)
    vy = v_drone_kmh * math.sin(ang_drone) + v_wind * math.sin(ang_wind)
    v_eff = math.hypot(vx, vy)
    return max(v_eff, 0.1)


# =============================
# AVALIAÇÃO (OTIMIZADA)
# =============================

def avaliar_rota_individual(individual, coord: Coordenadas, vento: Vento, drone: Drone, max_dias=7):
    """
    Versão otimizada da função de avaliação:
    - usa acesso direto a matrizes (dist_matrix, az_matrix)
    - usa segundos inteiros (sem datetime)
    - usa cache simples de v_eff para evitar trig repetido
    """
    rota, velocidades = individual
    custo_total = 0.0
    distancia_total = 0.0
    bateria = drone.autonomia_real
    dia = 1
    hora_atual_seg = 6 * 3600  # 06:00 em segundos
    pousos_forcados = 0
    recargas = []

    idx_map = coord.idx_map
    dist_matrix = coord.dist_matrix
    az_matrix = coord.az_matrix

    v_eff_cache = {}  # cache simples: (v_drone_rounded, az_rounded, vel_vento_rounded, dir_vento_rounded) -> v_eff

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i + 1]
        i1, i2 = idx_map[id1], idx_map[id2]
        dist = float(dist_matrix[i1, i2])
        distancia_total += dist

        v_chosen = float(velocidades[i])
        az = float(az_matrix[i1, i2])

        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        # cache key (pequena discretização)
        key = (int(round(v_chosen)), int(round(az)) % 360, int(round(vento_info["vel_kmh"])), int(round(vento_info["direcao_deg"])) )
        v_efetiva = v_eff_cache.get(key)
        if v_efetiva is None:
            v_efetiva = calcular_v_efetiva(v_chosen, az, vento_info)
            v_eff_cache[key] = v_efetiva

        tempo_seg = drone.tempo_voo_seg(dist, v_efetiva)

        # Penalidade por bateria
        if tempo_seg > bateria:
            excesso_min = (tempo_seg - bateria) / 60.0
            custo_total += 15.0 + 3.0 * excesso_min
            pousos_forcados += 1
            recargas.append((id1, "RECARGA FORÇADA"))
            bateria = drone.autonomia_real
            hora_atual_seg += drone.tempo_pouso_seg
        else:
            bateria -= tempo_seg

        # Custo: tempo (min) + distância (km ponderada)
        custo_total += (tempo_seg / 60.0) + (dist * 40.0)  # 40 = peso da distância

        hora_atual_seg += tempo_seg + drone.tempo_pouso_seg

        # checa fim do dia (>=19h)
        if hora_atual_seg >= 19 * 3600:
            dia += 1
            hora_atual_seg = 6 * 3600
            if dia > max_dias:
                custo_total += 1e7
                break

    custo_total += 5.0 * pousos_forcados

    # FITNESS: quanto menor o custo, maior o fitness
    custo_ref = 60000  # referência: ~1000km + 1000min
    fitness = 1.0 / (1.0 + (custo_total / custo_ref) ** 1.8)
    fitness = max(0.01, min(0.99, fitness))
    fitness = round(fitness, 5)

    return fitness, (rota, velocidades, distancia_total, custo_total, pousos_forcados, recargas)


# =============================
# CROSSOVER & MUTAÇÃO (mantidos)
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
    # preenche
    res = child_v + rest[:size - len(child_v)]
    if len(res) < size:
        res += [random.choice(v1 + v2) for _ in range(size - len(res))]
    return res


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
# ALGORITMO GENÉTICO V3.1 (OTIMIZADO)
# =============================

class GeneticAlgorithm:
    def __init__(self, coord, vento, drone,
                 n_pop=150, n_gen=500, elitismo=0.12,
                 taxa_mut_inicial=0.07, taxa_mut_final=0.35,
                 seed=42, n_workers=None, max_stagnation=120):
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
        self.base = [i for i in self.coord.ids if i != 1]
        self.max_stagnation = max_stagnation

    def inicializar_populacao(self):
        populacao = []
        for _ in range(self.n_pop):
            perm = random.sample(self.base, len(self.base))
            rota = [1] + perm + [1]
            velocidades = [random.choices(
                self.drone.velocidades,
                weights=[0.25]*max(1, len(self.drone.velocidades[:-6])) + [0.75]*6,
                k=1)[0] for _ in range(len(rota) - 1)]
            populacao.append((rota, velocidades))
        return populacao

    # Agora executa avaliações via ThreadPoolExecutor para evitar pickle overhead
    def avaliar_populacao_parallel(self, populacao, ex):
        futures = [ex.submit(avaliar_rota_individual, ind, self.coord, self.vento, self.drone)
                   for ind in populacao]
        results = [f.result() for f in futures]
        return results

    def selecionar_pais(self, avaliacoes, gen):
        k = 5 if gen < self.n_gen * 0.25 else 3
        contestants = random.sample(avaliacoes, min(k, len(avaliacoes)))
        return max(contestants, key=lambda x: x[0])[1]

    def executar(self, verbose=True):
        populacao = self.inicializar_populacao()

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            avaliacoes = self.avaliar_populacao_parallel(populacao, ex)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])

            melhor_global = avaliacoes[0]
            estagnado = 0

            if verbose:
                print("=== INICIANDO AG PURO V3.1 (OTIMIZADO) ===")
            for gen in range(self.n_gen):
                t = gen / (self.n_gen - 1) if self.n_gen > 1 else 1.0
                taxa_mut_atual = self.taxa_mut_inicial + t * (self.taxa_mut_final - self.taxa_mut_inicial)

                elite_n = max(3, int(self.elitismo * self.n_pop))
                nova_pop = [avaliacoes[i][1][:2] for i in range(elite_n)]

                # Diversidade (pequeno número)
                for _ in range(4):
                    perm = random.sample(self.base, len(self.base))
                    rota = [1] + perm + [1]
                    rota = mutacao_inversao(rota, taxa_mut_atual * 1.8)
                    velocidades = [random.choices(
                        self.drone.velocidades,
                        weights=[0.2]*max(1, len(self.drone.velocidades[:-6])) + [0.8]*6,
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

                # Reinício catastrófico (mantido)
                if estagnado > 40:
                    if verbose:
                        print(f"\n>>> REINÍCIO INTELIGENTE na G{gen+1} <<<")
                    nova_pop = [melhor_global[1][:2]]
                    for _ in range(self.n_pop - 1):
                        perm = random.sample(self.base, len(self.base))
                        rota = [1] + perm + [1]
                        rota = mutacao_inversao(rota, 0.75)
                        velocidades = [random.choices(
                            self.drone.velocidades,
                            weights=[0.15]*max(1, len(self.drone.velocidades[:-6])) + [0.85]*6,
                            k=1)[0] for _ in range(len(rota)-1)]
                        nova_pop.append((rota, velocidades))
                    populacao = nova_pop
                    avaliacoes = self.avaliar_populacao_parallel(populacao, ex)
                    avaliacoes.sort(reverse=True, key=lambda x: x[0])
                    estagnado = 0

                # parada antecipada por estagnação longa
                if estagnado > self.max_stagnation:
                    if verbose:
                        print(f"\nEncerrando por estagnação (G{gen+1})")
                    break

                if verbose and (gen % max(1, int(self.n_gen/50)) == 0 or gen < 10 or gen == self.n_gen-1):
                    best = avaliacoes[0]
                    dist = best[1][2]
                    tempo = best[1][3] / 40  # aproximado
                    print(f"G{gen+1:4d} | Fit: {best[0]:.5f} | Dist: {dist:.1f}km | Tempo: ~{tempo:.0f}min | Rec: {best[1][4]} | Stag: {estagnado}")

        if verbose:
            print("=== FIM DO AG PURO V3.1 ===")
        return melhor_global[1], melhor_global[0]


# =============================
# GERAÇÃO DO CSV FINAL (REAL) - agora recebe vento explicitamente
# =============================

def gerar_csv_final(info, coord: Coordenadas, vento: Vento, arquivo_saida="melhor_rota_ag_puro_v3_1.csv"):
    rota, velocidades, distancia_total, _, _, recargas = info
    linhas = []
    dia = 1
    hora_atual_seg = 6 * 3600
    drone_tempo_pouso = 72
    recarga_set = {(id_, msg) for id_, msg in recargas}

    idx_map = coord.idx_map
    dist_matrix = coord.dist_matrix
    az_matrix = coord.az_matrix

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
        # converter segundos para H:M:S
        def s2hms(s):
            h = int(s // 3600) % 24
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"
        linhas.append([
            c1["cep"], c1["lat"], c1["lon"],
            dia, s2hms(hora_atual_seg),
            velocidade, c2["cep"], c2["lat"], c2["lon"],
            pouso,
            s2hms(hora_final_seg),
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
# BLOCO PRINCIPAL (EXEMPLO)
# =============================

if __name__ == "__main__":
    print("=== OTIMIZAÇÃO DE ROTA DE DRONE (AG PURO V3.1 OTIMIZADO) ===\n")

    arquivo_coordenadas = "coordenadas.csv"
    arquivo_vento = "vento.csv"
    seed = 42
    arquivo_saida = "melhor_rota_ag_puro_v3_1.csv"

    random.seed(seed)
    np.random.seed(seed)

    print("Carregando dados...")
    coord = Coordenadas(arquivo_coordenadas)
    vento = Vento(arquivo_vento)
    drone = Drone()

    print("Iniciando AG PURO V3.1...")
    ga = GeneticAlgorithm(
        coord, vento, drone,
        n_pop=150, n_gen=500,      # valores padrão mais moderados
        elitismo=0.25,
        taxa_mut_inicial=0.07, taxa_mut_final=0.45,
        seed=seed,
        n_workers=max(2, (os.cpu_count() or 2) - 1),
        max_stagnation=200
    )

    melhor_info, melhor_fit = ga.executar()

    print("\n" + "="*70)
    print("MELHOR SOLUÇÃO ENCONTRADA (AG PURO V3.1)")
    print("="*70)
    print(f"Fitness: {melhor_fit:.5f}")
    print(f"Distância Total: {melhor_info[2]:.2f} km")
    print(f"Tempo Estimado: ~{melhor_info[3]/40:.0f} min")
    print(f"Recargas Forçadas: {melhor_info[4]}")
    print(f"Rota (IDs): {melhor_info[0][:8]}... → {melhor_info[0][-1]}")
    print("="*70)

    gerar_csv_final(melhor_info, coord, vento, arquivo_saida)
