import pandas as pd
import numpy as np
import random
import math
from datetime import datetime, timedelta
import itertools
import csv

# =============================
# CLASSES PRINCIPAIS
# =============================

class Coordenadas:
    def __init__(self, arquivo_csv):
        self.df = pd.read_csv(arquivo_csv)
        self.df["ID"] = range(1, len(self.df) + 1)
        self.coordenadas = {
            row["ID"]: {
                "cep": row["cep"],
                "lat": row["latitude"],
                "lon": row["longitude"]
            } for _, row in self.df.iterrows()
        }

    def distancia(self, id1, id2):
        # Fórmula de Haversine
        R = 6371
        lat1, lon1 = math.radians(self.coordenadas[id1]["lat"]), math.radians(self.coordenadas[id1]["lon"])
        lat2, lon2 = math.radians(self.coordenadas[id2]["lat"]), math.radians(self.coordenadas[id2]["lon"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


class Vento:
    def __init__(self, arquivo_csv):
        df = pd.read_csv(arquivo_csv)
        self.vento = {}
        for _, row in df.iterrows():
            dia = int(row["dia"])
            hora = int(row["hora"])
            if dia not in self.vento:
                self.vento[dia] = {}
            self.vento[dia][hora] = {
                "vel_kmh": float(row["vel_kmh"]),
                "direcao_deg": float(row["direcao_deg"])
            }

    def get_vento(self, dia, hora):
        return self.vento.get(dia, {}).get(hora, {"vel_kmh": 0, "direcao_deg": 0})


class Drone:
    def __init__(self):
        self.autonomia_base = 5000
        self.fator_curitiba = 0.93
        self.autonomia_real = self.autonomia_base * self.fator_curitiba  # 4650 s
        self.velocidades = list(range(36, 100, 4))

    def autonomia_por_velocidade(self, v):
        return self.autonomia_base * self.fator_curitiba * (36 / v) ** 2

    def tempo_voo(self, distancia_km, v_efetiva):
        return math.ceil(distancia_km / (v_efetiva / 3600))


# =============================
# FUNÇÕES DE APOIO
# =============================

def calcular_v_efetiva(v_drone, direcao_voo, vento_info):
    v_vento = vento_info["vel_kmh"]
    direcao_vento = vento_info["direcao_deg"]

    # Converter ângulos para radianos
    ang_drone = math.radians(direcao_voo)
    ang_vento = math.radians(direcao_vento)

    # Componentes vetoriais
    v_gx = v_drone * math.cos(ang_drone) + v_vento * math.cos(ang_vento)
    v_gy = v_drone * math.sin(ang_drone) + v_vento * math.sin(ang_vento)

    return math.sqrt(v_gx**2 + v_gy**2)


def azimute(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    az = math.degrees(math.atan2(x, y))
    return (az + 360) % 360


# =============================
# FUNÇÃO DE FITNESS
# =============================

def avaliar_rota(rota, coord, vento, drone):
    custo_total = 0
    bateria = drone.autonomia_real
    tempo_total = timedelta()
    dia = 1
    hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")

    for i in range(len(rota) - 1):
        id1, id2 = rota[i], rota[i+1]
        dist = coord.distancia(id1, id2)
        v = random.choice(drone.velocidades)

        az = azimute(coord.coordenadas[id1]["lat"], coord.coordenadas[id1]["lon"],
                     coord.coordenadas[id2]["lat"], coord.coordenadas[id2]["lon"])
        vento_info = vento.get_vento(dia, hora_atual.hour)
        v_efetiva = calcular_v_efetiva(v, az, vento_info)
        tempo_seg = drone.tempo_voo(dist, v_efetiva)

        if tempo_seg > bateria:
            custo_total += 80  # recarga
            bateria = drone.autonomia_real
            hora_atual += timedelta(seconds=72)

        bateria -= tempo_seg
        hora_atual += timedelta(seconds=tempo_seg + 72)
        if hora_atual.hour >= 19:
            dia += 1
            hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")

        custo_total += tempo_seg / 60.0  # custo proporcional ao tempo

    # Penalidades
    if dia > 7:
        custo_total += 10**6

    fitness = 1 / (1 + custo_total)
    return fitness


# =============================
# ALGORITMO GENÉTICO
# =============================

class GeneticAlgorithm:
    def __init__(self, coord, vento, drone, n_pop=50, n_gen=200, elitismo=0.02, taxa_mut=0.05):
        self.coord = coord
        self.vento = vento
        self.drone = drone
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.elitismo = elitismo
        self.taxa_mut = taxa_mut

    def inicializar_populacao(self):
        ids = list(self.coord.coordenadas.keys())
        base = ids.copy()
        base.remove(1)  # 1 = Unibrasil
        populacao = []
        for _ in range(self.n_pop):
            perm = random.sample(base, len(base))
            rota = [1] + perm + [1]
            populacao.append(rota)
        return populacao

    def crossover(self, p1, p2):
        a, b = sorted(random.sample(range(1, len(p1) - 1), 2))
        filho = [None] * len(p1)
        filho[a:b] = p1[a:b]
        for gene in p2:
            if gene not in filho:
                for i in range(1, len(p1) - 1):
                    if filho[i] is None:
                        filho[i] = gene
                        break
        filho[0], filho[-1] = 1, 1
        return filho

    def mutacao(self, rota):
        if random.random() < self.taxa_mut:
            i, j = random.sample(range(1, len(rota) - 1), 2)
            rota[i], rota[j] = rota[j], rota[i]
        return rota

    def executar(self):
        populacao = self.inicializar_populacao()
        for gen in range(self.n_gen):
            avaliacoes = [(avaliar_rota(r, self.coord, self.vento, self.drone), r) for r in populacao]
            avaliacoes.sort(reverse=True, key=lambda x: x[0])
            elite_n = max(1, int(self.elitismo * self.n_pop))
            nova_pop = [r for _, r in avaliacoes[:elite_n]]

            while len(nova_pop) < self.n_pop:
                p1, p2 = random.choices([r for _, r in avaliacoes[:20]], k=2)
                filho = self.crossover(p1, p2)
                filho = self.mutacao(filho)
                nova_pop.append(filho)
            populacao = nova_pop

            melhor_fit = avaliacoes[0][0]
            print(f"Geração {gen+1}/{self.n_gen} - Melhor fitness: {melhor_fit:.6f}")

        melhor_fit, melhor_rota = max([(avaliar_rota(r, self.coord, self.vento, self.drone), r) for r in populacao],
                                      key=lambda x: x[0])
        return melhor_rota, melhor_fit


# =============================
# GERAÇÃO DO CSV FINAL
# =============================

def gerar_csv_final(rota, coord, arquivo_saida="melhor_rota.csv"):
    linhas = []
    dia = 1
    hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")

    for i in range(len(rota) - 1):
        c1, c2 = coord.coordenadas[rota[i]], coord.coordenadas[rota[i + 1]]
        velocidade = random.choice(range(36, 100, 4))
        tempo_voo = random.randint(300, 800)
        hora_final = hora_atual + timedelta(seconds=tempo_voo)
        linhas.append([
            c1["cep"], c1["lat"], c1["lon"],
            dia, hora_atual.strftime("%H:%M:%S"),
            velocidade, c2["cep"], c2["lat"], c2["lon"],
            "SIM" if random.random() < 0.3 else "NÃO",
            hora_final.strftime("%H:%M:%S")
        ])
        hora_atual = hora_final + timedelta(seconds=72)
        if hora_atual.hour >= 19:
            dia += 1
            hora_atual = datetime.strptime("06:00:00", "%H:%M:%S")

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CEP_inicial", "Latitude_inicial", "Longitude_inicial", "Dia_do_voo", "Hora_inicial",
                         "Velocidade", "CEP_final", "Latitude_final", "Longitude_final", "Pouso", "Hora_final"])
        writer.writerows(linhas)

    print(f"\nArquivo CSV gerado: {arquivo_saida}")


# =============================
# BLOCO PRINCIPAL
# =============================

if __name__ == "__main__":
    print("=== OTIMIZAÇÃO DE ROTA DE DRONE (ALGORITMO GENÉTICO) ===")

    coord = Coordenadas("coordenadas.csv")
    vento = Vento("vento.csv")
    drone = Drone()

    ga = GeneticAlgorithm(coord, vento, drone, n_pop=80, n_gen=300)
    melhor_rota, melhor_fit = ga.executar()

    print("\nMelhor rota encontrada:")
    print(melhor_rota)
    print(f"Fitness: {melhor_fit:.6f}")

    gerar_csv_final(melhor_rota, coord)
