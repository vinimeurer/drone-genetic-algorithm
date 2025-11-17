# src/io_csv.py
import csv
import math
from typing import Tuple, List
from .coordenadas import Coordenadas
from .vento import Vento
from .drone import Drone

def reavaliar_preciso(info, coord: Coordenadas, vento: Vento, drone: Drone, max_dias: int = 31):
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

        vento_info = vento.get_vento(dia, int(hora_atual_seg // 3600))
        w_vel = vento_info["vel_kmh"]
        w_dir = vento_info["direcao_deg"]
        dir_to = (w_dir + 180.0) % 360.0

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


def gerar_csv_final(info, coord: Coordenadas, vento: Vento, arquivo_saida: str = "rota.csv"):
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
