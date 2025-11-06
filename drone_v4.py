#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA para roteamento de drone (TSP com janelas de tempo, vento e bateria)
Autor: Assistente (exemplo didático)
Descrição:
  - Implementa um Algoritmo Genético para planejar rota de um drone
    que parte e retorna ao CEP 82821020 (Unibrasil), visita cada CEP uma vez,
    em até 7 dias, voando apenas entre 06:00 e 19:00.
  - Considera vento (por dia/hora, tabela simulada), autonomia de bateria,
    custos de recarga e penalidades para soluções inviáveis.
  - Código didático, modular e comentado seguindo os 10 passos solicitados.

Passos (mapeados no código):
 1) entender objetivo        -> docstring + comments
 2) coletar dados            -> função load_ceps (simula dados se faltarem)
 3) calcular distâncias      -> haversine_distance
 4) modelar vento            -> wind_table + modelar_vento / efetivo
 5) calcular tempos          -> travel_time (com efeito do vento)
 6) definir cromossomo       -> permutation de CEPs
 7) implementar fitness      -> funcao_fitness + simulate_route
 8) operadores genéticos     -> crossover OX, mutation swap, seleção torneio
 9) executar algoritmo       -> main: roda GA, imprime e salva CSV
10) gerar CSV               -> save_route_csv

Observações/Assunções (conforme solicitado):
 - Sem internet: dados de CEPs e ventos são simulados (placeholders).
 - Matrícula começa com '2' (portanto aplicamos fator 0.93).
 - Autonomia base: 5000 segundos, aplicamos fator 0.93 => 4650s autonomia base.
 - Velocidades coerentes: usamos velocidades típicas ~36 km/h (10 m/s).
 - Custos de recarga: R$80 por recarga; se recarga ocorrer após 17:00, +R$80 extra.
 - Recarregamos quando necessário em qualquer ponto visitado (modelo simplificado).
 - Penalty alto se rota não completar em <=7 dias ou violar janelas de voo.
"""

import math
import random
import csv
import sys
import os
from datetime import datetime, timedelta, time
from typing import List, Tuple, Dict, Any

# --- IMPORTS E FIXES BÁSICOS ---
# libraries: numpy não é estritamente necessário - usaremos math/random
# random seed para reprodutibilidade durante testes
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# -----------------------------
# 1) ENTENDER OBJETIVO (comentários)
# -----------------------------
# Objetivo: Minimizar custo total = tempo de voo (econômico) + custo de recargas,
# respeitando janelas de voo (06:00-19:00), autonomia (bateria), efeitos do vento,
# limite de 7 dias, e visitar cada CEP uma vez (começando e terminando no CEP base).


# -----------------------------
# 2) COLETAR DADOS (simulações / placeholders)
# -----------------------------
def load_ceps() -> Tuple[Dict[str, Tuple[float, float]], str]:
    """
    Carrega CEP -> (lat, lon) a partir de 'coordenadas.csv' (se existir).
    Caso o arquivo não exista ou haja erro, usa a tabela simulada (fallback).
    Espera colunas: cep, longitude, latitude (ou variantes).
    """
    csv_path = os.path.join(os.path.dirname(__file__), "coordenadas.csv")
    cep_base = "82821020"
    dict_cep_coords: Dict[str, Tuple[float, float]] = {}

    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # tentar várias variantes de nomes de coluna
                    cep = (row.get("cep") or row.get("CEP") or row.get("Cep") or "").strip()
                    lat_s = (row.get("latitude") or row.get("lat") or row.get("Latitude") or "").strip()
                    lon_s = (row.get("longitude") or row.get("lon") or row.get("Longitude") or "").strip()
                    if not cep or not lat_s or not lon_s:
                        continue
                    try:
                        lat = float(lat_s)
                        lon = float(lon_s)
                        dict_cep_coords[cep] = (lat, lon)
                    except ValueError:
                        # pular linhas com valores inválidos
                        continue
            # garantir base presente; se não, adicionar fallback próximo
            if cep_base not in dict_cep_coords:
                # se arquivo fornecido não contém o base, mantemos coordenada padrão razoável
                dict_cep_coords[cep_base] = (-25.4300, -49.2450)
            return dict_cep_coords, cep_base
        except Exception as e:
            print(f"[WARN] erro lendo {csv_path}, usando dados simulados: {e}")

    return dict_cep_coords, cep_base


# -----------------------------
# 3) CALCULAR DISTÂNCIAS
# -----------------------------
def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calcula distância em metros entre dois pontos (lat, lon) usando fórmula Haversine.
    """
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0  # raio da Terra em metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    hav = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))
    return R * c  # metros


# -----------------------------
# 4) MODELAR VENTO
# -----------------------------
def generate_wind_table(num_days: int = 7) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Gera/Carrega tabela de vento:
      - se existir 'vento.csv' no mesmo diretório, carrega esse arquivo (espera colunas: dia,hora,vel_kmh,direcao_deg)
      - caso contrário, gera tabela simulada (comportamento anterior).
    Retorna wind_table[day_index][hour] = (vel_m_s, direction_deg)
    """
    csv_path = os.path.join(os.path.dirname(__file__), "vento.csv")
    if os.path.exists(csv_path):
        wind_table: Dict[int, Dict[int, Tuple[float, float]]] = {}
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dia = int(row.get("dia") or row.get("day") or 0)
                        hora = int(row.get("hora") or row.get("hour") or 0)
                        vel_kmh = float(row.get("vel_kmh") or row.get("velocidade_kmh") or row.get("vel") or 0.0)
                        direc = float(row.get("direcao_deg") or row.get("direction_deg") or row.get("direcao") or 0.0)
                    except (ValueError, TypeError):
                        continue
                    day_index = max(0, dia - 1)
                    if day_index not in wind_table:
                        wind_table[day_index] = {}
                    # converter km/h -> m/s
                    wind_table[day_index][hora] = (vel_kmh / 3.6, direc % 360.0)
            # preencher horas faltantes por cópia da hora mais próxima ou zeros
            for d in list(wind_table.keys()):
                hours_present = sorted(wind_table[d].keys())
                if not hours_present:
                    wind_table[d] = {h: (0.0, 0.0) for h in range(24)}
                    continue
                for h in range(24):
                    if h not in wind_table[d]:
                        # escolher hora mais próxima presente
                        nearest = min(hours_present, key=lambda x: abs(x - h))
                        wind_table[d][h] = wind_table[d][nearest]
            # se número de dias menor que solicitado, repetir ciclicamente até num_days
            if len(wind_table) < num_days:
                for d in range(num_days):
                    if d not in wind_table:
                        wind_table[d] = wind_table[d % max(1, len(wind_table))]
            return wind_table
        except Exception as e:
            print(f"[WARN] erro lendo {csv_path}, gerando tabela simulada: {e}")

    # Fallback: geração simulada (comportamento original)
    wind_table = {}
    for d in range(num_days):
        wind_table[d] = {}
        for h in range(24):
            base_speed = 2.0 + 3.0 * math.sin((h - 12) / 24.0 * 2 * math.pi)
            speed = max(0.0, base_speed + random.uniform(-0.5, 1.5))
            direction = (90 + 30 * math.sin((d + h) / 10.0) + random.uniform(-45, 45)) % 360
            wind_table[d][h] = (speed, direction)
    return wind_table


def wind_effect_on_leg(drone_heading_deg: float, wind_speed: float, wind_dir_deg: float) -> float:
    """
    Calcula componente de vento ao longo do rumo do drone (m/s).
    - drone_heading_deg: rumo do drone em graus (0 = norte, 90 = leste)
    - wind_dir_deg: direção de onde o vento vem (comum em meteorologia) - assumimos vindo de wind_dir_deg
    Retorna vento_along: componente (m/s) positiva ajuda, negativa atrapalha.
    """
    # converter para radianos e calcular diferença de ângulo
    # Precisamos do vento na direção do movimento:
    # se vento vem de 'wind_dir_deg', seu vetor aponta em sentido oposto (wind vector direction = wind_dir_deg + 180)
    wind_vector_dir = math.radians((wind_dir_deg + 180) % 360)
    drone_dir = math.radians(drone_heading_deg)
    # componente do vento na direção do movimento (proj)
    comp = wind_speed * math.cos(wind_vector_dir - drone_dir)
    return comp  # m/s (positivo = aumenta velocidade efetiva)


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calcula o rumo/bearing de a->b em graus (0 = norte, 90 = leste).
    """
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    # converter para 0..360, 0 = norte
    bearing = (brng + 360) % 360
    return bearing


# -----------------------------
# 5) CALCULAR TEMPOS (considerando vento e velocidades)
# -----------------------------
# Parâmetros do drone e bateria
AUTONOMIA_BASE_SEC = 5000  # base
FACTOR = 0.93
AUTONOMIA_SEC = math.ceil(AUTONOMIA_BASE_SEC * FACTOR)  # aplicar fator e arredondar para cima
# velocidades: coerentes com ~10 m/s (~36 km/h). Usamos múltiplas velocidades (m/s).
# Observação: velocidades em km/h múltiplas de 4 eram exigidas; 36 km/h corresponde a 9.999.. m/s ~ 10 m/s
SPEEDS_KMH = [36, 40, 44]  # km/h (múltiplos próximos e consistentes)
SPEEDS_MS = [v * 1000.0 / 3600.0 for v in SPEEDS_KMH]  # converter para m/s
CRUISE_SPEED_MS = SPEEDS_MS[0]  # escolha principal (10.0 m/s ≈ 36 km/h)
# consumo simplificado: 1 "energia por segundo" base; headwind aumenta consumo.
ENERGY_PER_SEC_BASE = 1.0


def travel_time_and_energy(
    a: Tuple[float, float],
    b: Tuple[float, float],
    depart_datetime: datetime,
    wind_table: Dict[int, Dict[int, Tuple[float, float]]],
    speed_ms: float = CRUISE_SPEED_MS
) -> Tuple[float, float]:
    """
    Calcula tempo (segundos) e energia (unidade arbitrária) necessários para voar A->B
    levando em conta vento na hora correspondente do depart_datetime.
    - Usa um modelo simplificado: efeito do vento evaliado na hora de partida.
    - speed_ms: velocidade do drone em m/s (ar sem vento).
    Retorna: (tempo_sec, energy_units)
    """
    dist_m = haversine_distance(a, b)
    # heading
    heading = bearing_deg(a, b)
    # pegar vento para o dia/hora simulados
    # mapear dia 0.. para diferenças de data: assumimos start em day 0
    day_index = (depart_datetime.date() - START_DATE.date()).days
    hour_index = depart_datetime.hour
    # bounding day
    if day_index < 0:
        day_index = 0
    if day_index not in wind_table:
        day_index = day_index % len(wind_table)
    wind_speed, wind_dir = wind_table[day_index].get(hour_index, (0.0, 0.0))
    wind_along = wind_effect_on_leg(heading, wind_speed, wind_dir)
    # velocidade efetiva (m/s)
    v_eff = speed_ms + wind_along
    # evitar velocidades negativas ou muito baixas
    v_eff = max(1.0, v_eff)  # limite inferior 1 m/s para evitar divisão por zero
    time_sec = dist_m / v_eff
    # energia: tempo * base * (1 + penalidade_por_headwind)
    # penalidade proporcional à razão do componente de vento contra a velocidade (se negativo)
    penalty = 0.0
    if wind_along < 0:
        # headwind magnitude reduz a eficiência; fator empírico 0.5
        penalty = min(1.5, (-wind_along / max(0.1, speed_ms)) * 0.5)
    energy = time_sec * ENERGY_PER_SEC_BASE * (1 + penalty)
    return time_sec, energy


# -----------------------------
# 6) DEFINIR CROMOSSOMO
# -----------------------------
# Cromossomo: lista permutada de CEPs (exclui o CEP base que é fixo como start/end)
# Por exemplo: ['80010020', '80210010', ...]
# Nós manteremos representação simples: lista de strings (ceps)


# -----------------------------
# 7) IMPLEMENTAR FITNESS (simulate_route)
# -----------------------------
# Parâmetros operacionais
FLIGHT_WINDOW_START = time(6, 0, 0)   # 06:00
FLIGHT_WINDOW_END = time(19, 0, 0)    # 19:00
MAX_DAYS = 7
RECHARGE_COST = 80.0  # R$ por recarga
RECHARGE_AFTER_17_EXTRA = 80.0  # adicional se recarregar após 17:00
CHARGING_TIME_SEC = 60 * 60  # 1 hora para recarga completa (simples suposição)
PENALTY_INFEASIBLE = 1e8

# para conversões/saídas
CSV_FIELDS = [
    "CEP_inicial", "Lat_inicial", "Lon_inicial",
    "CEP_visitado", "Lat_visitado", "Lon_visitado",
    "Dia", "Hora_chegada", "Hora_partida",
    "Tempo_voo_seg", "Bateria_restante_seg", "Custo_acumulado_R$"
]

# START_DATE determina o dia 0 da simulação
START_DATE = datetime(2025, 10, 24, 6, 0, 0)  # Data de início (exemplo); dia 0 começa às 06:00


def time_in_window(dt: datetime) -> bool:
    """
    Verifica se datetime dt está dentro da janela de voo do dia (06:00-19:00).
    """
    t = dt.time()
    return (t >= FLIGHT_WINDOW_START) and (t <= FLIGHT_WINDOW_END)


def next_window_start(dt: datetime) -> datetime:
    """
    Retorna o próximo datetime válido dentro da janela de voo (pula para próximo dia às 06:00).
    Se dt já estiver dentro da janela, retorna dt.
    """
    if time_in_window(dt):
        return dt
    # se antes das 06:00 -> hoje às 06:00
    if dt.time() < FLIGHT_WINDOW_START:
        return dt.replace(hour=FLIGHT_WINDOW_START.hour, minute=0, second=0, microsecond=0)
    # se após 19:00 -> amanhã às 06:00
    next_day = (dt + timedelta(days=1)).replace(hour=FLIGHT_WINDOW_START.hour, minute=0, second=0, microsecond=0)
    return next_day


def simulate_route(
    chromosome: List[str],
    cep_coords: Dict[str, Tuple[float, float]],
    wind_table: Dict[int, Dict[int, Tuple[float, float]]],
    speed_ms: float = CRUISE_SPEED_MS,
) -> Tuple[float, List[Dict[str, Any]], bool]:
    """
    Simula a rota definida pelo cromossomo (permutação de CEPs).
    Retorna:
      - total_cost: custo agregado (tempo_voo_seconds + recargas em R$, mas ponderados)
      - route_log: lista de registros para cada perna/visita (para CSV)
      - feasible: booleano indicando se a solução é viável (<= MAX_DAYS, janela respeitada)
    Modelo de custo:
      - tempo de voo total em segundos (peso 1)
      - custo monetário por recarga somado (R$), vamos converter R$ para 'segundos custo' por fator
        para agregar em unidade única. Para simplicidade, normalizamos: 1 R$ = 60 segundos de "custo tempo"
        (isto é arbitrário: escolhemos 1 minuto por real para demonstrar trade-off)
    Nota didática: em problemas reais, usualmente normalizamos unidades ou multiobjetivamos.
    """
    # parametros locais
    cep_base = BASE_CEP
    current_loc = cep_coords[cep_base]
    current_time = START_DATE  # começa no dia 0 às 06:00
    day_limit = START_DATE + timedelta(days=MAX_DAYS)
    battery = AUTONOMIA_SEC  # segundos de autonomia restante
    total_flight_time = 0.0  # segundos
    total_recharge_cost = 0.0  # R$
    route_log = []
    feasible = True

    # visita cada CEP em ordem do cromossomo
    visit_list = chromosome[:]  # cópia
    # acrescentar retorno à base no final (será avaliado ao final)
    full_stops = visit_list + [cep_base]

    for idx, next_cep in enumerate(full_stops):
        dest = cep_coords[next_cep]
        # assegurar que estamos dentro da janela de voo (se não, avançar para próximo window start)
        if not time_in_window(current_time):
            new_start = next_window_start(current_time)
            # se pular dias, não voa no tempo parado (simula espera)
            current_time = new_start
        # tempo de voo e energia para esta perna
        time_sec, energy_needed = travel_time_and_energy(current_loc, dest, current_time, wind_table, speed_ms)
        # verificar se a perna é possível com a bateria atual
        if energy_needed <= battery:
            # voa direto
            arrival_time = current_time + timedelta(seconds=time_sec)
            # se chegada estourar janela de voo, devemos pausar no meio? Modelo simplificado:
            # se arrival_time after window end, então essa perna não é permitida sem recarga/wait
            if arrival_time.time() > FLIGHT_WINDOW_END:
                # opção: esperar até next day 06:00 and then perform leg (no ganho de battery)
                # para modelo simples, forçamos esperar e then recompute with possibly different wind table
                current_time = next_window_start(current_time + timedelta(days=1))  # ir para dia seguinte 06:00
                # recomputar tempo e energia usando novo current_time
                time_sec, energy_needed = travel_time_and_energy(current_loc, dest, current_time, wind_table, speed_ms)
                arrival_time = current_time + timedelta(seconds=time_sec)
                # se ainda inviável por excesso de dias, penalizar
            # consumir bateria
            battery -= energy_needed
            total_flight_time += time_sec
            # registrar
            record = {
                "from_cep": None if idx == 0 and current_loc == cep_coords[cep_base] else None,
                "cep": next_cep,
                "lat": dest[0],
                "lon": dest[1],
                "day": (arrival_time.date() - START_DATE.date()).days,
                "arrival_time": arrival_time,
                "departure_time": arrival_time,  # assumes instant service (visita curta)
                "time_sec": math.ceil(time_sec),
                "battery_remain": math.ceil(battery),
                "cost_so_far": total_recharge_cost
            }
            route_log.append(record)
            # advance current_time and location
            current_time = arrival_time
            current_loc = dest
        else:
            # bateria insuficiente -> recarrega antes de partir (no ponto atual)
            # se ainda não houver infraestrutura, neste modelo simplificamos: pode recarregar em qualquer ponto
            # custo de recarga
            # cobrança base R$80, se recarga ocorrer após 17:00 acrescenta +80
            recharge_cost = RECHARGE_COST
            if current_time.time() >= time(17, 0, 0):
                recharge_cost += RECHARGE_AFTER_17_EXTRA
            # adiciona custo e tempo para recarga
            total_recharge_cost += recharge_cost
            # arredondamento para cima: se houver frações de recarga (modelo simplificado) usamos ceil
            total_recharge_cost = math.ceil(total_recharge_cost)
            # tempo de recarga
            current_time = current_time + timedelta(seconds=CHARGING_TIME_SEC)
            # restaurar bateria
            battery = AUTONOMIA_SEC
            # se recarga após janela de voo (19:00), pular para próximo dia 06:00
            if not time_in_window(current_time):
                current_time = next_window_start(current_time)
            # re-tentar mesma perna (recursivamente loop)
            # To avoid infinite loops, check day limit
            if current_time > day_limit:
                feasible = False
                break
            # after recharging, recompute travel
            time_sec, energy_needed = travel_time_and_energy(current_loc, dest, current_time, wind_table, speed_ms)
            # If still cannot fly even with full battery (very long leg) -> infeasible
            if energy_needed > AUTONOMIA_SEC * 1.05:  # margem 5%
                feasible = False
                break
            # else perform flight
            arrival_time = current_time + timedelta(seconds=time_sec)
            battery -= energy_needed
            total_flight_time += time_sec
            record = {
                "from_cep": None,
                "cep": next_cep,
                "lat": dest[0],
                "lon": dest[1],
                "day": (arrival_time.date() - START_DATE.date()).days,
                "arrival_time": arrival_time,
                "departure_time": arrival_time,
                "time_sec": math.ceil(time_sec),
                "battery_remain": math.ceil(battery),
                "cost_so_far": total_recharge_cost
            }
            route_log.append(record)
            current_time = arrival_time
            current_loc = dest

        # verificar se passado do limite de dias
        if current_time > day_limit:
            feasible = False
            break

    # custo agregado: transformamos R$ em segundos (1 R$ = 60 s) e somamos ao tempo de voo
    # (escolha arbitrária para converter unidades; serve para demonstrar otimização conjunta)
    cost_from_recharge_in_seconds = total_recharge_cost * 60.0
    total_cost = total_flight_time + cost_from_recharge_in_seconds

    # penalizar soluções que excedam dias ou que sejam inviáveis
    if not feasible:
        total_cost += PENALTY_INFEASIBLE

    return total_cost, route_log, feasible


def funcao_fitness(chromosome: List[str], *args, **kwargs) -> float:
    """
    Adaptação para o GA: menor custo = melhor. Como os algoritmos genéticos clássicos
    costumam maximizar, retornamos -cost (ou retornamos cost e faremos seleção por menor).
    Aqui retornaremos o custo (menor é melhor) e o GA fará seleção baseada nisso.
    """
    cep_coords, wind_table = args
    cost, _, feasible = simulate_route(chromosome, cep_coords, wind_table)
    return cost


# -----------------------------
# 8) OPERADORES GENÉTICOS
# -----------------------------
def initial_population(ceps: List[str], pop_size: int) -> List[List[str]]:
    """
    Gera população inicial com permutações aleatórias dos CEPs (exclui base).
    """
    population = []
    for _ in range(pop_size):
        ind = ceps[:]
        random.shuffle(ind)
        population.append(ind)
    return population


def tournament_selection(pop: List[List[str]], fitnesses: List[float], k: int = 3) -> List[str]:
    """
    Seleção por torneio (retorna um indivíduo).
    """
    selected = random.sample(list(zip(pop, fitnesses)), k)
    selected.sort(key=lambda x: x[1])  # menor custo é melhor
    return selected[0][0][:]  # retornar cópia do indivíduo vencedor


def ordered_crossover(parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
    """
    Crossover OX (Order Crossover) para permutações.
    """
    size = len(parent1)
    if size < 2:
        return parent1[:], parent2[:]
    a, b = sorted(random.sample(range(size), 2))
    # helper to produce child
    def ox(p1, p2):
        child = [None] * size
        # copy slice from p1
        child[a:b+1] = p1[a:b+1]
        # fill remaining positions with p2 order
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[p2_idx] in child:
                    p2_idx += 1
                child[i] = p2[p2_idx]
        return child
    return ox(parent1, parent2), ox(parent2, parent1)


def swap_mutation(ind: List[str], mutation_rate: float) -> List[str]:
    """
    Mutação por troca: com probabilidade mutation_rate, troque dois genes.
    """
    ind = ind[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind


# -----------------------------
# 9) EXECUTAR O ALGORITMO GENÉTICO
# -----------------------------
def run_ga(
    cep_coords: Dict[str, Tuple[float, float]],
    base_cep: str,
    wind_table: Dict[int, Dict[int, Tuple[float, float]]],
    pop_size: int = 60,
    generations: int = 300,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism: int = 2
) -> Tuple[List[str], float, List[Dict[str, Any]]]:
    """
    Executa GA customizado:
      - população de permutações
      - seleção por torneio
      - crossover OX
      - swap mutation
      - retorna o melhor indivíduo, seu custo e log da rota
    """
    # individual genes: todos os CEPs menos a base
    genes = [c for c in cep_coords.keys() if c != base_cep]
    population = initial_population(genes, pop_size)
    best_solution = None
    best_cost = float('inf')
    best_log = []

    # avaliar população inicial
    fitnesses = [funcao_fitness(ind, cep_coords, wind_table) for ind in population]

    for gen in range(generations):
        new_pop = []
        # elitismo: copiar os melhores diretamente
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1])
        elites = [ind for ind, fit in ranked[:elitism]]
        for e in elites:
            new_pop.append(e[:])

        # gerar restante da nova população
        while len(new_pop) < pop_size:
            # seleção
            parent1 = tournament_selection(population, fitnesses, k=3)
            parent2 = tournament_selection(population, fitnesses, k=3)
            # crossover
            if random.random() < crossover_rate:
                child1, child2 = ordered_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            # mutação
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        population = new_pop
        fitnesses = [funcao_fitness(ind, cep_coords, wind_table) for ind in population]

        # atualizar melhor
        min_idx = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
        if fitnesses[min_idx] < best_cost:
            best_cost = fitnesses[min_idx]
            best_solution = population[min_idx][:]
            # recalcular log para o melhor
            _, best_log, _ = simulate_route(best_solution, cep_coords, wind_table)

        # Progresso simples (poderia ser comentado em runs longos)
        if gen % max(1, generations // 10) == 0:
            print(f"[GA] generation {gen}/{generations} best_cost = {best_cost:.2f}")

    return best_solution, best_cost, best_log


# -----------------------------
# 10) GERAR CSV (saída)
# -----------------------------
def save_route_csv(filename: str, base_cep: str, cep_coords: Dict[str, Tuple[float, float]], route_log: List[Dict[str, Any]]):
    """
    Gera CSV com colunas especificadas: CEP inicial, Lat inicial, Lon inicial, CEP visitado, Lat, Lon,
    Dia, Hora_chegada, Hora_partida, Tempo_voo_seg, Bateria_restante_seg, Custo_acumulado_R$
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        # escrever linha por visita
        for rec in route_log:
            cep_v = rec["cep"]
            lat_v = rec["lat"]
            lon_v = rec["lon"]
            arr = rec["arrival_time"]
            dep = rec["departure_time"]
            writer.writerow([
                base_cep,
                cep_coords[base_cep][0],
                cep_coords[base_cep][1],
                cep_v,
                lat_v,
                lon_v,
                rec["day"],
                arr.strftime("%Y-%m-%d %H:%M:%S"),
                dep.strftime("%Y-%m-%d %H:%M:%S"),
                rec["time_sec"],
                rec["battery_remain"],
                rec["cost_so_far"]
            ])
    print(f"[IO] CSV salvo em: {filename}")


# -----------------------------
# MAIN: execução e organização
# -----------------------------
def main():
    """
    Função principal que compõe tudo:
      - carrega CEPs (simulados)
      - gera tabela de vento (simulada)
      - executa GA
      - salva CSV com rota ótima encontrada
    """
    print("[INFO] Iniciando rotina de otimização do roteiro do drone (GA)")

    # 2) Coletar dados
    cep_coords, base_cep = load_ceps()
    global BASE_CEP
    BASE_CEP = base_cep

    # Imprimir resumo dos nós
    print(f"[DATA] CEP base: {base_cep}, total pontos (incluindo base): {len(cep_coords)}")
    print("[DATA] Lista de CEPs (excluindo base):")
    for c in cep_coords:
        if c != base_cep:
            print("  -", c, cep_coords[c])

    # 4) Modelar vento (simulado)
    wind_table = generate_wind_table(num_days=MAX_DAYS)
    print("[DATA] Tabela de vento simulada para", MAX_DAYS, "dias gerada.")

    # Parâmetros GA (ajustáveis)
    POP_SIZE = 80
    GENERATIONS = 200
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.25
    ELITISM = 3

    # 9) Executar algoritmo genético
    try:
        best_solution, best_cost, best_log = run_ga(
            cep_coords,
            base_cep,
            wind_table,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            elitism=ELITISM
        )
    except Exception as e:
        print("[ERROR] Erro durante execução do GA:", e)
        sys.exit(1)

    if best_solution is None:
        print("[RESULT] Nenhuma solução encontrada.")
        sys.exit(1)

    # Apresentar resultado resumido
    print("\n[RESULT] Melhor rota encontrada (sequência de CEPs):")
    print("Início no base ->", " -> ".join(best_solution), "->", base_cep)
    print(f"[RESULT] Custo total (tempo equiv. s + custo recargas convertido): {best_cost:.2f}")

    # 10) Gerar CSV
    output_csv = "rota_drone_saida.csv"
    save_route_csv(output_csv, base_cep, cep_coords, best_log)

    # Exibir sumário do log (primeiras e últimas linhas)
    print("\n[LOG] Exemplo de registros (até 10):")
    for rec in best_log[:10]:
        print(f"  CEP {rec['cep']} | chegada {rec['arrival_time']} | bateria {rec['battery_remain']}s | tempo_voo {rec['time_sec']}s | custo_R$ {rec['cost_so_far']}")

    print("\n[INFO] Execução finalizada. Verifique o CSV gerado para detalhes.")


# Apenas executa main quando rodado diretamente
if __name__ == "__main__":
    main()
