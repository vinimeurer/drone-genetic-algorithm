
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic algorithm for routing a drone to visit a list of CEPs in Curitiba.
Implements specification provided by the user exactly:
 - reads 'coordenadas.csv' and 'vento.csv' from working directory
 - Haversine distances
 - wind model affecting effective speed
 - autonomy model with correction factor and speed dependency
 - representation: permutation (path) + day/hours/velocities per leg
 - GA with PMX crossover for path, single-point for schedules/speeds
 - outputs CSV with optimized route (one row per flight leg)
 
Usage:
    Place 'coordenadas.csv' and 'vento.csv' in the same folder and run:
        python3 drone_ga_curitiba.py

Notes:
 - The code expects 'coordenadas.csv' with columns: cep,longitude,latitude
 - The code expects 'vento.csv' with columns: dia,hora,vel_kmh,direcao_deg
 - Start/End CEP is assumed to be the first CEP in coordenadas.csv (Unibrasil) with id 1.
 - The script writes 'rota_otimizada.csv' as final output.
 
This file implements classes and functions per the user's description.
"""

from __future__ import annotations
import math
import random
import csv
import copy
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time

# -----------------------
# Constants / Parameters
# -----------------------

R_EARTH_KM = 6371.0
BASE_AUTONOMY_SECONDS = 5000  # 1h 23m 20s = 5000s
CORRECTION_FACTOR = 0.93
BASE_AUTONOMY_CORRECTED = int(round(BASE_AUTONOMY_SECONDS * CORRECTION_FACTOR))  # 4650
MIN_SPEED = 36  # km/h
MAX_SPEED = 96  # km/h
SPEED_STEP = 4  # km/h increments
VALID_SPEEDS = list(range(MIN_SPEED, MAX_SPEED + 1, SPEED_STEP))  # [36,40,...,96]
STOP_SECONDS = 72  # seconds per stop (photo or recharge)
FLIGHT_WINDOW_START_HOUR = 6  # 06:00
FLIGHT_WINDOW_END_HOUR = 19  # 19:00
MAX_DAYS = 7  # days allowed
RECHARGE_COST = 80  # R$ per pouso (landing to recharge)
RECHARGE_LATE_SURCHARGE = 80  # +R$ if landing after 17:00
PENALTY_HIGH = 10_000_000  # heavy penalty for infeasible solutions
# Conversion factor to combine monetary cost with time (arbitrary but consistent)
# 1 R$ = 3600 seconds worth of cost. This choice produces a combined scalar cost.
MONEY_TO_SECONDS = 3600.0

# GA parameters (default)
POPULATION_SIZE = 80  # between 50-100 recommended
GENERATIONS = 300  # between 100-500 recommended
TOURNAMENT_K = 3
MUTATION_RATE = 0.05
ELITISM = True
ELITISM_COUNT = 2
CROSSOVER_RATE = 0.9

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------
# Data classes
# -----------------------


@dataclass
class CEPPoint:
    id: int
    cep: str
    longitude: float
    latitude: float


@dataclass
class WindRecord:
    dia: int
    hora: int
    vel_kmh: float
    direcao_deg: float  # direction in degrees (assumed as vector pointing to this direction)


# -----------------------
# Utilities
# -----------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two (lat, lon) points in km."""
    # convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R_EARTH_KM * c
    return d


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing (azimuth) in degrees from point1 to point2.
    Result in degrees [0,360), where 0 is North, 90 is East.
    We'll return degrees with 0 = north, increasing clockwise.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lambda1 = math.radians(lon1)
    lambda2 = math.radians(lon2)
    y = math.sin(lambda2 - lambda1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lambda2 - lambda1)
    theta = math.atan2(y, x)
    bearing = (math.degrees(theta) + 360) % 360
    # Convert from mathematical bearing (0 = east) to navigational (0 = north)
    # The above already yields 0 = north due to formula; keep as-is.
    return bearing


def vector_from_speed_direction_kmh(speed_kmh: float, direction_deg: float) -> Tuple[float, float]:
    """
    Convert a speed and direction into vector components (vx, vy) in km/h.
    direction_deg: degrees where 0 is North, increasing clockwise.
    We'll convert to standard math coordinates where:
      - x axis is East, y axis is North.
    direction_deg 0 -> (0, +1)
    So:
      vx = speed * sin(dir_rad)
      vy = speed * cos(dir_rad)
    """
    rad = math.radians(direction_deg)
    vx = speed_kmh * math.sin(rad)
    vy = speed_kmh * math.cos(rad)
    return vx, vy


def time_to_hhmmss(seconds: int) -> str:
    return str(timedelta(seconds=int(seconds)))


def float_hour_to_hhmm(hour_float: float) -> str:
    # hour_float may be like 6.5 for 06:30
    total_seconds = int(round(hour_float * 3600))
    hh = (total_seconds // 3600) % 24
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


# -----------------------
# Data loaders
# -----------------------


class DataLoader:
    def __init__(self, coords_path: str = "coordenadas.csv", wind_path: str = "vento.csv"):
        self.coords_path = coords_path
        self.wind_path = wind_path
        self.ceps: List[CEPPoint] = []
        # wind[dia][hora] = {'velocidade_kmh': ..., 'direcao_graus': ...}
        self.wind: Dict[int, Dict[int, Dict[str, float]]] = {}

    def load_coords(self):
        df = pd.read_csv(self.coords_path, dtype=str)
        # expect columns cep,longitude,latitude
        required = {"cep", "longitude", "latitude"}
        if not required.issubset(set(df.columns)):
            raise ValueError("coordenadas.csv must contain columns: cep,longitude,latitude")
        self.ceps = []
        for idx, row in df.iterrows():
            cep = str(row["cep"]).strip()
            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
            except Exception as e:
                raise ValueError(f"Invalid coordinates at row {idx}: {e}")
            self.ceps.append(CEPPoint(id=idx + 1, cep=cep, longitude=lon, latitude=lat))
        if len(self.ceps) < 2:
            raise ValueError("Need at least two CEPs (start and some destinations).")
        # Start/End is considered the first CEP (id=1) per specification.
        return self.ceps

    def load_wind(self):
        df = pd.read_csv(self.wind_path)
        required = {"dia", "hora", "vel_kmh", "direcao_deg"}
        if not required.issubset(set(df.columns)):
            raise ValueError("vento.csv must contain columns: dia,hora,vel_kmh,direcao_deg")
        # Build dictionary
        wind: Dict[int, Dict[int, Dict[str, float]]] = {}
        for _, row in df.iterrows():
            dia = int(row["dia"])
            hora = int(row["hora"])
            vel = float(row["vel_kmh"])
            direcao = float(row["direcao_deg"])
            wind.setdefault(dia, {})[hora] = {"velocidade_kmh": vel, "direcao_graus": direcao}
        # For hours missing in day, we could interpolate or raise; specification: extract 7 days a hora table - we assume provided.
        self.wind = wind
        return self.wind


# -----------------------
# Distance matrix
# -----------------------


class DistanceCalculator:
    def __init__(self, ceps: List[CEPPoint]):
        self.ceps = ceps
        self.n = len(ceps)
        # distance matrix indexed by id (1-based) -> id
        self.dist_matrix_km: Dict[Tuple[int, int], float] = {}
        self.bearing_matrix_deg: Dict[Tuple[int, int], float] = {}
        self._compute_all_pairs()

    def _compute_all_pairs(self):
        for a in self.ceps:
            for b in self.ceps:
                d = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
                bear = bearing_deg(a.latitude, a.longitude, b.latitude, b.longitude)
                self.dist_matrix_km[(a.id, b.id)] = d
                self.bearing_matrix_deg[(a.id, b.id)] = bear

    def distance(self, id1: int, id2: int) -> float:
        return self.dist_matrix_km[(id1, id2)]

    def bearing(self, id1: int, id2: int) -> float:
        return self.bearing_matrix_deg[(id1, id2)]


# -----------------------
# Wind model
# -----------------------


class WindModel:
    def __init__(self, wind_table: Dict[int, Dict[int, Dict[str, float]]]):
        # assumed wind_table[dia][hora] exists for days 1..7 and hours 6..18 as provided
        self.wind = wind_table

    def get_wind_vector(self, dia: int, hora: int) -> Tuple[float, float]:
        """
        Returns wind vector components (vx, vy) in km/h for the given day and hour.
        If exact hour not present, attempt to use nearest available hour within same day; otherwise assume zero.
        """
        if dia not in self.wind:
            return 0.0, 0.0
        day_wind = self.wind[dia]
        if hora in day_wind:
            rec = day_wind[hora]
            return vector_from_speed_direction_kmh(rec["velocidade_kmh"], rec["direcao_graus"])
        # find nearest hour key
        hours = sorted(day_wind.keys())
        if not hours:
            return 0.0, 0.0
        nearest = min(hours, key=lambda h: abs(h - hora))
        rec = day_wind[nearest]
        return vector_from_speed_direction_kmh(rec["velocidade_kmh"], rec["direcao_graus"])


# -----------------------
# Drone autonomy and consumption
# -----------------------


class DroneAutonomy:
    """
    Models battery autonomy and consumption based on speed choices and wind effects.
    """

    def __init__(self, base_autonomy_seconds: int = BASE_AUTONOMY_SECONDS, correction: float = CORRECTION_FACTOR):
        self.base = base_autonomy_seconds
        self.correction = correction
        self.base_corrected = int(round(self.base * self.correction))

    def autonomy_seconds_for_speed(self, speed_kmh: float) -> int:
        """
        A(v) = 5000 * 0.93 * (36 / v)^2
        Returns autonomy in seconds for given speed.
        If speed < 36, we'll use autonomy at 36 as minimum per spec (36 is min).
        """
        v = max(speed_kmh, MIN_SPEED)
        A = self.base * self.correction * (36.0 / float(v)) ** 2
        return int(round(A))

    # Note: consumption per segment is represented simply by the time t in seconds.
    # The "battery" is decremented by t seconds. If remaining battery < t, recharge is needed beforehand.


# -----------------------
# Flight simulation for a single route (chromosome)
# -----------------------


@dataclass
class LegResult:
    cep_from: str
    lat_from: float
    lon_from: float
    dia: int
    hora_inicio_float: float  # fractional hour (e.g., 6.5 for 06:30)
    velocidade_kmh: int
    cep_to: str
    lat_to: float
    lon_to: float
    pouso: str  # 'SIM' or 'NAO' indicating recharge landed
    hora_final_float: float  # fractional hour when arrival+stop ends
    distance_km: float
    time_flight_s: int
    time_total_s: int  # includes stops
    RECHARGE_COST: float


class FlightSimulator:
    """
    Simulates a route given a chromosome (order + day/hour/vel per leg) and computes total cost.
    """

    def __init__(self,
                 ceps: List[CEPPoint],
                 dist_calc: DistanceCalculator,
                 wind_model: WindModel,
                 autonomy_model: DroneAutonomy):
        self.ceps = {c.id: c for c in ceps}
        self.dist_calc = dist_calc
        self.wind_model = wind_model
        self.autonomy_model = autonomy_model
        self.per_stop_seconds = STOP_SECONDS

    def _speed_effective_kmh(self, cruise_speed_kmh: float, azimuth_deg: float, dia: int, hora_int: int) -> float:
        """
        Compute effective ground speed magnitude (km/h) from drone cruise speed and wind vector at given dia/hora.
        - drone vector is cruise_speed_kmh towards azimuth_deg
        - wind vector is obtained from wind model (direction degrees as given)
        - effective speed is magnitude of vector sum (vx, vy)
        """
        v_drone_x, v_drone_y = vector_from_speed_direction_kmh(cruise_speed_kmh, azimuth_deg)
        v_wind_x, v_wind_y = self.wind_model.get_wind_vector(dia, hora_int)
        v_sum_x = v_drone_x + v_wind_x
        v_sum_y = v_drone_y + v_wind_y
        v_eff = math.hypot(v_sum_x, v_sum_y)
        # Ensure non-zero to avoid division issues
        v_eff = max(0.0001, v_eff)
        return v_eff

    def simulate_route(self,
                       order_ids: List[int],
                       dias: List[int],
                       horas_float: List[float],
                       velocidades: List[int],
                       start_time_info: Optional[Tuple[int, float]] = None
                       ) -> Tuple[float, List[LegResult], bool]:
        """
        Simulates the entire route.
        - order_ids: list of CEP ids representing visitation order excluding the fixed start at id=1 (Unibrasil).
          We will create legs that start at Unibrasil (id 1), go through order_ids, and return to Unibrasil.
        - dias/hours/velocities: vectors with length equal to number of legs (n_legs).
            n_legs = len(order_ids) + 1 (return leg to start)
        - start_time_info is unused; dni/horas define schedule per leg explicitly as required by specification.
        Returns:
            total_cost_scalar: float (seconds + money converted to seconds + penalties if any)
            leg_results: list of LegResult dataclasses for CSV output
            feasible: bool whether the solution violated unrepairable constraints
        """
        n_visits = len(order_ids)
        # Build full route of ids: start (1) -> order_ids -> start (1)
        start_id = 1
        route = [start_id] + order_ids + [start_id]
        n_legs = len(route) - 1
        if not (len(dias) == n_legs == len(horas_float) == len(velocidades)):
            raise ValueError("Length of dias/horas/velocidades must equal number of legs (visits+return).")
        # Initialize battery for first leg based on first leg speed
        current_battery_seconds = self.autonomy_model.autonomy_seconds_for_speed(velocidades[0])
        # Simulate leg by leg
        total_flight_seconds = 0
        total_RECHARGE_COST = 0.0
        leg_results: List[LegResult] = []
        feasible = True
        penalty = 0.0

        for i in range(n_legs):
            from_id = route[i]
            to_id = route[i + 1]
            dia = int(dias[i])
            hora_float = float(horas_float[i])
            velocidade = int(velocidades[i])
            # Validate day/time constraints
            if dia < 1 or dia > MAX_DAYS:
                feasible = False
                penalty += PENALTY_HIGH
            # compute integer hour to lookup wind (use floor of hour)
            hora_int = max(0, min(23, int(math.floor(hora_float))))
            # Validate flight allowed hours: departure must be within 06:00-19:00 window
            if not (FLIGHT_WINDOW_START_HOUR <= hora_int <= FLIGHT_WINDOW_END_HOUR):
                feasible = False
                penalty += PENALTY_HIGH
            # compute distance and azimuth
            dist_km = self.dist_calc.distance(from_id, to_id)
            azimuth_deg = self.dist_calc.bearing(from_id, to_id)
            # effective ground speed (km/h) considering wind at that dia/hora
            v_eff_kmh = self._speed_effective_kmh(velocidade, azimuth_deg, dia, hora_int)
            # time to fly in seconds: ceil(dist / (v_eff/3600))
            # v_eff_kmh -> km/h, convert to km/s by /3600
            if v_eff_kmh <= 0:
                # cannot progress
                feasible = False
                penalty += PENALTY_HIGH
                time_flight_s = 10**9
            else:
                time_flight_s = int(math.ceil(dist_km / (v_eff_kmh / 3600.0)))
            # If battery insufficient, force recharge at 'from' CEP before departing
            pouso_realizado = False
            recharge_cost_for_leg = 0.0
            if time_flight_s > current_battery_seconds:
                # perform recharge at from location BEFORE departing
                pouso_realizado = True
                # landing cost
                recharge_cost_for_leg += RECHARGE_COST
                # additional surcharge if landing occurs after 17:00 local time
                # Determine local clock hour of landing: it's the hora_float provided (before departure)
                if hora_float >= 17.0:
                    recharge_cost_for_leg += RECHARGE_LATE_SURCHARGE
                # recharging sets battery to autonomy according to speed of next leg (we assume recharge gives full battery)
                current_battery_seconds = self.autonomy_model.autonomy_seconds_for_speed(velocidade)
                # recharging consumes stop_seconds for recharge
                recharge_stop_s = self.per_stop_seconds
            else:
                recharge_stop_s = 0
            # After recharge or not, attempt flight
            # Additional stop for photographing at arrival (72s) — per spec stops include photo or recharge
            arrival_photo_stop_s = self.per_stop_seconds
            # total time including stops (if recharge happened before flight, add recharge_stop_s)
            total_leg_time_s = int(time_flight_s + recharge_stop_s + arrival_photo_stop_s)
            # Deduct battery by flight time only (stopping/hover/charging doesn't reduce flight battery; recharging resets)
            # Here we interpret battery as flight-capacity seconds: after flight, subtract time_flight_s
            current_battery_seconds -= min(current_battery_seconds, time_flight_s)
            # If battery becomes negative, infeasible (battery zero during flight)
            if current_battery_seconds < 0:
                feasible = False
                penalty += PENALTY_HIGH
            # After arrival, we do NOT automatically recharge unless next leg requires it; but we already added arrival_photo_stop_s
            # The recharge cost (if any) counted for this leg
            total_flight_seconds += time_flight_s
            total_RECHARGE_COST += recharge_cost_for_leg
            # Build hora_final: hora_float is departure hour fractional; add total_leg_time_s in hours
            hora_final_float = hora_float + (total_leg_time_s / 3600.0)
            # Create leg result (note: pouso indicates recharge landing)
            leg_results.append(
                LegResult(
                    cep_from=self.ceps[from_id].cep,
                    lat_from=self.ceps[from_id].latitude,
                    lon_from=self.ceps[from_id].longitude,
                    dia=dia,
                    hora_inicio_float=hora_float,
                    velocidade_kmh=velocidade,
                    cep_to=self.ceps[to_id].cep,
                    lat_to=self.ceps[to_id].latitude,
                    lon_to=self.ceps[to_id].longitude,
                    pouso="SIM" if pouso_realizado else "NAO",
                    hora_final_float=hora_final_float,
                    distance_km=dist_km,
                    time_flight_s=time_flight_s,
                    time_total_s=total_leg_time_s,
                    RECHARGE_COST=recharge_cost_for_leg
                )
            )
            # If we performed a recharge, after arrival battery remains at whatever left (recharge was before flight),
            # but if next leg chosen speed is different, autonomy recalculated when needed.
            # For safety, if battery is zero or near zero, do not allow next leg without recharge (will be forced next loop).
            if current_battery_seconds <= 0:
                # Set to zero and require recharge before next departure
                current_battery_seconds = 0

        # Build cost scalar: total flight time seconds + money converted to seconds + penalties
        total_cost_scalar = float(total_flight_seconds) + float(total_RECHARGE_COST) * MONEY_TO_SECONDS + penalty
        return total_cost_scalar, leg_results, feasible


# -----------------------
# Chromosome representation
# -----------------------


@dataclass
class Chromosome:
    order_ids: List[int]  # permutation of destination ids excluding start (1)
    dias: List[int]  # one per leg (len = n_visits + 1)
    horas_float: List[float]  # fractional hours per leg
    velocidades: List[int]  # speed per leg in km/h

    def copy(self) -> "Chromosome":
        return Chromosome(
            order_ids=self.order_ids.copy(),
            dias=self.dias.copy(),
            horas_float=self.horas_float.copy(),
            velocidades=self.velocidades.copy()
        )


# -----------------------
# Genetic operators
# -----------------------


def pmx_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Partially Mapped Crossover (PMX) for permutations.
    Returns two offspring permutations.
    """
    size = len(parent1)
    if size <= 2:
        return parent1.copy(), parent2.copy()
    # choose crossover points
    cx1 = random.randint(0, size - 2)
    cx2 = random.randint(cx1 + 1, size - 1)
    def pmx(p1, p2):
        child = [-1] * size
        # copy slice from p1
        child[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
        # map the rest from p2
        for i in range(cx1, cx2 + 1):
            # value in p2 that needs to be placed if not already present
            val = p2[i]
            if val not in child:
                pos = i
                mapped = p1[pos]
                # find position to place mapped until free
                while True:
                    pos = p2.index(mapped)
                    if child[pos] == -1:
                        child[pos] = val
                        break
                    mapped = p1[pos]
        # fill remaining positions from p2
        for i in range(size):
            if child[i] == -1:
                child[i] = p2[i]
        return child
    o1 = pmx(parent1, parent2)
    o2 = pmx(parent2, parent1)
    return o1, o2


def single_point_crossover_vec(vec1: List, vec2: List) -> Tuple[List, List]:
    """Single point crossover for lists (dias/horas/velocidades)."""
    size = len(vec1)
    if size <= 1:
        return vec1.copy(), vec2.copy()
    pt = random.randint(1, size - 1)
    child1 = vec1[:pt] + vec2[pt:]
    child2 = vec2[:pt] + vec1[pt:]
    return child1, child2


def mutate_permutation_swap(perm: List[int], mutation_rate: float) -> None:
    """Mutation by swapping two elements with probability mutation_rate per chromosome."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]


def mutate_inversion(perm: List[int], mutation_rate: float) -> None:
    """Mutation by inverting a subsequence."""
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(perm)), 2))
        perm[i:j + 1] = list(reversed(perm[i:j + 1]))


def mutate_displacement(perm: List[int], mutation_rate: float) -> None:
    """Mutation by removing a subsequence and inserting it at another position."""
    if random.random() < mutation_rate:
        size = len(perm)
        i, j = sorted(random.sample(range(size), 2))
        block = perm[i:j + 1]
        rest = perm[:i] + perm[j + 1:]
        k = random.randint(0, len(rest))
        perm[:] = rest[:k] + block + rest[k:]


def mutate_numeric_vector(vec: List, min_vals, max_vals, mutation_rate: float) -> None:
    """
    Mutate a numeric vector:
    - For dias: integer between 1 and 7
    - For horas_float: mutate by small random delta within [6,19]
    - For velocidades: choose valid speed from VALID_SPEEDS
    """
    size = len(vec)
    for i in range(size):
        if random.random() < mutation_rate:
            if isinstance(vec[i], int):
                # dias or velocidades (both int): if in VALID_SPEEDS choose new random speed, else if dia choose 1..7
                if vec[i] in VALID_SPEEDS:
                    vec[i] = random.choice(VALID_SPEEDS)
                else:
                    vec[i] = random.randint(1, MAX_DAYS)
            else:
                # float hour
                # add small perturbation up to +/- 1 hour
                delta = random.uniform(-1.0, 1.0)
                newh = vec[i] + delta
                # clamp to allowed hours range
                newh = max(float(FLIGHT_WINDOW_START_HOUR), min(float(FLIGHT_WINDOW_END_HOUR), newh))
                vec[i] = newh


# -----------------------
# Fitness and GA main
# -----------------------


class GeneticAlgorithm:
    def __init__(self,
                 ceps: List[CEPPoint],
                 dist_calc: DistanceCalculator,
                 wind_model: WindModel,
                 autonomy_model: DroneAutonomy,
                 population_size: int = POPULATION_SIZE,
                 generations: int = GENERATIONS,
                 mutation_rate: float = MUTATION_RATE,
                 tournament_k: int = TOURNAMENT_K,
                 elitism: bool = ELITISM,
                 elitism_count: int = ELITISM_COUNT):
        self.ceps = ceps
        self.dist_calc = dist_calc
        self.wind_model = wind_model
        self.autonomy_model = autonomy_model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.elitism_count = elitism_count
        # Precompute helper
        self.n_visits = len(ceps) - 1  # excluding start which is id=1
        self.all_destination_ids = [c.id for c in ceps if c.id != 1]

        # flight simulator
        self.simulator = FlightSimulator(ceps, dist_calc, wind_model, autonomy_model)

    def initial_population(self) -> List[Chromosome]:
        pop: List[Chromosome] = []
        for _ in range(self.population_size):
            perm = self.all_destination_ids.copy()
            random.shuffle(perm)
            n_legs = len(perm) + 1
            # Random dias per leg [1..7]
            dias = [random.randint(1, MAX_DAYS) for _ in range(n_legs)]
            # Random horas per leg float between 6.0 and 19.0
            horas = [random.uniform(float(FLIGHT_WINDOW_START_HOUR), float(FLIGHT_WINDOW_END_HOUR)) for _ in range(n_legs)]
            # Random velocidades per leg from VALID_SPEEDS
            velocidades = [random.choice(VALID_SPEEDS) for _ in range(n_legs)]
            pop.append(Chromosome(order_ids=perm, dias=dias, horas_float=horas, velocidades=velocidades))
        return pop

    def fitness_of(self, chromo: Chromosome) -> Tuple[float, float, bool]:
        """
        Returns (fitness_value, raw_cost, feasible)
        fitness = 1 / (1 + CustoTotal)
        """
        raw_cost, leg_results, feasible = self.simulator.simulate_route(
            chromo.order_ids, chromo.dias, chromo.horas_float, chromo.velocidades
        )
        # If infeasible, add big penalty to cost
        if not feasible:
            raw_cost += PENALTY_HIGH
        fitness = 1.0 / (1.0 + raw_cost)
        return fitness, raw_cost, feasible

    def tournament_selection(self, population: List[Chromosome], fitnesses: List[float]) -> Chromosome:
        selected = random.sample(range(len(population)), k=min(self.tournament_k, len(population)))
        best_idx = max(selected, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def evolve(self) -> Tuple[Chromosome, float, List[LegResult]]:
        # initialize population
        population = self.initial_population()
        best_solution: Optional[Chromosome] = None
        best_fitness = -1.0
        best_raw_cost = float("inf")
        best_leg_results: List[LegResult] = []

        for gen in range(self.generations):
            # evaluate fitnesses
            fitnesses = []
            raw_costs = []
            feasibilities = []
            for ind in population:
                f, rc, feas = self.fitness_of(ind)
                fitnesses.append(f)
                raw_costs.append(rc)
                feasibilities.append(feas)
                if f > best_fitness and feas:
                    best_fitness = f
                    best_solution = ind.copy()
                    best_raw_cost = rc
                    # store leg results from simulate route again to produce final CSV later
                    _, legs, _ = self.simulator.simulate_route(ind.order_ids, ind.dias, ind.horas_float, ind.velocidades)
                    best_leg_results = legs

            # Print progress occasionally
            if gen % 50 == 0 or gen == self.generations - 1:
                print(f"Gen {gen}/{self.generations}: best_fitness={best_fitness:.8f}, best_cost={best_raw_cost:.2f}")

            # Create new generation
            new_pop: List[Chromosome] = []
            # elitism: carry top individuals
            if self.elitism:
                # sort population by fitness descending
                sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
                for k in range(min(self.elitism_count, len(population))):
                    new_pop.append(population[sorted_idx[k]].copy())

            # generate rest
            while len(new_pop) < self.population_size:
                # selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                # crossover for order_ids via PMX
                child1 = parent1.copy()
                child2 = parent2.copy()
                if random.random() < CROSSOVER_RATE:
                    o1, o2 = pmx_crossover(parent1.order_ids, parent2.order_ids)
                    child1.order_ids = o1
                    child2.order_ids = o2
                    # crossover numeric vectors single point
                    d1, d2 = single_point_crossover_vec(parent1.dias, parent2.dias)
                    h1, h2 = single_point_crossover_vec(parent1.horas_float, parent2.horas_float)
                    v1, v2 = single_point_crossover_vec(parent1.velocidades, parent2.velocidades)
                    child1.dias = d1
                    child1.horas_float = h1
                    child1.velocidades = v1
                    child2.dias = d2
                    child2.horas_float = h2
                    child2.velocidades = v2
                # mutation: permutation and numeric vectors
                mutate_permutation_swap(child1.order_ids, self.mutation_rate)
                mutate_inversion(child1.order_ids, self.mutation_rate)
                mutate_displacement(child1.order_ids, self.mutation_rate)
                mutate_numeric_vector(child1.dias, 1, MAX_DAYS, self.mutation_rate)
                mutate_numeric_vector(child1.horas_float, float(FLIGHT_WINDOW_START_HOUR), float(FLIGHT_WINDOW_END_HOUR), self.mutation_rate)
                mutate_numeric_vector(child1.velocidades, min(VALID_SPEEDS), max(VALID_SPEEDS), self.mutation_rate)

                mutate_permutation_swap(child2.order_ids, self.mutation_rate)
                mutate_inversion(child2.order_ids, self.mutation_rate)
                mutate_displacement(child2.order_ids, self.mutation_rate)
                mutate_numeric_vector(child2.dias, 1, MAX_DAYS, self.mutation_rate)
                mutate_numeric_vector(child2.horas_float, float(FLIGHT_WINDOW_START_HOUR), float(FLIGHT_WINDOW_END_HOUR), self.mutation_rate)
                mutate_numeric_vector(child2.velocidades, min(VALID_SPEEDS), max(VALID_SPEEDS), self.mutation_rate)

                new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    new_pop.append(child2)

            population = new_pop

        # After all generations, return best found (if none feasible found, pick best even if infeasible)
        if best_solution is None:
            # pick best by raw cost anyway
            best_idx = np.argmin(raw_costs)
            best_solution = population[best_idx].copy()
            best_fitness, best_raw_cost, _ = self.fitness_of(best_solution)
            _, best_leg_results, _ = self.simulator.simulate_route(best_solution.order_ids, best_solution.dias, best_solution.horas_float, best_solution.velocidades)

        return best_solution, best_raw_cost, best_leg_results


# -----------------------
# CSV Output
# -----------------------


def write_output_csv(leg_results: List[LegResult], output_path: str = "rota_otimizada.csv"):
    """
    Writes CSV with columns:
    CEP_inicial, Latitude_inicial, Longitude_inicial, Dia_do_voo, Hora_inicial, Velocidade, CEP_final, Latitude_final, Longitude_final, Pouso (SIM/NÃO), Hora_final
    """
    headers = ["CEP_inicial", "Latitude_inicial", "Longitude_inicial",
               "Dia_do_voo", "Hora_inicial", "Velocidade",
               "CEP_final", "Latitude_final", "Longitude_final",
               "Pouso", "Hora_final"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for leg in leg_results:
            writer.writerow([
                leg.cep_from,
                f"{leg.lat_from:.8f}",
                f"{leg.lon_from:.8f}",
                int(leg.dia),
                float_hour_to_hhmm(leg.hora_inicio_float),
                int(leg.velocidade_kmh),
                leg.cep_to,
                f"{leg.lat_to:.8f}",
                f"{leg.lon_to:.8f}",
                leg.pouso,
                float_hour_to_hhmm(leg.hora_final_float)
            ])


# -----------------------
# Main execution
# -----------------------


def main():
    print("Iniciando carregamento de dados...")
    loader = DataLoader(coords_path="coordenadas.csv", wind_path="vento.csv")
    try:
        ceps = loader.load_coords()
    except Exception as e:
        print(f"Erro ao carregar coordenadas: {e}")
        sys.exit(1)
    try:
        wind = loader.load_wind()
    except Exception as e:
        print(f"Erro ao carregar vento: {e}")
        sys.exit(1)

    print(f"Loaded {len(ceps)} CEPs (start is id=1).")
    print("Construindo matriz de distâncias...")
    dist_calc = DistanceCalculator(ceps)
    wind_model = WindModel(wind)
    autonomy_model = DroneAutonomy(base_autonomy_seconds=BASE_AUTONOMY_SECONDS, correction=CORRECTION_FACTOR)

    ga = GeneticAlgorithm(
        ceps=ceps,
        dist_calc=dist_calc,
        wind_model=wind_model,
        autonomy_model=autonomy_model,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        tournament_k=TOURNAMENT_K,
        elitism=ELITISM,
        elitism_count=ELITISM_COUNT
    )

    print("Executando algoritmo genético...")
    best_solution, best_cost, best_leg_results = ga.evolve()

    print("Melhor solução encontrada:")
    print(f" Custo total (unidade escalada): {best_cost:.2f}")
    # Write CSV
    write_output_csv(best_leg_results, output_path="output_gpt4.csv")
    print("Arquivo 'output_gpt4.csv' gerado com o roteiro otimizado.")

    # Optionally print summary of legs (brief)
    total_time = sum(l.time_flight_s for l in best_leg_results)
    total_recharges = sum(1 for l in best_leg_results if l.pouso == "SIM")
    total_recharge_cost = sum(l.RECHARGE_COST for l in best_leg_results)
    print(f"Resumo: tempo de voo total {int(total_time)} s ({time_to_hhmmss(int(total_time))}), "
          f"recharges {int(total_recharges)}, custo recargas R$ {total_recharge_cost:.2f}")


if __name__ == "__main__":
    main()
