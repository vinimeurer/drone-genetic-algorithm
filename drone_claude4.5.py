"""
Drone route optimizer using a Genetic Algorithm.

- Reads /home/vinicius/drone-genetic-algorythm/coordenadas.csv and vento.csv
- Implements classes for distances, wind model, drone autonomy, route simulation and GA
- Produces best_route.csv in the working folder with the final itinerary

Notes / assumptions (kept small and explicit):
- CEP with id 1 is the base (first row in coordenadas.csv, expected 82821020)
- Chromosome encodes permutation of visiting nodes excluding base (so start/end at base)
- Allowed departure hours are the wind table hours: [6, 9, 12, 15, 18]
- Wind direction is treated as the direction TOWARD which wind blows (vector points in that direction)
- If battery is insufficient for a segment, a forced recharge (landing) is performed at current CEP before departure:
  cost R$80 ( + R$80 if landing AFTER 17:00 local hour)
- Monetary costs converted to seconds by factor MONEY_TO_SECONDS = 60 (1 R$ = 60 seconds) to combine time+money in single cost
- Big penalties for invalid solutions (1e7)
- Fitness = 1 / (1 + total_cost)
"""

import csv
import math
import random
import copy
import sys
from typing import Dict, List, Tuple

# ----------------------------
# Configuration / Constants
# ----------------------------
COORD_FILE = "coordenadas.csv"
WIND_FILE = "vento.csv"
OUTPUT_FILE = "output_claude4.5.csv"

R_EARTH_KM = 6371.0
BASE_AUTONOMY_SECONDS = 5000
AUTONOMY_CORRECTION = 0.93
AUTONOMY_BASE = BASE_AUTONOMY_SECONDS * AUTONOMY_CORRECTION  # 4650 seconds
PHOTO_STOP_SECONDS = 72
LANDING_COST = 80
LATE_LANDING_EXTRA = 80
MONEY_TO_SECONDS = 60  # convert R$ to seconds in objective combination

VALID_SPEEDS = list(range(36, 97, 4))  # 36,40,...,96 km/h
VALID_HOURS = [6, 9, 12, 15, 18]  # departure hours that align with wind data
MIN_DAY = 1
MAX_DAY = 7

PENALTY_INVALID = 1e7

# GA hyperparameters (tweakable)
POPULATION_SIZE = 80
GENERATIONS = 250
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.08
ELITISM = 4
CROSSOVER_NUMERIC_POINT = 1  # used for days/hours/speeds single point crossover


# ----------------------------
# Utilities / Data Loader
# ----------------------------
class DataLoader:
    def __init__(self, coord_path: str, wind_path: str):
        self.coord_path = coord_path
        self.wind_path = wind_path
        self.nodes = {}  # id -> {'cep', 'longitude', 'latitude'}
        self.id_by_cep = {}
        self.wind = {}  # wind[day][hour] = {'velocidade_kmh', 'direcao_graus'}
        self._load_coordinates()
        self._load_wind()

    def _load_coordinates(self):
        path = self.coord_path
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                idx = 1
                for r in reader:
                    cep = r['cep'].strip()
                    lon = float(r['longitude'])
                    lat = float(r['latitude'])
                    self.nodes[idx] = {'id': idx, 'cep': cep, 'longitude': lon, 'latitude': lat}
                    self.id_by_cep[cep] = idx
                    idx += 1
        except Exception as e:
            print(f"Error loading coordinates file {path}: {e}")
            sys.exit(1)

    def _load_wind(self):
        path = self.wind_path
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    day = int(r['dia'])
                    hour = int(r['hora'])
                    vel = float(r['vel_kmh'])
                    direc = float(r['direcao_deg'])
                    self.wind.setdefault(day, {})[hour] = {'velocidade_kmh': vel, 'direcao_graus': direc}
        except Exception as e:
            print(f"Error loading wind file {path}: {e}")
            sys.exit(1)


# ----------------------------
# Geometry / Haversine / Bearing
# ----------------------------
class Geo:
    @staticmethod
    def haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> float:
        # convert degrees to radians
        lat1 = math.radians(lat1_deg)
        lon1 = math.radians(lon1_deg)
        lat2 = math.radians(lat2_deg)
        lon2 = math.radians(lon2_deg)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R_EARTH_KM * c

    @staticmethod
    def bearing_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> float:
        # Bearing from point1 to point2 in degrees from North clockwise
        lat1 = math.radians(lat1_deg)
        lon1 = math.radians(lon1_deg)
        lat2 = math.radians(lat2_deg)
        lon2 = math.radians(lon2_deg)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing_rad = math.atan2(x, y)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        return bearing_deg


# ----------------------------
# Wind Model
# ----------------------------
class WindModel:
    def __init__(self, wind_data: Dict[int, Dict[int, dict]]):
        self.wind_data = wind_data

    def get_wind(self, day: int, hour: int) -> dict:
        # If exact hour available return it, otherwise pick nearest known hour for that day.
        if day not in self.wind_data:
            # fallback: return calm wind
            return {'velocidade_kmh': 0.0, 'direcao_graus': 0.0}
        day_winds = self.wind_data[day]
        if hour in day_winds:
            return day_winds[hour]
        # find nearest hour key
        available = sorted(day_winds.keys())
        nearest = min(available, key=lambda h: abs(h - hour))
        return day_winds[nearest]


# ----------------------------
# Drone Autonomy & Speed effects
# ----------------------------
class Drone:
    def __init__(self, base_autonomy_seconds: float = AUTONOMY_BASE):
        self.base = base_autonomy_seconds  # 4650 by problem

    def autonomy_seconds(self, v_kmh: float) -> float:
        # A(v) = 5000 * 0.93 * (36 / v)^2 = base * (36 / v)^2
        if v_kmh <= 0:
            return 0.0
        return self.base * (36.0 / v_kmh) ** 2


# ----------------------------
# Route Simulation (fitness calc)
# ----------------------------
class RouteSimulator:
    def __init__(self, nodes: Dict[int, dict], wind_model: WindModel, drone: Drone):
        self.nodes = nodes
        self.wind_model = wind_model
        self.drone = drone
        # Precompute distance matrix and bearing matrix
        self.n = len(nodes)
        self.dist_matrix = [[0.0] * (self.n + 1) for _ in range(self.n + 1)]
        self.bearing_matrix = [[0.0] * (self.n + 1) for _ in range(self.n + 1)]
        self._precompute()

    def _precompute(self):
        for i in range(1, self.n + 1):
            a = self.nodes[i]
            for j in range(1, self.n + 1):
                b = self.nodes[j]
                if i == j:
                    self.dist_matrix[i][j] = 0.0
                    self.bearing_matrix[i][j] = 0.0
                else:
                    d = Geo.haversine_km(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
                    brg = Geo.bearing_deg(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
                    self.dist_matrix[i][j] = d
                    self.bearing_matrix[i][j] = brg

    @staticmethod
    def _vec_components(speed_kmh: float, heading_deg: float) -> Tuple[float, float]:
        # Convert heading in degrees from North clockwise to vector components (x east, y north)
        rad = math.radians(heading_deg)
        vx = speed_kmh * math.sin(rad)
        vy = speed_kmh * math.cos(rad)
        return vx, vy

    def simulate_route(self, order_perm: List[int], days: List[int], hours: List[int], speeds: List[int]) -> Tuple[float, dict]:
        """
        Simulate full route and return (total_cost_seconds_combined, details)
        order_perm: permutation of node ids excluding base (1)
        days/hours/speeds: lists aligned with segments (length = number of segments)
        The full route is: base (1) -> perm[0] -> perm[1] -> ... -> perm[-1] -> base (1)
        """
        # Build full path
        base = 1
        full_path = [base] + order_perm + [base]
        segments = len(full_path) - 1

        # Validate input lengths
        if not (len(days) == len(hours) == len(speeds) == segments):
            # invalid chromosome; heavy penalty
            return PENALTY_INVALID, {'reason': 'length_mismatch'}

        total_flight_time = 0.0  # seconds (flight times)
        total_photo_time = 0.0
        total_money_cost = 0.0  # R$
        battery = self.drone.autonomy_seconds(36.0)  # start fully charged at baseline speed 36? Use full battery.
        battery = self.drone.base  # full base autonomy
        prev_arrival_time = -1.0

        # We'll track each segment detail for output CSV
        details_lines = []

        for seg_idx in range(segments):
            from_id = full_path[seg_idx]
            to_id = full_path[seg_idx + 1]
            day = days[seg_idx]
            hour = hours[seg_idx]
            v_choice = speeds[seg_idx]

            # Basic validity checks
            if day < MIN_DAY or day > MAX_DAY:
                return PENALTY_INVALID, {'reason': 'day_out_of_range', 'seg': seg_idx + 1}
            if hour < 6 or hour > 18:
                return PENALTY_INVALID, {'reason': 'hour_out_of_range', 'seg': seg_idx + 1}
            if hour not in VALID_HOURS:
                # allow but penalize small (not fatal) by rounding to nearest valid when computing wind
                pass
            if v_choice not in VALID_SPEEDS:
                return PENALTY_INVALID, {'reason': 'invalid_speed', 'seg': seg_idx + 1}

            # Get segment geometry
            dist_km = self.dist_matrix[from_id][to_id]
            if dist_km == 0.0 and from_id != to_id:
                # degenerate
                return PENALTY_INVALID, {'reason': 'zero_distance_unexpected', 'seg': seg_idx + 1}

            bearing = self.bearing_matrix[from_id][to_id]
            # Get wind at departure
            wind = self.wind_model.get_wind(day, hour)
            wind_speed = wind['velocidade_kmh']
            wind_dir = wind['direcao_graus']

            # Compute effective vector: drone vector + wind vector
            v_drone = v_choice
            dgx, dgy = self._vec_components(v_drone, bearing)
            wx, wy = self._vec_components(wind_speed, wind_dir)
            v_eff = math.hypot(dgx + wx, dgy + wy)
            # ensure v_eff not extremely small
            if v_eff < 1e-6:
                return PENALTY_INVALID, {'reason': 'v_eff_zero', 'seg': seg_idx + 1}

            # Flight time in seconds
            time_sec = math.ceil((dist_km * 3600.0) / v_eff)
            # Battery required is time_sec
            # If battery insufficient, recharge at 'from' before departure
            recharge_performed = False
            recharge_cost_here = 0.0
            if battery < time_sec:
                # perform recharge at 'from' node
                recharge_performed = True
                # landing time consumes PHOTO_STOP_SECONDS (use same duration)
                # and battery resets to full
                battery = self.drone.base
                recharge_cost_here += LANDING_COST
                # landing time occurs at departure time; if landing after 17:00 local -> extra cost
                # Local time of landing assumed to be departure hour (since we recharge at that time)
                if hour >= 17:
                    recharge_cost_here += LATE_LANDING_EXTRA
                # Add landing time (we'll count it as photo/stop time)
                total_photo_time += PHOTO_STOP_SECONDS

            # Now check battery again (if still insufficient -> impossible)
            if battery < time_sec:
                # can't perform segment even after recharge -> invalid
                return PENALTY_INVALID, {'reason': 'insufficient_battery_after_recharge', 'seg': seg_idx + 1}

            # Consume battery by flight time
            battery -= time_sec
            total_flight_time += time_sec
            total_money_cost += recharge_cost_here

            # Arrival: perform the photo stop (unless it's the final base return and maybe not needed; spec says photos at each CEP)
            total_photo_time += PHOTO_STOP_SECONDS
            battery -= PHOTO_STOP_SECONDS
            if battery < 0:
                # battery zero during/after photo -> force recharge at arrival (count landing at arrival)
                # But if arrival photo consumed battery negative, we must have recharged at arrival before photo, which implies landing cost.
                battery = self.drone.base
                total_money_cost += LANDING_COST
                # if arrival local hour after 17:00 add extra
                arrival_hour = hour  # approximate; not computing exact crossing of days/hours
                if arrival_hour >= 17:
                    total_money_cost += LATE_LANDING_EXTRA
                # battery reset; subtract photo time
                battery -= PHOTO_STOP_SECONDS
                if battery < 0:
                    return PENALTY_INVALID, {'reason': 'battery_negative_after_arrival_recharge', 'seg': seg_idx + 1}

            # record details
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]
            # compute arrival hour approximate: departure hour plus flight time; we won't convert to hh:mm here precisely
            # but compute arrival hour in fractional hours to check late landing extra if needed
            arrival_time_seconds = ((day - 1) * 86400) + (hour * 3600) + time_sec
            arrival_local_hour = (arrival_time_seconds % 86400) // 3600
            landed_flag = 'SIM' if recharge_performed else 'NÃO'
            details_lines.append({
                'CEP_inicial': from_node['cep'],
                'Latitude_inicial': from_node['latitude'],
                'Longitude_inicial': from_node['longitude'],
                'Dia_do_voo': day,
                'Hora_inicial': hour,
                'Velocidade': v_choice,
                'CEP_final': to_node['cep'],
                'Latitude_final': to_node['latitude'],
                'Longitude_final': to_node['longitude'],
                'Pouso': landed_flag,
                'Hora_final': int(arrival_local_hour)
            })

        # Combine costs: flight time + photo time + money converted to seconds
        total_time_sec = total_flight_time + total_photo_time
        money_seconds = total_money_cost * MONEY_TO_SECONDS
        total_cost = total_time_sec + money_seconds

        # Return details for CSV and cost
        info = {
            'total_flight_time_sec': total_flight_time,
            'total_photo_time_sec': total_photo_time,
            'total_money_R$': total_money_cost,
            'total_cost_combined': total_cost,
            'lines': details_lines
        }
        return total_cost, info


# ----------------------------
# Genetic Algorithm Utilities
# ----------------------------
def pmx_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """PMX crossover for permutations. parent lists of equal length."""
    size = len(parent1)
    if size <= 2:
        return parent1[:], parent2[:]
    a, b = sorted(random.sample(range(size), 2))
    def pmx(p1, p2):
        child = [-1] * size
        # copy slice from p1
        child[a:b+1] = p1[a:b+1]
        # mapping from slice of p2 -> p1
        for i in range(a, b+1):
            val = p2[i]
            if val not in child:
                pos = i
                mapped = p1[pos]
                while True:
                    pos = p2.index(mapped)
                    if child[pos] == -1:
                        child[pos] = val
                        break
                    mapped = p1[pos]
        # fill remaining with p2 values
        for i in range(size):
            if child[i] == -1:
                child[i] = p2[i]
        return child
    return pmx(parent1, parent2), pmx(parent2, parent1)


def tournament_selection(population: List[dict], k: int) -> dict:
    contenders = random.sample(population, k)
    contenders.sort(key=lambda ind: ind['fitness'], reverse=True)
    return contenders[0]


def single_point_crossover_numeric(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
    size = len(a)
    if size <= 1:
        return a[:], b[:]
    pt = random.randint(1, size - 1)
    c1 = a[:pt] + b[pt:]
    c2 = b[:pt] + a[pt:]
    return c1, c2


def mutate_permutation_swap(perm: List[int], rate: float) -> List[int]:
    p = perm[:]
    for i in range(len(p)):
        if random.random() < rate:
            j = random.randrange(len(p))
            p[i], p[j] = p[j], p[i]
    return p


def mutate_numeric_random_reset(arr: List[int], rate: float, domain_fn):
    r = arr[:]
    for i in range(len(r)):
        if random.random() < rate:
            r[i] = domain_fn()
    return r


# ----------------------------
# Genetic Algorithm Class
# ----------------------------
class GeneticAlgorithm:
    def __init__(self, simulator: RouteSimulator, nodes_count: int):
        self.simulator = simulator
        # nodes_count includes base; number of visits to permute = nodes_count - 1
        self.n_visits = nodes_count - 1

    def random_individual(self) -> dict:
        # permutation of internal nodes (2..n)
        internal_nodes = list(range(2, self.n_visits + 2))  # if n_visits=K, nodes are 2..K+1
        random.shuffle(internal_nodes)
        # segments = n_visits + 1 (including return to base)
        segments = len(internal_nodes) + 1
        days = [random.randint(MIN_DAY, MAX_DAY) for _ in range(segments)]
        hours = [random.choice(VALID_HOURS) for _ in range(segments)]
        speeds = [random.choice(VALID_SPEEDS) for _ in range(segments)]
        return {'perm': internal_nodes, 'days': days, 'hours': hours, 'speeds': speeds, 'fitness': None, 'cost': None, 'info': None}

    def evaluate(self, individual: dict):
        cost, info = self.simulator.simulate_route(individual['perm'], individual['days'], individual['hours'], individual['speeds'])
        individual['cost'] = cost
        if cost >= PENALTY_INVALID:
            individual['fitness'] = 1.0 / (1.0 + cost)
        else:
            individual['fitness'] = 1.0 / (1.0 + cost)
        individual['info'] = info

    def run(self, pop_size=POPULATION_SIZE, generations=GENERATIONS) -> dict:
        # initialize
        population = [self.random_individual() for _ in range(pop_size)]
        for ind in population:
            self.evaluate(ind)

        best = max(population, key=lambda x: x['fitness'])
        print(f"Initial best fitness: {best['fitness']:.8f}, cost: {best['cost']:.2f}")

        for gen in range(generations):
            new_pop = []
            # elitism
            population.sort(key=lambda x: x['fitness'], reverse=True)
            elites = population[:ELITISM]
            new_pop.extend(copy.deepcopy(elites))

            # produce rest
            while len(new_pop) < pop_size:
                # selection
                p1 = tournament_selection(population, TOURNAMENT_SIZE)
                p2 = tournament_selection(population, TOURNAMENT_SIZE)
                # crossover for permutation via PMX
                child_perm1, child_perm2 = pmx_crossover(p1['perm'], p2['perm'])
                # numeric crossover for days/hours/speeds
                child_days1, child_days2 = single_point_crossover_numeric(p1['days'], p2['days'])
                child_hours1, child_hours2 = single_point_crossover_numeric(p1['hours'], p2['hours'])
                child_speeds1, child_speeds2 = single_point_crossover_numeric(p1['speeds'], p2['speeds'])
                # mutation
                child_perm1 = mutate_permutation_swap(child_perm1, MUTATION_RATE)
                child_perm2 = mutate_permutation_swap(child_perm2, MUTATION_RATE)
                # domain functions
                child_days1 = mutate_numeric_random_reset(child_days1, MUTATION_RATE, lambda: random.randint(MIN_DAY, MAX_DAY))
                child_days2 = mutate_numeric_random_reset(child_days2, MUTATION_RATE, lambda: random.randint(MIN_DAY, MAX_DAY))
                child_hours1 = mutate_numeric_random_reset(child_hours1, MUTATION_RATE, lambda: random.choice(VALID_HOURS))
                child_hours2 = mutate_numeric_random_reset(child_hours2, MUTATION_RATE, lambda: random.choice(VALID_HOURS))
                child_speeds1 = mutate_numeric_random_reset(child_speeds1, MUTATION_RATE, lambda: random.choice(VALID_SPEEDS))
                child_speeds2 = mutate_numeric_random_reset(child_speeds2, MUTATION_RATE, lambda: random.choice(VALID_SPEEDS))

                new_pop.append({'perm': child_perm1, 'days': child_days1, 'hours': child_hours1, 'speeds': child_speeds1, 'fitness': None, 'cost': None, 'info': None})
                if len(new_pop) < pop_size:
                    new_pop.append({'perm': child_perm2, 'days': child_days2, 'hours': child_hours2, 'speeds': child_speeds2, 'fitness': None, 'cost': None, 'info': None})

            # evaluate population
            for ind in new_pop:
                if ind['fitness'] is None:
                    self.evaluate(ind)

            population = new_pop
            generation_best = max(population, key=lambda x: x['fitness'])
            if generation_best['fitness'] > best['fitness']:
                best = copy.deepcopy(generation_best)
            if (gen + 1) % 25 == 0 or gen == 0:
                print(f"Gen {gen+1}/{generations} best fitness: {best['fitness']:.8f}, cost: {best['cost']:.2f}")

        return best


# ----------------------------
# CSV output helper
# ----------------------------
def write_route_csv(lines: List[dict], output_path: str):
    fieldnames = ['CEP_inicial', 'Latitude_inicial', 'Longitude_inicial', 'Dia_do_voo', 'Hora_inicial', 'Velocidade', 'CEP_final', 'Latitude_final', 'Longitude_final', 'Pouso', 'Hora_final']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lines:
            writer.writerow(row)


# ----------------------------
# Main execution
# ----------------------------
def main():
    loader = DataLoader(COORD_FILE, WIND_FILE)
    wind_model = WindModel(loader.wind)
    drone = Drone()
    simulator = RouteSimulator(loader.nodes, wind_model, drone)
    ga = GeneticAlgorithm(simulator, nodes_count=len(loader.nodes))

    print("Starting Genetic Algorithm...")
    best = ga.run()

    print(f"Best fitness: {best['fitness']:.8f}, cost: {best['cost']:.2f}")
    # write CSV
    if best['info'] and 'lines' in best['info']:
        write_route_csv(best['info']['lines'], OUTPUT_FILE)
        print(f"Best route written to {OUTPUT_FILE}")
    else:
        print("No valid route details to write.")

if __name__ == "__main__":
    main()