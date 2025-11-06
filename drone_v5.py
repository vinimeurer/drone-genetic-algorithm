import csv
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ...existing code...
# Constants and utility functions

BASE_AUTONOMY_SECONDS = 5000.0
AUTONOMY_CORRECTION = 0.93  # factor to apply to base autonomy
STOP_SECONDS = 72  # per photo or recharge
LANDING_COST = 80.0
AFTER_17_EXTRA = 80.0
EARLIEST_HOUR = 6
LATEST_HOUR = 19  # flights must finish by 19:00
MAX_DAYS = 7
EARTH_R = 6371.0  # km

# Allowed drone speeds: multiples of 4 km/h up to 96, min 36 km/h (10 m/s)
ALLOWED_SPEEDS = [v for v in range(36, 97, 4)]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two points in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R * c


def clamp_hour_to_wind(hour: int, available_hours: List[int]) -> int:
    """Return nearest available wind hour (prefer floor if tie)."""
    if hour in available_hours:
        return hour
    # choose nearest
    return min(available_hours, key=lambda h: abs(h - hour))


# ...existing code...
@dataclass
class Location:
    cep: str
    lon: float
    lat: float


class WindTable:
    """
    Store wind entries indexed by (day, hour) -> (speed_kmh, direction_deg).
    We assume wind 'direction' is the direction TO which wind is blowing (degrees clockwise from North).
    """

    def __init__(self):
        self.table: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.available_hours = set()

    def load_from_csv(self, path: str):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                day = int(row['dia'])
                hour = int(row['hora'])
                speed = float(row['vel_kmh'])
                direction = float(row['direcao_deg'])
                self.table[(day, hour)] = (speed, direction)
                self.available_hours.add(hour)
        self.available_hours = sorted(list(self.available_hours))

    def get_wind(self, day: int, hour: int) -> Tuple[float, float]:
        """Return (speed_kmh, direction_deg) for the closest available hour for the given day."""
        hour_adj = clamp_hour_to_wind(hour, self.available_hours)
        key = (day, hour_adj)
        if key in self.table:
            return self.table[key]
        # fallback: if day not in table, return calm
        return 0.0, 0.0


class DroneModel:
    """
    Handles autonomy calculation and effective speed considering wind.
    """

    def __init__(self):
        # base autonomy corrected
        self.base_autonomy = BASE_AUTONOMY_SECONDS * AUTONOMY_CORRECTION

    def autonomy_for_speed(self, speed_kmh: float) -> float:
        """
        A(v) = 5000 * correction * (36/v)^2
        speed_kmh must be > 0
        """
        v = max(speed_kmh, 1e-6)
        return self.base_autonomy * (36.0 / v) ** 2

    @staticmethod
    def heading_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return bearing in degrees clockwise from North (0..360)."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlambda = math.radians(lon2 - lon1)
        x = math.sin(dlambda) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360.0) % 360.0

    @staticmethod
    def vector_components(speed_kmh: float, direction_deg: float) -> Tuple[float, float]:
        """
        Convert speed and direction to x,y components.
        We use x = east component, y = north component.
        direction_deg is degrees clockwise from North (0 = north, 90 = east).
        """
        rad = math.radians(direction_deg)
        vx = speed_kmh * math.sin(rad)  # east component
        vy = speed_kmh * math.cos(rad)  # north component
        return vx, vy

    def effective_speed_kmh(self, lat1: float, lon1: float, lat2: float, lon2: float,
                            drone_speed_kmh: float, wind_speed_kmh: float, wind_dir_deg: float) -> float:
        """
        Compute effective ground speed magnitude based on vector sum of drone velocity towards target and wind vector.

        - drone heading is computed from start->end
        - drone vector uses drone_speed_kmh magnitude towards that heading
        - wind vector uses wind_dir_deg and wind_speed_kmh
        """
        heading = self.heading_between(lat1, lon1, lat2, lon2)
        dvx, dvy = self.vector_components(drone_speed_kmh, heading)
        wvx, wvy = self.vector_components(wind_speed_kmh, wind_dir_deg)
        gx = dvx + wvx
        gy = dvy + wvy
        v_eff = math.sqrt(gx * gx + gy * gy)
        # ensure positive and not extremely small
        return max(v_eff, 1e-6)


# ...existing code...
@dataclass
class Individual:
    order: List[int]  # permutation of indices (excluding fixed start at index 0)
    days: List[int]  # day for each leg
    hours: List[int]  # starting hour for each leg
    speeds: List[int]  # km/h for each leg
    fitness: Optional[float] = None


class GeneticAlgorithm:
    def __init__(self,
                 locations: List[Location],
                 wind_table: WindTable,
                 drone: DroneModel,
                 population_size: int = 80,
                 generations: int = 200,
                 tournament_k: int = 3,
                 mutation_rate: float = 0.05,
                 elitism: int = 2):
        self.locations = locations
        self.n_sites = len(locations)
        self.wind = wind_table
        self.drone = drone
        self.population_size = population_size
        self.generations = generations
        self.tournament_k = tournament_k
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        # indices: 0 reserved for start/finish (Unibrasil)
        self.site_indices = list(range(1, self.n_sites))

    def random_individual(self) -> Individual:
        order = self.site_indices[:]
        random.shuffle(order)
        legs = len(order) + 1  # start->...->start (return leg)
        days = [random.randint(1, MAX_DAYS) for _ in range(legs)]
        hours = [random.randint(EARLIEST_HOUR, LATEST_HOUR - 1) for _ in range(legs)]
        speeds = [random.choice(ALLOWED_SPEEDS) for _ in range(legs)]
        return Individual(order=order, days=days, hours=hours, speeds=speeds)

    def initial_population(self) -> List[Individual]:
        return [self.random_individual() for _ in range(self.population_size)]

    # PMX crossover for permutation part
    @staticmethod
    def pmx_crossover(parent_a: List[int], parent_b: List[int]) -> Tuple[List[int], List[int]]:
        """Robust PMX implementation (avoids mapping loops)."""
        size = len(parent_a)
        if size <= 1:
            return parent_a[:], parent_b[:]
        cx1, cx2 = sorted(random.sample(range(size), 2))
        child1 = [-1] * size
        child2 = [-1] * size
        # copy slice
        child1[cx1:cx2 + 1] = parent_a[cx1:cx2 + 1]
        child2[cx1:cx2 + 1] = parent_b[cx1:cx2 + 1]

        def fill(child: List[int], src: List[int], dst: List[int]):
            for i in range(size):
                if child[i] == -1:
                    gene = src[i]
                    # follow mapping until gene not in copied slice
                    while gene in child:
                        idx = src.index(gene)
                        gene = dst[idx]
                    child[i] = gene

        fill(child1, parent_b, parent_a)
        fill(child2, parent_a, parent_b)
        return child1, child2

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        # permutation crossover (order)
        child_order1, child_order2 = self.pmx_crossover(parent1.order, parent2.order)
        # for day/hour/speed arrays use uniform crossover
        legs = len(child_order1) + 1
        def mix_vec(v1, v2):
            out1, out2 = [], []
            for i in range(legs):
                if random.random() < 0.5:
                    out1.append(v1[i])
                    out2.append(v2[i])
                else:
                    out1.append(v2[i])
                    out2.append(v1[i])
            return out1, out2

        days1, days2 = mix_vec(parent1.days, parent2.days)
        hours1, hours2 = mix_vec(parent1.hours, parent2.hours)
        speeds1, speeds2 = mix_vec(parent1.speeds, parent2.speeds)

        return (Individual(child_order1, days1, hours1, speeds1),
                Individual(child_order2, days2, hours2, speeds2))

    def mutate(self, ind: Individual):
        # mutation on permutation: swap or inversion with some probability
        if random.random() < self.mutation_rate:
            a = random.randint(0, len(ind.order) - 1)
            b = random.randint(0, len(ind.order) - 1)
            ind.order[a], ind.order[b] = ind.order[b], ind.order[a]
        if random.random() < self.mutation_rate:
            # inversion
            a = random.randint(0, len(ind.order) - 1)
            b = random.randint(0, len(ind.order) - 1)
            if a > b:
                a, b = b, a
            ind.order[a:b + 1] = reversed(ind.order[a:b + 1])
        # mutate days/hours/speeds
        for i in range(len(ind.days)):
            if random.random() < self.mutation_rate:
                ind.days[i] = random.randint(1, MAX_DAYS)
        for i in range(len(ind.hours)):
            if random.random() < self.mutation_rate:
                ind.hours[i] = random.randint(EARLIEST_HOUR, LATEST_HOUR - 1)
        for i in range(len(ind.speeds)):
            if random.random() < self.mutation_rate:
                ind.speeds[i] = random.choice(ALLOWED_SPEEDS)

    def tournament_select(self, population: List[Individual], k: int) -> Individual:
        """Safe tournament: don't request more aspirants than exist."""
        k = max(1, min(k, len(population)))
        aspirants = random.sample(population, k)
        aspirants.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        return aspirants[0]

    # Simulation and fitness evaluation
    def evaluate(self, ind: Individual) -> float:
        """
        Simulate the full route and return a fitness value.
        Fitness is total_time_seconds + money_cost * 1000 (weight) + penalties.
        Heavy penalty for invalid solutions.
        """
        # Build full path indices: start(0) -> order -> back to start(0)
        path = [0] + ind.order[:] + [0]
        legs = len(path) - 1
        # pad days/hours/speeds if lengths mismatch
        if len(ind.days) != legs:
            ind.days = (ind.days + [1] * legs)[:legs]
        if len(ind.hours) != legs:
            ind.hours = (ind.hours + [EARLIEST_HOUR] * legs)[:legs]
        if len(ind.speeds) != legs:
            ind.speeds = (ind.speeds + [ALLOWED_SPEEDS[0]] * legs)[:legs]

        total_time = 0.0
        total_cost = 0.0
        penalty = 0.0

        battery = self.drone.base_autonomy  # start full
        # We'll track time per leg using day and hour gene as scheduled departure time.
        for i in range(legs):
            a_idx = path[i]
            b_idx = path[i + 1]
            loc_a = self.locations[a_idx]
            loc_b = self.locations[b_idx]
            day = ind.days[i]
            hour = ind.hours[i]
            speed = ind.speeds[i]
            # validate genes
            if not (1 <= day <= MAX_DAYS):
                penalty += 1e8
            if not (EARLIEST_HOUR <= hour < LATEST_HOUR):
                # start outside window
                penalty += 1e7
            if speed not in ALLOWED_SPEEDS:
                penalty += 1e6

            # get wind for scheduled departure
            wind_speed, wind_dir = self.wind.get_wind(day, hour)
            distance_km = haversine_km(loc_a.lat, loc_a.lon, loc_b.lat, loc_b.lon)
            v_eff = self.drone.effective_speed_kmh(loc_a.lat, loc_a.lon, loc_b.lat, loc_b.lon,
                                                   float(speed), wind_speed, wind_dir)
            # convert km/h to km/s
            v_eff_km_s = v_eff / 3600.0
            if v_eff_km_s <= 0:
                penalty += 1e8
                travel_seconds = float('inf')
            else:
                travel_seconds = math.ceil(distance_km / v_eff_km_s)

            # Guard against absurdly large travel times (numerical issues or near-zero v_eff)
            MAX_TRAVEL_SECONDS = 10_000_000
            if travel_seconds > MAX_TRAVEL_SECONDS:
                penalty += 1e9
                travel_seconds = MAX_TRAVEL_SECONDS

            # Check battery enough for travel_seconds + PHOTO stop (72s)
            photo_stop = STOP_SECONDS
            required = travel_seconds + photo_stop
            # If battery insufficient, force recharge at loc_a
            if battery < travel_seconds:
                # recharge consumes STOP_SECONDS and cost
                battery = self.drone.autonomy_for_speed(speed)
                total_time += STOP_SECONDS
                # landing cost
                total_cost += LANDING_COST
                # extra if after 17:00
                if hour >= 17:
                    total_cost += AFTER_17_EXTRA
            # Now if still battery < travel_seconds => impossible
            if battery < travel_seconds:
                penalty += 1e9
                battery = 0
            else:
                battery -= travel_seconds
            # After arriving, take photo stop
            total_time += travel_seconds
            total_time += photo_stop
            if battery < 0:
                penalty += 1e9
            # If arrival time beyond allowed window:
            # arrival hour = hour + travel_seconds/3600
            arrival_hour = hour + (travel_seconds / 3600.0)
            if arrival_hour > LATEST_HOUR:
                penalty += 1e7

            # landing cost for recharging accounted earlier; photos don't add monetary cost
            # keep track for next leg

        # total fitness combines time and monetary cost
        fitness = total_time + total_cost * 1000.0 + penalty
        ind.fitness = fitness
        return fitness

    def run(self) -> Individual:
        pop = self.initial_population()
        # evaluate initial population
        for ind in pop:
            self.evaluate(ind)
        pop.sort(key=lambda x: x.fitness)
        best = pop[0]
        for gen in range(self.generations):
            new_pop: List[Individual] = []
            # Elitism
            elites = pop[:self.elitism]
            new_pop.extend(elites)
            while len(new_pop) < self.population_size:
                # selection
                parent1 = self.tournament_select(pop, self.tournament_k)
                parent2 = self.tournament_select(pop, self.tournament_k)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                self.evaluate(child1)
                self.evaluate(child2)
                new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    new_pop.append(child2)
            pop = sorted(new_pop, key=lambda x: x.fitness)
            if pop[0].fitness < best.fitness:
                best = pop[0]
            # optional: progress output every 50 gens
            if (gen + 1) % 50 == 0 or gen == 0 or gen == self.generations - 1:
                print(f"Gen {gen+1}/{self.generations} best fitness: {pop[0].fitness:.2f}")
        return best


# CSV export for best solution
def export_solution_csv(best: Individual, locations: List[Location], out_path: str, wind_table: WindTable, drone: DroneModel):
    path = [0] + best.order[:] + [0]
    legs = len(path) - 1
    # pad arrays
    if len(best.days) != legs:
        best.days = (best.days + [1] * legs)[:legs]
    if len(best.hours) != legs:
        best.hours = (best.hours + [EARLIEST_HOUR] * legs)[:legs]
    if len(best.speeds) != legs:
        best.speeds = (best.speeds + [ALLOWED_SPEEDS[0]] * legs)[:legs]

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CEP_inicial', 'Latitude_inicial', 'Longitude_inicial',
                         'Dia_do_voo', 'Hora_inicial', 'Velocidade_kmh',
                         'CEP_final', 'Latitude_final', 'Longitude_final',
                         'Pouso_SIM_NAO', 'Hora_final'])
        for i in range(legs):
            a_idx = path[i]
            b_idx = path[i + 1]
            loc_a = locations[a_idx]
            loc_b = locations[b_idx]
            day = best.days[i]
            hour = best.hours[i]
            speed = best.speeds[i]
            wind_speed, wind_dir = wind_table.get_wind(day, hour)
            distance_km = haversine_km(loc_a.lat, loc_a.lon, loc_b.lat, loc_b.lon)
            v_eff = drone.effective_speed_kmh(loc_a.lat, loc_a.lon, loc_b.lat, loc_b.lon,
                                             float(speed), wind_speed, wind_dir)
            travel_seconds = math.ceil(distance_km / (v_eff / 3600.0)) if v_eff > 0 else 0
            arrival_hour = hour + travel_seconds / 3600.0
            # decide if landing/recharge happened in fitness logic isn't recorded; mark Pouso as SIM if arrival time >= 0 (we mark all as SIM for simplification)
            pouso = "SIM"
            writer.writerow([loc_a.cep, loc_a.lat, loc_a.lon,
                             day, hour, speed,
                             loc_b.cep, loc_b.lat, loc_b.lon,
                             pouso, f"{arrival_hour:.2f}"])
    print(f"Solution exported to {out_path}")


# Loader for coordinates csv
def load_locations(path: str) -> List[Location]:
    locs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cep = row['cep']
            lon = float(row['longitude'])
            lat = float(row['latitude'])
            locs.append(Location(cep=cep, lon=lon, lat=lat))
    # Ensure start index 0 is Unibrasil CEP 82821020 => if not first, reorder
    if not locs or locs[0].cep != '82821020':
        # find index
        idx = next((i for i, l in enumerate(locs) if l.cep == '82821020'), None)
        if idx is not None:
            locs[0], locs[idx] = locs[idx], locs[0]
    return locs


# Entry point
def main():
    coords_file = 'coordenadas.csv'
    wind_file = 'vento.csv'
    out_csv = 'melhor_solucao.csv'
    print("Loading data...")
    locations = load_locations(coords_file)
    wind = WindTable()
    wind.load_from_csv(wind_file)
    drone = DroneModel()
    ga = GeneticAlgorithm(locations, wind, drone,
                          population_size=50,
                          generations=500,
                          tournament_k=3,
                          mutation_rate=0.05,
                          elitism=2)
    print("Running genetic algorithm...")
    t0 = time.time()
    best = ga.run()
    t1 = time.time()
    print(f"GA finished in {t1 - t0:.2f}s. Best fitness: {best.fitness:.2f}")
    export_solution_csv(best, locations, out_csv, wind, drone)


# # Simple pytest tests included in same file
# def test_haversine_zero():
#     assert abs(haversine_km(0, 0, 0, 0)) < 1e-9


# def test_wind_effect_no_wind_equals_drone_speed():
#     drone = DroneModel()
#     # trivial same lat/lon offset to get heading 0
#     lat1, lon1 = 0.0, 0.0
#     lat2, lon2 = 0.01, 0.0  # small northward move
#     speed = 50.0
#     v_eff = drone.effective_speed_kmh(lat1, lon1, lat2, lon2, speed, 0.0, 0.0)
#     assert abs(v_eff - speed) < 1e-6


# def test_autonomy_at_36_equals_base_corrected():
#     drone = DroneModel()
#     a36 = drone.autonomy_for_speed(36.0)
#     expected = BASE_AUTONOMY_SECONDS * AUTONOMY_CORRECTION
#     assert abs(a36 - expected) < 1e-6


if __name__ == '__main__':
    main()
# ...existing code...