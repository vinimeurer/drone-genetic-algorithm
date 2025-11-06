import csv
import math
import random
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Constants
EARTH_RADIUS_KM = 6371
BASE_AUTONOMY_SECONDS = 5000 * 0.93  # 4650 seconds
STOP_TIME_SECONDS = 72
VALID_SPEEDS = list(range(36, 97, 4))  # 36, 40, ..., 96 km/h
RECHARGE_COST_BASE = 80
RECHARGE_COST_AFTER_17 = 160  # 80 + 80
PENALTY_INVALID = 1e7  # High penalty for invalid solutions
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.05
ELITISM_RATE = 0.1
START_CEP = "82821020"

class DataLoader:
    def __init__(self, coordenadas_file: str, vento_file: str):
        self.cep_to_id = {}
        self.id_to_cep = {}
        self.coordinates = {}  # id -> (lat, lon)
        self.vento = defaultdict(lambda: defaultdict(dict))  # day -> hour -> {'vel': , 'dir': }
        self.load_coordenadas(coordenadas_file)
        self.load_vento(vento_file)

    def load_coordenadas(self, file_path: str):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                cep = row['cep']
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                self.cep_to_id[cep] = idx
                self.id_to_cep[idx] = cep
                self.coordinates[idx] = (lat, lon)

    def load_vento(self, file_path: str):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                day = int(row['dia'])
                hour = int(row['hora'])
                vel = float(row['vel_kmh'])
                dir_deg = float(row['direcao_deg'])
                self.vento[day][hour] = {'vel': vel, 'dir': dir_deg}

class DistanceCalculator:
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_KM * c

    def get_distance_matrix(self, coordinates: Dict[int, Tuple[float, float]]) -> Dict[Tuple[int, int], float]:
        matrix = {}
        ids = list(coordinates.keys())
        for i in ids:
            for j in ids:
                if i != j:
                    lat1, lon1 = coordinates[i]
                    lat2, lon2 = coordinates[j]
                    matrix[(i, j)] = self.haversine(lat1, lon1, lat2, lon2)
        return matrix

class WindCalculator:
    @staticmethod
    def calculate_effective_speed(drone_speed_kmh: float, wind_vel_kmh: float, wind_dir_deg: float, bearing_deg: float) -> float:
        # Bearing is the direction from point A to B
        # Wind direction is where wind is blowing FROM (standard: 0° = north, blowing south)
        wind_rad = math.radians(wind_dir_deg)
        bearing_rad = math.radians(bearing_deg)
        
        # Drone velocity vector (towards bearing)
        v_drone_x = drone_speed_kmh * math.sin(bearing_rad)  # East component
        v_drone_y = drone_speed_kmh * math.cos(bearing_rad)  # North component
        
        # Wind velocity vector: wind blows in the direction opposite to wind_dir_deg
        # If wind_dir_deg = 0 (north wind), it blows south, so x=0, y=-wind_vel
        wind_x = -wind_vel_kmh * math.sin(wind_rad)  # Negative sin for correct direction
        wind_y = -wind_vel_kmh * math.cos(wind_rad)
        
        # Effective velocity
        v_eff_x = v_drone_x + wind_x
        v_eff_y = v_drone_y + wind_y
        v_eff = math.sqrt(v_eff_x**2 + v_eff_y**2)
        return max(v_eff, 0.1)  # Avoid zero or negative

    @staticmethod
    def calculate_autonomy(drone_speed_kmh: float) -> float:
        return BASE_AUTONOMY_SECONDS * (36 / drone_speed_kmh)**2

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dlon = math.radians(lon2 - lon1)
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        bearing = math.atan2(x, y)
        return math.degrees(bearing) % 360

class FitnessEvaluator:
    def __init__(self, data_loader: DataLoader, distance_matrix: Dict[Tuple[int, int], float]):
        self.data_loader = data_loader
        self.distance_matrix = distance_matrix
        self.wind_calc = WindCalculator()
        self.start_id = data_loader.cep_to_id[START_CEP]

    def evaluate(self, chromosome: List[int], days: List[int], hours: List[int], speeds: List[float]) -> float:
        # chromosome: order of IDs, starting and ending with start_id
        # days: day for each leg (len = len(chromosome)-1)
        # hours: departure hour for each leg
        # speeds: speed for each leg
        total_cost = 0
        battery = 0  # Start with empty battery, recharge at start
        current_time_seconds = 0  # Total seconds from day 1, 00:00
        invalid = False
        
        for i in range(len(chromosome) - 1):
            cep_from = chromosome[i]
            cep_to = chromosome[i + 1]
            day = days[i]
            hour = hours[i]
            speed = speeds[i]
            
            if day < 1 or day > 7 or hour < 6 or hour > 18:
                invalid = True
                break
            
            # Calculate departure time in total seconds
            departure_seconds = (day - 1) * 86400 + hour * 3600  # 86400 seconds per day
            
            if departure_seconds < current_time_seconds:
                invalid = True  # Can't go back in time
                break
            
            # If battery low, recharge
            autonomy = self.wind_calc.calculate_autonomy(speed)
            if battery < autonomy * 0.1:  # Threshold for recharge
                recharge_cost = RECHARGE_COST_AFTER_17 if hour >= 17 else RECHARGE_COST_BASE
                total_cost += recharge_cost
                battery = autonomy
                total_cost += STOP_TIME_SECONDS
            
            # Get wind
            wind = self.data_loader.vento[day].get(hour, {'vel': 0, 'dir': 0})
            wind_vel = wind['vel']
            wind_dir = wind['dir']
            
            # Distance and bearing
            dist = self.distance_matrix[(cep_from, cep_to)]
            lat1, lon1 = self.data_loader.coordinates[cep_from]
            lat2, lon2 = self.data_loader.coordinates[cep_to]
            bearing = self.wind_calc.calculate_bearing(lat1, lon1, lat2, lon2)
            
            # Effective speed
            v_eff = self.wind_calc.calculate_effective_speed(speed, wind_vel, wind_dir, bearing)
            
            # Time to fly
            time_fly = math.ceil(dist / (v_eff / 3600))  # seconds
            
            # Check battery
            if battery < time_fly:
                recharge_cost = RECHARGE_COST_AFTER_17 if hour >= 17 else RECHARGE_COST_BASE
                total_cost += recharge_cost
                battery = autonomy
                total_cost += STOP_TIME_SECONDS
            
            # Consume battery and time
            battery -= time_fly
            total_cost += time_fly + STOP_TIME_SECONDS
            current_time_seconds = departure_seconds + time_fly + STOP_TIME_SECONDS
        
        if invalid or chromosome[0] != self.start_id or chromosome[-1] != self.start_id or len(set(chromosome[1:-1])) != len(chromosome) - 2:
            total_cost += PENALTY_INVALID
        
        # Normalize fitness: 1 / (1 + total_cost)
        fitness = 1 / (1 + total_cost)
        return fitness

class GeneticAlgorithm:
    def __init__(self, data_loader: DataLoader, fitness_evaluator: FitnessEvaluator, num_ceps: int):
        self.data_loader = data_loader
        self.fitness_evaluator = fitness_evaluator
        self.num_ceps = num_ceps
        self.start_id = data_loader.cep_to_id[START_CEP]
        self.population = []

    def initialize_population(self):
        for _ in range(POPULATION_SIZE):
            # Order: start_id + random permutation of others + start_id
            others = [i for i in range(1, self.num_ceps + 1) if i != self.start_id]
            random.shuffle(others)
            order = [self.start_id] + others + [self.start_id]
            days = [random.randint(1, 7) for _ in range(len(order) - 1)]
            hours = [random.choice([6, 9, 12, 15, 18]) for _ in range(len(order) - 1)]
            speeds = [random.choice(VALID_SPEEDS) for _ in range(len(order) - 1)]
            self.population.append((order, days, hours, speeds))

    def select_parents(self) -> List[Tuple]:
        # Tournament selection
        selected = []
        for _ in range(2):
            candidates = random.sample(self.population, 3)
            best = max(candidates, key=lambda x: self.fitness_evaluator.evaluate(*x))
            selected.append(best)
        return selected

    def pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        # For order
        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size
        start, end = sorted(random.sample(range(1, size - 1), 2))  # Avoid start and end
        
        # Copy segment
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Mapping
        def fill_child(child, parent, start, end):
            for i in range(size):
                if child[i] == -1:
                    val = parent[i]
                    while val in child[start:end]:
                        pos = child[start:end].index(val) + start
                        val = parent[pos]
                    child[i] = val
        
        fill_child(child1, parent2, start, end)
        fill_child(child2, parent1, start, end)
        return child1, child2

    def crossover(self, parent1: Tuple, parent2: Tuple) -> Tuple[Tuple, Tuple]:
        order1, days1, hours1, speeds1 = parent1
        order2, days2, hours2, speeds2 = parent2
        
        # Crossover order with PMX
        new_order1, new_order2 = self.pmx_crossover(order1, order2)
        
        # Single point crossover for others
        point = random.randint(1, len(days1) - 1)
        new_days1 = days1[:point] + days2[point:]
        new_days2 = days2[:point] + days1[point:]
        new_hours1 = hours1[:point] + hours2[point:]
        new_hours2 = hours2[:point] + hours1[point:]
        new_speeds1 = speeds1[:point] + speeds2[point:]
        new_speeds2 = speeds2[:point] + speeds1[point:]
        
        return (new_order1, new_days1, new_hours1, new_speeds1), (new_order2, new_days2, new_hours2, new_speeds2)

    def mutate(self, individual: Tuple) -> Tuple:
        order, days, hours, speeds = individual
        if random.random() < MUTATION_RATE:
            # Swap two in order
            i, j = random.sample(range(1, len(order) - 1), 2)
            order[i], order[j] = order[j], order[i]
        # Mutate days, hours, speeds
        for i in range(len(days)):
            if random.random() < MUTATION_RATE:
                days[i] = random.randint(1, 7)
            if random.random() < MUTATION_RATE:
                hours[i] = random.choice([6, 9, 12, 15, 18])
            if random.random() < MUTATION_RATE:
                speeds[i] = random.choice(VALID_SPEEDS)
        return (order, days, hours, speeds)

    def run(self) -> Tuple:
        self.initialize_population()
        for gen in range(GENERATIONS):
            new_population = []
            # Elitism
            sorted_pop = sorted(self.population, key=lambda x: self.fitness_evaluator.evaluate(*x), reverse=True)
            elite_count = int(POPULATION_SIZE * ELITISM_RATE)
            new_population.extend(sorted_pop[:elite_count])
            
            while len(new_population) < POPULATION_SIZE:
                parents = self.select_parents()
                child1, child2 = self.crossover(parents[0], parents[1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            self.population = new_population[:POPULATION_SIZE]
        
        best = max(self.population, key=lambda x: self.fitness_evaluator.evaluate(*x))
        return best

def main():
    data_loader = DataLoader('coordenadas.csv', 'vento.csv')
    distance_calc = DistanceCalculator()
    distance_matrix = distance_calc.get_distance_matrix(data_loader.coordinates)
    fitness_eval = FitnessEvaluator(data_loader, distance_matrix)
    num_ceps = len(data_loader.coordinates)
    ga = GeneticAlgorithm(data_loader, fitness_eval, num_ceps)
    best_order, best_days, best_hours, best_speeds = ga.run()
    
    # Output CSV
    with open('drone_route.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CEP_inicial', 'Latitude_inicial', 'Longitude_inicial', 'Dia_do_voo', 'Hora_inicial', 'Velocidade', 'CEP_final', 'Latitude_final', 'Longitude_final', 'Pouso', 'Hora_final'])
        
        current_time_seconds = 0
        for i in range(len(best_order) - 1):
            cep_from = data_loader.id_to_cep[best_order[i]]
            cep_to = data_loader.id_to_cep[best_order[i + 1]]
            lat1, lon1 = data_loader.coordinates[best_order[i]]
            lat2, lon2 = data_loader.coordinates[best_order[i + 1]]
            day = best_days[i]
            hour = best_hours[i]
            speed = best_speeds[i]
            pouso = 'SIM' if i < len(best_order) - 2 else 'NÃO'
            
            # Calculate time
            dist = distance_matrix[(best_order[i], best_order[i + 1])]
            wind = data_loader.vento[day].get(hour, {'vel': 0, 'dir': 0})
            bearing = fitness_eval.wind_calc.calculate_bearing(lat1, lon1, lat2, lon2)
            v_eff = fitness_eval.wind_calc.calculate_effective_speed(speed, wind['vel'], wind['dir'], bearing)
            time_fly = math.ceil(dist / (v_eff / 3600))
            final_time_seconds = current_time_seconds + time_fly + STOP_TIME_SECONDS
            final_hour = (final_time_seconds // 3600) % 24
            final_min = (final_time_seconds % 3600) // 60
            writer.writerow([cep_from, lat1, lon1, day, f"{hour}:00:00", speed, cep_to, lat2, lon2, pouso, f"{final_hour:02d}:{final_min:02d}:00"])
            current_time_seconds = final_time_seconds

if __name__ == "__main__":
    main()