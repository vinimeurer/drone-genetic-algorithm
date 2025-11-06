import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DAY_SECONDS = 24 * 3600
START_WINDOW = 6 * 3600
END_WINDOW = 19 * 3600
PHOTO_TIME = 72
RECHARGE_TIME = 72
PENALTY_SEVERE = 10_000_000
ALLOWED_SPEEDS = list(range(36, 100, 4))


@dataclass(frozen=True)
class CEPRecord:
    id: int
    cep: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class WindRecord:
    velocity_kmh: float
    direction_deg: float


@dataclass
class FlightSegment:
    cep_inicial: str
    latitude_inicial: float
    longitude_inicial: float
    dia_do_voo: int
    hora_inicial: str
    velocidade: int
    cep_final: str
    latitude_final: float
    longitude_final: float
    pouso: bool
    hora_final: str


@dataclass
class SimulationResult:
    total_time_seconds: float
    total_monetary_cost: float
    penalty_cost: float
    total_cost: float
    fitness: float
    valid: bool
    total_recargas: int
    segments: List[FlightSegment] = field(default_factory=list)


@dataclass
class Chromosome:
    order: List[int]
    days: List[int]
    minutes: List[int]
    speeds: List[int]
    fitness: float = 0.0
    result: Optional[SimulationResult] = None

    def clone(self) -> "Chromosome":
        return Chromosome(
            order=self.order[:],
            days=self.days[:],
            minutes=self.minutes[:],
            speeds=self.speeds[:],
            fitness=self.fitness,
            result=self.result,
        )


class CEPDataLoader:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def load(self) -> List[CEPRecord]:
        records: List[CEPRecord] = []
        with self.filepath.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            idx = 1
            for row in reader:
                cep = row["cep"].strip()
                longitude = float(row["longitude"])
                latitude = float(row["latitude"])
                records.append(CEPRecord(id=idx, cep=cep, latitude=latitude, longitude=longitude))
                idx += 1
        return records


class WindDataLoader:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def load(self) -> Dict[int, Dict[int, WindRecord]]:
        wind: Dict[int, Dict[int, WindRecord]] = {}
        with self.filepath.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                day = int(row["dia"])
                hour = int(row["hora"])
                vel = float(row["vel_kmh"])
                direction = float(row["direcao_deg"])
                wind.setdefault(day, {})[hour] = WindRecord(velocity_kmh=vel, direction_deg=direction)
        return wind


class DistanceCalculator:
    def __init__(self, records: List[CEPRecord]) -> None:
        self.records_map: Dict[int, CEPRecord] = {rec.id: rec for rec in records}
        self.count = len(records)
        self.lat_rad: Dict[int, float] = {rec.id: math.radians(rec.latitude) for rec in records}
        self.lon_rad: Dict[int, float] = {rec.id: math.radians(rec.longitude) for rec in records}
        self.dist_matrix: List[List[float]] = [
            [0.0 for _ in range(self.count + 1)] for _ in range(self.count + 1)
        ]
        self.bearing_matrix: List[List[float]] = [
            [0.0 for _ in range(self.count + 1)] for _ in range(self.count + 1)
        ]
        self._precompute()

    def _precompute(self) -> None:
        for id1 in range(1, self.count + 1):
            for id2 in range(id1, self.count + 1):
                if id1 == id2:
                    distance = 0.0
                    bearing = 0.0
                else:
                    distance = self._haversine(id1, id2)
                    bearing = self._bearing(id1, id2)
                self.dist_matrix[id1][id2] = distance
                self.dist_matrix[id2][id1] = distance
                self.bearing_matrix[id1][id2] = bearing
                self.bearing_matrix[id2][id1] = (bearing + math.pi) % (2 * math.pi)

    def _haversine(self, id1: int, id2: int) -> float:
        lat1 = self.lat_rad[id1]
        lon1 = self.lon_rad[id1]
        lat2 = self.lat_rad[id2]
        lon2 = self.lon_rad[id2]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371.0 * c

    def _bearing(self, id1: int, id2: int) -> float:
        lat1 = self.lat_rad[id1]
        lat2 = self.lat_rad[id2]
        dlon = self.lon_rad[id2] - self.lon_rad[id1]
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(x, y)
        return (bearing + 2 * math.pi) % (2 * math.pi)

    def distance(self, id1: int, id2: int) -> float:
        return self.dist_matrix[id1][id2]

    def bearing(self, id1: int, id2: int) -> float:
        return self.bearing_matrix[id1][id2]

    def record(self, record_id: int) -> CEPRecord:
        return self.records_map[record_id]


class WindModel:
    def __init__(self, wind_data: Dict[int, Dict[int, WindRecord]]) -> None:
        self.wind_data = wind_data
        self.available_hours = [6, 9, 12, 15, 18]

    def get(self, day: int, seconds_in_day: int) -> WindRecord:
        if day not in self.wind_data:
            return WindRecord(velocity_kmh=0.0, direction_deg=0.0)
        hours_map = self.wind_data[day]
        hour = seconds_in_day // 3600
        chosen_hour = self.available_hours[0]
        for ref_hour in self.available_hours:
            if hour >= ref_hour:
                chosen_hour = ref_hour
            else:
                break
        if chosen_hour not in hours_map:
            return WindRecord(velocity_kmh=0.0, direction_deg=0.0)
        return hours_map[chosen_hour]


class AutonomyModel:
    def __init__(self) -> None:
        self.base_capacity = 5000 * 0.93  # 4650 seconds

    def energy_capacity(self) -> float:
        return self.base_capacity

    def energy_consumption(self, time_seconds: float, speed_kmh: int) -> float:
        rate = (speed_kmh / 36.0) ** 2
        return time_seconds * rate


class BatteryModel:
    def __init__(self, autonomy_model: AutonomyModel) -> None:
        self.autonomy_model = autonomy_model
        self.capacity = autonomy_model.energy_capacity()
        self.level = self.capacity

    def energy_needed(self, flight_time: float, speed: int) -> float:
        return self.autonomy_model.energy_consumption(flight_time, speed)

    def has_enough(self, flight_time: float, speed: int) -> bool:
        return self.level >= self.energy_needed(flight_time, speed)

    def consume(self, flight_time: float, speed: int) -> None:
        self.level -= self.energy_needed(flight_time, speed)

    def recharge(self) -> None:
        self.level = self.capacity


class WindEffectCalculator:
    def __init__(self, distance_calculator: DistanceCalculator) -> None:
        self.distance_calculator = distance_calculator

    def effective_speed(
        self,
        start_id: int,
        end_id: int,
        commanded_speed: int,
        wind_record: WindRecord,
    ) -> float:
        bearing = self.distance_calculator.bearing(start_id, end_id)
        drone_vx = commanded_speed * math.sin(bearing)
        drone_vy = commanded_speed * math.cos(bearing)

        wind_direction_from = math.radians(wind_record.direction_deg)
        wind_flow_angle = (wind_direction_from + math.pi) % (2 * math.pi)
        wind_vx = wind_record.velocity_kmh * math.sin(wind_flow_angle)
        wind_vy = wind_record.velocity_kmh * math.cos(wind_flow_angle)

        ground_vx = drone_vx + wind_vx
        ground_vy = drone_vy + wind_vy
        ground_speed = math.hypot(ground_vx, ground_vy)
        return max(ground_speed, 1.0)


class FlightSimulator:
    def __init__(
        self,
        distance_calculator: DistanceCalculator,
        wind_model: WindModel,
        wind_effect_calculator: WindEffectCalculator,
        autonomy_model: AutonomyModel,
        base_id: int,
        allowed_speeds: List[int],
    ) -> None:
        self.distance_calculator = distance_calculator
        self.wind_model = wind_model
        self.wind_effect_calculator = wind_effect_calculator
        self.autonomy_model = autonomy_model
        self.base_id = base_id
        self.allowed_speeds = allowed_speeds
        self.max_minutes = (END_WINDOW - START_WINDOW) // 60 - 1

    def simulate(self, chromosome: Chromosome) -> SimulationResult:
        route = [self.base_id] + chromosome.order + [self.base_id]
        legs = len(route) - 1
        battery = BatteryModel(self.autonomy_model)
        ready_time = START_WINDOW
        absolute_time = ready_time
        total_time = 0.0
        monetary_cost = 0.0
        penalty_cost = 0.0
        total_recargas = 0
        segments: List[FlightSegment] = []

        if len(chromosome.days) != legs or len(chromosome.minutes) != legs or len(chromosome.speeds) != legs:
            penalty_cost += PENALTY_SEVERE
            total_cost = total_time + monetary_cost + penalty_cost
            fitness = 1.0 / (1.0 + total_cost)
            return SimulationResult(
                total_time_seconds=total_time,
                total_monetary_cost=monetary_cost,
                penalty_cost=penalty_cost,
                total_cost=total_cost,
                fitness=fitness,
                valid=False,
                total_recargas=total_recargas,
            )

        for idx in range(legs):
            start_id = route[idx]
            end_id = route[idx + 1]
            distance_km = self.distance_calculator.distance(start_id, end_id)
            gene_day = max(1, min(7, chromosome.days[idx]))
            gene_minutes = max(0, min(self.max_minutes, chromosome.minutes[idx]))
            desired_absolute = (gene_day - 1) * DAY_SECONDS + START_WINDOW + gene_minutes * 60
            absolute_time = max(absolute_time, desired_absolute)
            absolute_time = self._repair_to_window(absolute_time)

            commanded_speed = self._clamp_speed(chromosome.speeds[idx])

            success = False
            recharge_used = False
            attempts = 0
            actual_start = absolute_time
            actual_flight_time = 0.0
            actual_arrival = absolute_time

            while not success:
                attempts += 1
                if attempts > 100:
                    penalty_cost += PENALTY_SEVERE
                    break
                start_day = actual_start // DAY_SECONDS + 1
                if start_day > 7:
                    penalty_cost += PENALTY_SEVERE
                    break

                seconds_in_day = actual_start % DAY_SECONDS
                if seconds_in_day < START_WINDOW or seconds_in_day > END_WINDOW:
                    actual_start = self._repair_to_window(actual_start)
                    continue

                wind_record = self.wind_model.get(start_day, seconds_in_day)
                effective_speed = self.wind_effect_calculator.effective_speed(
                    start_id=start_id,
                    end_id=end_id,
                    commanded_speed=commanded_speed,
                    wind_record=wind_record,
                )
                if effective_speed <= 0.0:
                    penalty_cost += PENALTY_SEVERE
                    break

                if effective_speed < 4.0:
                    penalty_cost += PENALTY_SEVERE / 10
                    effective_speed = max(effective_speed, 4.0)

                if distance_km == 0.0:
                    actual_flight_time = 0.0
                else:
                    actual_flight_time = math.ceil(distance_km / (effective_speed / 3600.0))

                energy_needed = battery.energy_needed(actual_flight_time, commanded_speed)
                if energy_needed > battery.level + 1e-6:
                    recharge_used = True
                    recharge_cost = 80.0
                    if seconds_in_day >= 17 * 3600:
                        recharge_cost += 80.0
                    monetary_cost += recharge_cost
                    total_recargas += 1
                    battery.recharge()
                    actual_start += RECHARGE_TIME
                    continue

                actual_arrival = actual_start + actual_flight_time
                arrival_day = actual_arrival // DAY_SECONDS + 1
                if arrival_day > 7:
                    penalty_cost += PENALTY_SEVERE
                    break

                arrival_seconds = actual_arrival % DAY_SECONDS
                if arrival_seconds > END_WINDOW:
                    actual_start = self._shift_to_next_day_start(start_day)
                    continue

                success = True

            if not success:
                break

            wait_time = max(0.0, actual_start - absolute_time)
            total_time += wait_time
            battery.consume(actual_flight_time, commanded_speed)
            if battery.level < -1e-6:
                penalty_cost += PENALTY_SEVERE
                break

            total_time += actual_flight_time
            absolute_time = actual_arrival + PHOTO_TIME
            total_time += PHOTO_TIME

            start_day = actual_start // DAY_SECONDS + 1
            end_day = actual_arrival // DAY_SECONDS + 1
            start_record = self.distance_calculator.record(start_id)
            end_record = self.distance_calculator.record(end_id)

            segments.append(
                FlightSegment(
                    cep_inicial=start_record.cep,
                    latitude_inicial=start_record.latitude,
                    longitude_inicial=start_record.longitude,
                    dia_do_voo=int(start_day),
                    hora_inicial=self._format_time(actual_start % DAY_SECONDS),
                    velocidade=commanded_speed,
                    cep_final=end_record.cep,
                    latitude_final=end_record.latitude,
                    longitude_final=end_record.longitude,
                    pouso=recharge_used,
                    hora_final=self._format_time(actual_arrival % DAY_SECONDS),
                )
            )

        valid = penalty_cost == 0.0 and len(segments) == legs
        total_cost = total_time + monetary_cost + penalty_cost
        fitness = 1.0 / (1.0 + total_cost)
        return SimulationResult(
            total_time_seconds=total_time,
            total_monetary_cost=monetary_cost,
            penalty_cost=penalty_cost,
            total_cost=total_cost,
            fitness=fitness,
            valid=valid,
            total_recargas=total_recargas,
            segments=segments if valid else [],
        )

    def _repair_to_window(self, absolute_time: float) -> float:
        day = int(absolute_time // DAY_SECONDS)
        seconds_in_day = absolute_time % DAY_SECONDS
        if seconds_in_day < START_WINDOW:
            return day * DAY_SECONDS + START_WINDOW
        if seconds_in_day > END_WINDOW:
            return (day + 1) * DAY_SECONDS + START_WINDOW
        return absolute_time

    def _shift_to_next_day_start(self, current_day: int) -> float:
        next_day = current_day + 1
        return (next_day - 1) * DAY_SECONDS + START_WINDOW

    def _clamp_speed(self, value: int) -> int:
        return min(self.allowed_speeds, key=lambda s: abs(s - value))

    @staticmethod
    def _format_time(seconds_in_day: float) -> str:
        seconds = int(round(seconds_in_day))
        seconds = max(0, min(DAY_SECONDS - 1, seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class GeneticAlgorithm:
    def __init__(
        self,
        simulator: FlightSimulator,
        base_id: int,
        location_ids: List[int],
        allowed_speeds: List[int],
        population_size: int = 80,
        generations: int = 150,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        order_mutation_rate: float = 0.05,
        attribute_mutation_rate: float = 0.05,
        elitism_fraction: float = 0.1,
    ) -> None:
        self.simulator = simulator
        self.base_id = base_id
        self.location_ids = [loc for loc in location_ids if loc != base_id]
        self.allowed_speeds = allowed_speeds
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.order_mutation_rate = order_mutation_rate
        self.attribute_mutation_rate = attribute_mutation_rate
        self.elitism_count = max(1, int(population_size * elitism_fraction))
        self.legs_count = len(self.location_ids) + 1

    def run(self) -> Chromosome:
        population = self._initialize_population()
        best_chromosome: Optional[Chromosome] = None

        for _ in range(self.generations):
            self._evaluate_population(population)
            population.sort(key=lambda c: c.fitness, reverse=True)

            if best_chromosome is None or population[0].fitness > best_chromosome.fitness:
                best_chromosome = population[0].clone()
                best_chromosome.result = population[0].result

            new_population: List[Chromosome] = [population[i].clone() for i in range(self.elitism_count)]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()
                self._mutate(child1)
                self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        if best_chromosome is None:
            best_chromosome = population[0]
        if best_chromosome.result is None:
            best_chromosome.result = self.simulator.simulate(best_chromosome)
        return best_chromosome

    def _initialize_population(self) -> List[Chromosome]:
        population: List[Chromosome] = []
        for _ in range(self.population_size):
            order = random.sample(self.location_ids, len(self.location_ids))
            days = [random.randint(1, 7) for _ in range(self.legs_count)]
            minutes = [random.randint(0, self.simulator.max_minutes) for _ in range(self.legs_count)]
            speeds = [random.choice(self.allowed_speeds) for _ in range(self.legs_count)]
            population.append(Chromosome(order=order, days=days, minutes=minutes, speeds=speeds))
        return population

    def _evaluate_population(self, population: List[Chromosome]) -> None:
        for chrom in population:
            result = self.simulator.simulate(chrom)
            chrom.fitness = result.fitness
            chrom.result = result

    def _tournament_select(self, population: List[Chromosome]) -> Chromosome:
        contenders = random.sample(population, self.tournament_size)
        contenders.sort(key=lambda c: c.fitness, reverse=True)
        return contenders[0]

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        child_order1, child_order2 = self._pmx(parent1.order, parent2.order)
        point = random.randint(1, self.legs_count - 1)
        child_days1 = parent1.days[:point] + parent2.days[point:]
        child_days2 = parent2.days[:point] + parent1.days[point:]
        child_minutes1 = parent1.minutes[:point] + parent2.minutes[point:]
        child_minutes2 = parent2.minutes[:point] + parent1.minutes[point:]
        child_speeds1 = parent1.speeds[:point] + parent2.speeds[point:]
        child_speeds2 = parent2.speeds[:point] + parent1.speeds[point:]
        return (
            Chromosome(order=child_order1, days=child_days1, minutes=child_minutes1, speeds=child_speeds1),
            Chromosome(order=child_order2, days=child_days2, minutes=child_minutes2, speeds=child_speeds2),
        )

    def _pmx(self, order1: List[int], order2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(order1)
        cx_point1 = random.randint(0, size - 2)
        cx_point2 = random.randint(cx_point1 + 1, size - 1)
        child1 = [None] * size
        child2 = [None] * size

        child1[cx_point1:cx_point2] = order1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = order2[cx_point1:cx_point2]

        def fill(child: List[Optional[int]], parent_x: List[int], parent_y: List[int]) -> None:
            for idx in range(size):
                if child[idx] is None:
                    candidate = parent_y[idx]
                    while candidate in child:
                        candidate = parent_y[parent_x.index(candidate)]
                    child[idx] = candidate

        fill(child1, order1, order2)
        fill(child2, order2, order1)
        return child1, child2  # type: ignore

    def _mutate(self, chrom: Chromosome) -> None:
        if random.random() < self.order_mutation_rate:
            idx1, idx2 = random.sample(range(len(chrom.order)), 2)
            chrom.order[idx1], chrom.order[idx2] = chrom.order[idx2], chrom.order[idx1]

        for i in range(self.legs_count):
            if random.random() < self.attribute_mutation_rate:
                chrom.days[i] = random.randint(1, 7)
            if random.random() < self.attribute_mutation_rate:
                chrom.minutes[i] = random.randint(0, self.simulator.max_minutes)
            if random.random() < self.attribute_mutation_rate:
                chrom.speeds[i] = random.choice(self.allowed_speeds)


class OutputWriter:
    @staticmethod
    def write(filepath: Path, segments: List[FlightSegment]) -> None:
        header = [
            "CEP_inicial",
            "Latitude_inicial",
            "Longitude_inicial",
            "Dia_do_voo",
            "Hora_inicial",
            "Velocidade",
            "CEP_final",
            "Latitude_final",
            "Longitude_final",
            "Pouso",
            "Hora_final",
        ]
        with filepath.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            for segment in segments:
                writer.writerow(
                    [
                        segment.cep_inicial,
                        f"{segment.latitude_inicial:.12f}",
                        f"{segment.longitude_inicial:.12f}",
                        segment.dia_do_voo,
                        segment.hora_inicial,
                        segment.velocidade,
                        segment.cep_final,
                        f"{segment.latitude_final:.12f}",
                        f"{segment.longitude_final:.12f}",
                        "SIM" if segment.pouso else "NAO",
                        segment.hora_final,
                    ]
                )


def main() -> None:
    random.seed(42)
    base_path = Path(__file__).resolve().parent
    coordenadas_path = base_path / "coordenadas.csv"
    vento_path = base_path / "vento.csv"

    cep_records = CEPDataLoader(coordenadas_path).load()
    wind_data = WindDataLoader(vento_path).load()

    base_cep = "82821020"
    base_record = next((rec for rec in cep_records if rec.cep == base_cep), None)
    if base_record is None:
        raise ValueError("CEP base 82821020 não encontrado.")

    distance_calculator = DistanceCalculator(cep_records)
    wind_model = WindModel(wind_data)
    autonomy_model = AutonomyModel()
    wind_effect_calculator = WindEffectCalculator(distance_calculator)
    simulator = FlightSimulator(
        distance_calculator=distance_calculator,
        wind_model=wind_model,
        wind_effect_calculator=wind_effect_calculator,
        autonomy_model=autonomy_model,
        base_id=base_record.id,
        allowed_speeds=ALLOWED_SPEEDS,
    )

    ga = GeneticAlgorithm(
        simulator=simulator,
        base_id=base_record.id,
        location_ids=[rec.id for rec in cep_records],
        allowed_speeds=ALLOWED_SPEEDS,
        population_size=80,
        generations=200,
        tournament_size=3,
        crossover_rate=0.9,
        order_mutation_rate=0.05,
        attribute_mutation_rate=0.05,
        elitism_fraction=0.1,
    )

    best_chromosome = ga.run()
    best_result = best_chromosome.result
    if best_result is None or not best_result.valid:
        best_result = simulator.simulate(best_chromosome)

    if best_result.valid:
        output_path = base_path / "rota_otimizada.csv"
        OutputWriter.write(output_path, best_result.segments)


if __name__ == "__main__":
    main()