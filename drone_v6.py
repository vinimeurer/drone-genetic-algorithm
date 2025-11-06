"""
Drone route optimizer using a Genetic Algorithm.

Reads:
 - coordenadas.csv with columns: cep,lat,lon  (header optional)
 - vento.csv with columns: day,hour,speed,dir,unit? (day: 0..6 or 1..7), hour: 0..23
Outputs:
 - best_route.csv with one row per leg:
   CEP inicial, Latitude inicial, Longitude inicial, Dia do voo, Hora inicial, Velocidade,
   CEP final, Latitude final, Longitude final, Pouso (SIM/NÃO), Hora final

Assumptions / notes:
 - Unibrasil (start/finish) is the CEP equal to '82821020' if present; otherwise first row is start.
 - Wind direction is meteorological (direction FROM which wind blows). We convert to vector blowing TO = dir + 180.
 - Wind speed in vento.csv assumed km/h; if values are tiny (<10) it's possibly in knots, then converted (1 knot = 1.852 km/h).
 - Drone allowed speeds: multiples of 4 km/h from 36 to 96 inclusive.
 - Base autonomy: 5000 seconds; corrected by factor 0.93 => base_capacity_seconds = 5000*0.93
 - For v > 36 km/h autonomy A(v) = 5000 * (36/v)^2 * 0.93; for v <=36 autonomy = 5000*0.93
 - Energy accounting: "energy" is normalized so starting battery = 1. For a segment with time t and speed v, energy consumed = t / A(v).
 - Each stop (photo or recharge) consumes 72 seconds.
 - Landing (recharge) costs R$80; additional R$80 if landing after 17:00 (>=17:00 local day).
 - Flights can only be between 06:00 and 19:00. If a planned departure would end after 19:00, simulation will shift it to next valid day/time.
 - If visits exceed 7 days or other hard constraints violated, heavy penalty applied to fitness.

This script implements a GA with:
 - permutation encoding for visit order (start/end fixed)
 - per-leg genes for day (0..6), hour (6..18), speed (index into allowed speeds)
 - PMX crossover for permutation, uniform crossover for auxiliary genes
 - mutation: swap/inversion for permutation, random reset for other genes

Run:
 python3 drone_ga.py

"""

import csv
import math
import random
import copy
import os
import sys
from datetime import timedelta, datetime

# ----- Configuration -----
COORDINATES_CSV = "coordenadas.csv"
WIND_CSV = "vento.csv"
OUTPUT_CSV = "best_route.csv"

POPULATION_SIZE = 80
GENERATIONS = 300
TOURNAMENT_SIZE = 3
ELITISM = 2
MUTATION_RATE = 0.06
P_CROSSOVER = 0.9

# Drone constants
BASE_AUTONOMY_S = 5000.0
CORRECTION_FACTOR = 0.93
BASE_AUTONOMY_S_CORR = BASE_AUTONOMY_S * CORRECTION_FACTOR  # seconds
PHOTO_OR_LAND_S = 72  # seconds per stop (photo or recharge)
MIN_SPEED_KMH = 36  # 10 m/s -> 36 km/h
MAX_SPEED_KMH = 96
SPEED_STEP = 4
ALLOWED_SPEEDS = list(range(MIN_SPEED_KMH, MAX_SPEED_KMH + 1, SPEED_STEP))

EARTH_R_KM = 6371.0

# Time window
DAY_START_H = 6
DAY_END_H = 19  # flights must be within [06:00, 19:00]

# Penalties
BIG_PENALTY = 1e9

# Costs
LANDING_COST_R = 80.0
EXTRA_LANDING_AFTER_17_R = 80.0

# Random seed for reproducibility (optional)
RANDOM_SEED = None
# --------------------------


def read_coordinates(path):
    """
    Reads coordenadas.csv expecting at least columns: cep,lat,lon (accepts many header variants).
    Returns list of dicts: [{'cep':str,'lat':float,'lon':float}, ...]
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        # try DictReader and flexible header matching
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        names = {n.lower(): n for n in fieldnames if n}
        def find_key(cands):
            for c in cands:
                if c in names:
                    return names[c]
            for k in names:
                for c in cands:
                    if c in k:
                        return names[k]
            return None
        cep_k = find_key(['cep','zip'])
        lat_k = find_key(['lat','latitude','latitude_deg'])
        lon_k = find_key(['lon','longitude','long'])
        if cep_k and lat_k and lon_k:
            for r in reader:
                try:
                    cep = r[cep_k].strip()
                    lat = float(r[lat_k])
                    lon = float(r[lon_k])
                    rows.append({'cep': cep, 'lat': lat, 'lon': lon})
                except Exception:
                    continue
            return rows
        # fallback: positional parsing (robust to orders like cep,longitude,latitude)
        f.seek(0)
        r2 = csv.reader(f)
        for r in r2:
            if not r: continue
            if len(r) < 3: continue
            cep = r[0].strip()
            # try common orders: cep,lat,lon OR cep,lon,lat
            lat = lon = None
            try:
                lat = float(r[1]); lon = float(r[2])
            except Exception:
                try:
                    lat = float(r[2]); lon = float(r[1])
                except Exception:
                    continue
            rows.append({'cep': cep, 'lat': lat, 'lon': lon})
    return rows


def read_wind(path):
    """
    Reads vento.csv accepting Portuguese/English headers. Produces wind[day][hour] dict filled for all hours 0..23.
    """
    wind = {d: {} for d in range(7)}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        names = {n.lower(): n for n in fieldnames if n}
        def find_key(cands):
            for c in cands:
                if c in names:
                    return names[c]
            for k in names:
                for c in cands:
                    if c in k:
                        return names[k]
            return None
        day_k = find_key(['day','dia'])
        hour_k = find_key(['hour','hora','hora_local'])
        speed_k = find_key(['speed','vel','vel_kmh','vel_km_h','velocidade','vel_km/h'])
        dir_k = find_key(['dir','direcao','direcao_deg','direcao_grau','direcao_deg','dir_deg'])
        # if keys not found, fallback to positional parsing below
        if day_k and hour_k and speed_k and dir_k:
            for r in reader:
                try:
                    raw_day = r[day_k]
                    raw_hour = r[hour_k]
                    raw_speed = r[speed_k]
                    raw_dir = r[dir_k]
                except Exception:
                    continue
                try:
                    day = int(float(raw_day))
                except Exception:
                    day = 0
                # normalize to 0..6
                if 1 <= day <= 7:
                    day_idx = day - 1
                else:
                    try:
                        day_idx = max(0, min(6, int(day)))
                    except Exception:
                        day_idx = 0
                hour = int(float(raw_hour)) % 24
                try:
                    speed = float(raw_speed)
                except Exception:
                    speed = 0.0
                # heuristic: if speeds are small (<12) consider knots -> convert
                if speed < 12:
                    speed_kmh = speed * 1.852
                else:
                    speed_kmh = speed
                try:
                    dir_deg = float(raw_dir) % 360.0
                except Exception:
                    dir_deg = 0.0
                wind[day_idx][hour] = {'speed_kmh': speed_kmh, 'dir_deg': dir_deg}
        else:
            # positional fallback: expect CSV columns day,hour,speed,dir (or close)
            f.seek(0)
            pr = csv.reader(f)
            for r in pr:
                if not r: continue
                if len(r) < 4: continue
                try:
                    day = int(float(r[0]))
                except:
                    day = 0
                if 1 <= day <= 7:
                    day_idx = day - 1
                else:
                    day_idx = max(0, min(6, int(day)))
                hour = int(float(r[1])) % 24
                speed = float(r[2])
                if speed < 12:
                    speed_kmh = speed * 1.852
                else:
                    speed_kmh = speed
                dir_deg = float(r[3]) % 360.0
                wind[day_idx][hour] = {'speed_kmh': speed_kmh, 'dir_deg': dir_deg}
    # Fill missing hours/days by nearest available (simple forward/backward fill)
    for d in range(7):
        # if day empty, try to copy from any other day
        if not wind[d]:
            for dd in range(7):
                if wind[dd]:
                    wind[d] = wind[dd].copy()
                    break
    for d in range(7):
        # ensure all 24 hours present by nearest-hour fill
        available_hours = sorted(wind[d].keys())
        if not available_hours:
            # as a last resort, fill zeros
            for h in range(24):
                wind[d][h] = {'speed_kmh': 0.0, 'dir_deg': 0.0}
            continue
        for h in range(24):
            if h not in wind[d]:
                # find nearest available hour
                nearest = min(available_hours, key=lambda ah: abs(ah - h))
                wind[d][h] = wind[d][nearest]
    return wind

# ----- geographic utilities -----
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R_KM * c


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point1 to point2 in degrees (0..360) where 0 = north"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    theta = math.atan2(x, y)
    bearing = (math.degrees(theta) + 360.0) % 360.0
    return bearing


def vector_from_bearing_deg(speed_kmh, bearing_deg):
    """Return (vx, vy) components with x=east, y=north in km/h"""
    rad = math.radians(bearing_deg)
    vx = speed_kmh * math.sin(rad)  # east component
    vy = speed_kmh * math.cos(rad)  # north component
    return vx, vy


# ----- energy / autonomy -----
def autonomy_seconds_for_speed(v_kmh):
    """Return available flight seconds at airspeed v_kmh according to model"""
    if v_kmh <= 36.0:
        return BASE_AUTONOMY_S_CORR
    else:
        return BASE_AUTONOMY_S * (36.0 / v_kmh) ** 2 * CORRECTION_FACTOR


# ----- genetic operators (permutation PMX etc.) -----
def pmx_crossover(parent_a, parent_b):
    """Partially Mapped Crossover for permutations (list of ints)"""
    size = len(parent_a)
    a = parent_a[:]
    b = parent_b[:]
    if size <= 2:
        return a[:], b[:]
    i, j = sorted(random.sample(range(size), 2))
    def pmx(p1, p2):
        child = [-1] * size
        # copy segment
        child[i:j+1] = p1[i:j+1]
        # map rest from p2
        for idx in range(i, j+1):
            val = p2[idx]
            if val not in child:
                pos = idx
                while True:
                    val2 = p1[pos]
                    pos = p2.index(val2)
                    if child[pos] == -1:
                        child[pos] = val
                        break
        # fill remaining
        for k in range(size):
            if child[k] == -1:
                child[k] = p2[k]
        return child
    return pmx(a, b), pmx(b, a)


def tournament_selection(pop, fitnesses, k):
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] < fitnesses[best]:
            best = i
    return copy.deepcopy(pop[best])


# ----- individual representation -----
class Individual:
    """
    Genes:
      order: list of indices for visit order excluding the fixed base at index 0.
      days: list of length n_legs (including return) day indices 0..6
      hours: list of length n_legs hour values within [6..18] (start hour)
      speeds_idx: list of indices into ALLOWED_SPEEDS for each leg
    Note: number_of_legs = number_of_points (since includes return to base)
    """
    def __init__(self, order, days, hours, speeds_idx):
        self.order = order
        self.days = days
        self.hours = hours
        self.speeds_idx = speeds_idx


# ----- simulation & fitness -----
class Simulator:
    def __init__(self, coords, winds, start_cep='82821020'):
        self.coords = coords  # list of dicts
        self.n = len(coords)
        self.winds = winds
        # find start index by cep
        self.start_idx = 0
        for i, c in enumerate(coords):
            if c['cep'] == start_cep:
                self.start_idx = i
                break
        # precompute distance and bearings
        self.dist_km = [[0.0]*self.n for _ in range(self.n)]
        self.bearing = [[0.0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                self.dist_km[i][j] = haversine_km(coords[i]['lat'], coords[i]['lon'], coords[j]['lat'], coords[j]['lon'])
                self.bearing[i][j] = bearing_deg(coords[i]['lat'], coords[i]['lon'], coords[j]['lat'], coords[j]['lon'])

    def simulate_individual(self, ind: Individual):
        """
        Simulate full route of individual.
        Returns tuple: (fitness_score, details_dict)
        details_dict includes: segments list for CSV output, total_time_s, total_recharges, total_cost_R, valid(bool)
        """
        # Build full route: start -> permuted nodes -> start
        perm = ind.order[:]
        # ensure perm contains all indices except start
        all_nodes = [i for i in range(self.n) if i != self.start_idx]
        if sorted(perm) != sorted(all_nodes):
            # invalid permutation -> heavy penalty
            return BIG_PENALTY, {'valid': False}
        route = [self.start_idx] + perm + [self.start_idx]
        legs = len(route) - 1

        # Sanity: gene lengths
        if not (len(ind.days) == legs and len(ind.hours) == legs and len(ind.speeds_idx) == legs):
            return BIG_PENALTY, {'valid': False}

        # state
        battery_energy = 1.0  # normalized
        segments = []
        total_time_s = 0.0
        total_recharges = 0
        total_cost_R = 0.0
        current_global_day = None  # track day index for crossing days as we simulate
        current_time_of_day_s = None

        # We'll interpret each leg's gene as preferred departure day/hour; but simulation may adjust forward to meet constraints.
        for leg_i in range(legs):
            a = route[leg_i]
            b = route[leg_i+1]
            gene_day = int(ind.days[leg_i])
            gene_hour = int(ind.hours[leg_i])
            # clamp gene_hour to allowed window
            if gene_hour < DAY_START_H: gene_hour = DAY_START_H
            if gene_hour >= DAY_END_H: gene_hour = DAY_END_H - 1

            # Determine departure absolute day/time in seconds-from-day-start
            depart_day = gene_day
            depart_time_s = gene_hour * 3600  # start at hour:00 precisely
            # If current simulated time is later than this, we must start after current
            if current_global_day is not None:
                # compute absolute seconds of gene and current
                curr_abs = current_global_day * 86400 + current_time_of_day_s
                gene_abs = depart_day * 86400 + depart_time_s
                if gene_abs < curr_abs:
                    # shift to next available slot >= curr_abs (keep same hour but increase day)
                    delta_days = math.ceil((curr_abs - gene_abs) / 86400.0)
                    depart_day += delta_days
            # enforce window: depart_time must be within allowed day hours; we may shift to earliest allowed (6:00) if needed
            if depart_time_s < DAY_START_H * 3600:
                depart_time_s = DAY_START_H * 3600
            if depart_time_s >= DAY_END_H * 3600:
                # shift to next day at DAY_START_H
                depart_day += 1
                depart_time_s = DAY_START_H * 3600

            # If depart_day > 6 -> exceeds 7 days
            if depart_day > 6:
                return BIG_PENALTY, {'valid': False}

            # get wind for depart day/hour (use hour integer)
            wind_hour = int(depart_time_s // 3600) % 24
            wind_day = depart_day
            wind_info = self.winds.get(wind_day, {}).get(wind_hour)
            if wind_info is None:
                wind_info = self.winds[wind_day][wind_hour]  # fallback
            wind_speed = wind_info['speed_kmh']
            wind_dir = wind_info['dir_deg']
            # convert wind from 'from' to 'to' direction
            wind_to_dir = (wind_dir + 180.0) % 360.0

            # segment distance and heading
            d_km = self.dist_km[a][b]
            heading = self.bearing[a][b]  # direction of drone movement (to)
            # drone's selected airspeed
            speed_kmh = ALLOWED_SPEEDS[ind.speeds_idx[leg_i] % len(ALLOWED_SPEEDS)]
            # drone airspeed vector (relative to air) pointing to 'heading'
            dvx, dvy = vector_from_bearing_deg(speed_kmh, heading)
            # wind vector (to)
            wvx, wvy = vector_from_bearing_deg(wind_speed, wind_to_dir)
            # ground vector
            gvx = dvx + wvx
            gvy = dvy + wvy
            ground_speed_kmh = math.hypot(gvx, gvy)
            # prevent unrealistically low ground speed
            if ground_speed_kmh < 1e-3:
                ground_speed_kmh = 0.1  # minimal
            # time to traverse (seconds)
            time_s = math.ceil((d_km / (ground_speed_kmh / 3600.0)))
            # energy required at this leg based on chosen airspeed
            A_v = autonomy_seconds_for_speed(speed_kmh)
            energy_needed = time_s / A_v

            # if battery insufficient -> schedule recharge at current location BEFORE departing
            landed_for_recharge = False
            if energy_needed > battery_energy:
                # force recharge at 'a'
                total_recharges += 1
                total_cost_R += LANDING_COST_R
                # check if current local hour >=17 then extra
                # compute local hour of recharge time = depart_time_s (we recharge before depart)
                if depart_time_s / 3600.0 >= 17.0:
                    total_cost_R += EXTRA_LANDING_AFTER_17_R
                # add recharge time
                depart_time_s += PHOTO_OR_LAND_S
                time_s_added = PHOTO_OR_LAND_S
                total_time_s += time_s_added
                battery_energy = 1.0
                landed_for_recharge = True

            # Now check again energy (should be enough)
            if energy_needed > battery_energy + 1e-9:
                # still not enough -> infeasible
                return BIG_PENALTY, {'valid': False}

            # Also verify that the flight will end before DAY_END_H (19:00). If not, shift depart to next day start.
            end_time_of_day_s = depart_time_s + time_s
            end_day = depart_day
            if end_time_of_day_s >= DAY_END_H * 3600:
                # shift depart to next day at DAY_START_H
                depart_day += 1
                if depart_day > 6:
                    return BIG_PENALTY, {'valid': False}
                depart_time_s = DAY_START_H * 3600
                # recompute wind for new slot
                wind_hour = int(depart_time_s // 3600) % 24
                wind_day = depart_day
                wind_info = self.winds.get(wind_day, {}).get(wind_hour)
                if wind_info is None:
                    wind_info = self.winds[wind_day][wind_hour]
                wind_speed = wind_info['speed_kmh']
                wind_dir = wind_info['dir_deg']
                wind_to_dir = (wind_dir + 180.0) % 360.0
                wvx, wvy = vector_from_bearing_deg(wind_speed, wind_to_dir)
                gvx = dvx + wvx
                gvy = dvy + wvy
                ground_speed_kmh = math.hypot(gvx, gvy)
                if ground_speed_kmh < 1e-3:
                    ground_speed_kmh = 0.1
                time_s = math.ceil((d_km / (ground_speed_kmh / 3600.0)))
                A_v = autonomy_seconds_for_speed(speed_kmh)
                energy_needed = time_s / A_v
                # If after shifting we still cannot fit due to battery, recharge
                if energy_needed > battery_energy:
                    total_recharges += 1
                    total_cost_R += LANDING_COST_R
                    if depart_time_s / 3600.0 >= 17.0:
                        total_cost_R += EXTRA_LANDING_AFTER_17_R
                    depart_time_s += PHOTO_OR_LAND_S
                    total_time_s += PHOTO_OR_LAND_S
                    battery_energy = 1.0

            # perform flight: consume energy, add flight time
            battery_energy -= energy_needed
            total_time_s += time_s
            # arrival: photo stop consumes PHOTO_OR_LAND_S seconds (but not a recharge unless battery 0 next leg)
            total_time_s += PHOTO_OR_LAND_S
            # record segment
            # compute arrival hour/min for output
            # absolute start time
            if current_global_day is None:
                current_global_day = depart_day
                current_time_of_day_s = depart_time_s
            else:
                current_global_day = depart_day
                current_time_of_day_s = depart_time_s
            arrival_time_of_day_s = (current_time_of_day_s + time_s) % 86400
            # update current time pointer to arrival + photo time
            current_time_of_day_s = (current_time_of_day_s + time_s + PHOTO_OR_LAND_S) % 86400

            segments.append({
                'cep_from': self.coords[a]['cep'],
                'lat_from': self.coords[a]['lat'],
                'lon_from': self.coords[a]['lon'],
                'day': depart_day,
                'hour_start_s': depart_time_s,
                'speed_kmh': speed_kmh,
                'cep_to': self.coords[b]['cep'],
                'lat_to': self.coords[b]['lat'],
                'lon_to': self.coords[b]['lon'],
                'landed_for_recharge': landed_for_recharge,
                'arrival_time_s': arrival_time_of_day_s
            })

            # if battery is extremely low (<1e-6) set to 0
            if battery_energy < 1e-12:
                battery_energy = 0.0

        # After completing route, return fitness
        # Fitness: total_time_seconds + monetary cost weighted (convert R$ to seconds-equivalent weight)
        # Use simple conversion factor: 1 R$ == 100 seconds (configurable). This merges time and money.
        MONEY_WEIGHT = 100.0
        fitness = total_time_s + total_cost_R * MONEY_WEIGHT

        details = {
            'segments': segments,
            'total_time_s': total_time_s,
            'total_recharges': total_recharges,
            'total_cost_R': total_cost_R,
            'valid': True
        }
        return fitness, details


# ----- GA main -----
def create_random_individual(n_points, start_idx):
    """
    n_points: total points including start
    start_idx: index of base to exclude from permutation
    returns Individual
    """
    nodes = [i for i in range(n_points) if i != start_idx]
    order = nodes[:]
    random.shuffle(order)
    # legs include return to start
    legs = len(order) + 1
    days = [random.randint(0, 6) for _ in range(legs)]
    hours = [random.randint(DAY_START_H, DAY_END_H - 1) for _ in range(legs)]
    speeds_idx = [random.randrange(len(ALLOWED_SPEEDS)) for _ in range(legs)]
    return Individual(order, days, hours, speeds_idx)


def crossover(parent1: Individual, parent2: Individual):
    # permutation PMX
    child_order_a, child_order_b = pmx_crossover(parent1.order, parent2.order)
    # other genes: uniform crossover
    legs = len(child_order_a) + 1
    def mix(a_list, b_list):
        res = []
        for i in range(legs):
            res.append(a_list[i] if random.random() < 0.5 else b_list[i])
        return res
    child_days = mix(parent1.days, parent2.days)
    child_hours = mix(parent1.hours, parent2.hours)
    child_speeds = mix(parent1.speeds_idx, parent2.speeds_idx)
    return Individual(child_order_a, child_days, child_hours, child_speeds), Individual(child_order_b, child_days, child_hours, child_speeds)


def mutate(ind: Individual):
    # mutate permutation: swap with some chance
    if random.random() < MUTATION_RATE:
        a, b = random.sample(range(len(ind.order)), 2)
        ind.order[a], ind.order[b] = ind.order[b], ind.order[a]
    # small inversion
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(len(ind.order)), 2))
        ind.order[i:j+1] = list(reversed(ind.order[i:j+1]))
    # mutate days/hours/speeds
    for i in range(len(ind.days)):
        if random.random() < MUTATION_RATE:
            ind.days[i] = random.randint(0, 6)
        if random.random() < MUTATION_RATE:
            ind.hours[i] = random.randint(DAY_START_H, DAY_END_H - 1)
        if random.random() < MUTATION_RATE:
            ind.speeds_idx[i] = random.randrange(len(ALLOWED_SPEEDS))
    return ind


def run_ga(coords, winds, generations=GENERATIONS, pop_size=POPULATION_SIZE):
    sim = Simulator(coords, winds)
    # init population
    pop = [create_random_individual(len(coords), sim.start_idx) for _ in range(pop_size)]
    best = None
    best_fit = float('inf')
    best_details = None

    for gen in range(generations):
        # evaluate all
        fitnesses = []
        details_list = []
        for ind in pop:
            f, d = sim.simulate_individual(ind)
            fitnesses.append(f)
            details_list.append(d)
            if f < best_fit and d.get('valid', False):
                best_fit = f
                best = copy.deepcopy(ind)
                best_details = d
        # print progress
        if gen % max(1, generations//10) == 0 or gen == generations-1:
            print(f"Gen {gen}: best fitness so far = {best_fit:.1f}")
        # next generation
        new_pop = []
        # elitism: carry top ELITISM
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
        for ei in range(min(ELITISM, len(pop))):
            new_pop.append(copy.deepcopy(pop[sorted_idx[ei]]))
        # fill rest
        while len(new_pop) < pop_size:
            if random.random() < P_CROSSOVER:
                p1 = tournament_selection(pop, fitnesses, TOURNAMENT_SIZE)
                p2 = tournament_selection(pop, fitnesses, TOURNAMENT_SIZE)
                c1, c2 = crossover(p1, p2)
                new_pop.append(mutate(c1))
                if len(new_pop) < pop_size:
                    new_pop.append(mutate(c2))
            else:
                # reproduction
                p = tournament_selection(pop, fitnesses, TOURNAMENT_SIZE)
                new_pop.append(mutate(p))
        pop = new_pop

    return best, best_fit, best_details, sim


def write_output_csv(details, sim: Simulator, out_path=OUTPUT_CSV):
    """
    Writes CSV with columns:
    CEP inicial, Latitude inicial, Longitude inicial, Dia do voo, Hora inicial, Velocidade,
    CEP final, Latitude final, Longitude final, Pouso (SIM/NÃO), Hora final
    """
    rows = []
    for seg in details['segments']:
        day = seg['day']
        hour_start = int(seg['hour_start_s'] // 3600)
        hour_final = int(seg['arrival_time_s'] // 3600)
        pouso = "SIM" if seg['landed_for_recharge'] else "NÃO"
        rows.append([
            seg['cep_from'],
            f"{seg['lat_from']:.6f}",
            f"{seg['lon_from']:.6f}",
            str(day+1),  # human-friendly 1..7
            f"{hour_start:02d}:00:00",
            f"{seg['speed_kmh']:.0f}",
            seg['cep_to'],
            f"{seg['lat_to']:.6f}",
            f"{seg['lon_to']:.6f}",
            pouso,
            f"{hour_final:02d}:00:00"
        ])
    # write CSV
    header = ["CEP inicial","Latitude inicial","Longitude inicial","Dia do voo","Hora inicial","Velocidade",
              "CEP final","Latitude final","Longitude final","Pouso (SIM/NÃO)","Hora final"]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote output CSV: {out_path}")


# def main():
#     if RANDOM_SEED is not None:
#         random.seed(RANDOM_SEED)
#     # check files
#     if not os.path.exists(COORDINATES_CSV):
#         print(f"Missing {COORDINATES_CSV} in current directory. Place coordenadas.csv here.")
#         sys.exit(1)
#     if not os.path.exists(WIND_CSV):
#         print(f"Missing {WIND_CSV} in current directory. Place vento.csv here.")
#         sys.exit(1)
#     coords = read_coordinates(COORDINATES_CSV)
#     winds = read_wind(WIND_CSV)
#     print(f"Loaded {len(coords)} coordinates and wind table for 7 days.")
#     best_ind, best_fit, best_details, sim = run_ga(coords, winds)
#     if best_ind is None or best_details is None:
#         print("No valid solution found.")
#         sys.exit(2)
#     print(f"Best fitness: {best_fit:.2f}")
#     print(f"Total time (s): {best_details['total_time_s']:.1f}, recharges: {best_details['total_recharges']}, cost R$: {best_details['total_cost_R']:.2f}")
#     write_output_csv(best_details, sim, OUTPUT_CSV)

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    # check files
    if not os.path.exists(COORDINATES_CSV):
        print(f"Missing {COORDINATES_CSV} in current directory. Place coordenadas.csv here.")
        sys.exit(1)
    if not os.path.exists(WIND_CSV):
        print(f"Missing {WIND_CSV} in current directory. Place vento.csv here.")
        sys.exit(1)
    coords = read_coordinates(COORDINATES_CSV)
    winds = read_wind(WIND_CSV)

    # LIMIT FOR TESTING: reduce number of coordinates to a manageable amount
    MAX_POINTS = None  # <-- adjust for experiments; set to None to keep all
    if MAX_POINTS is not None and len(coords) > MAX_POINTS:
        print(f"Too many coordinates ({len(coords)}). Truncating to first {MAX_POINTS} for testing.")
        coords = coords[:MAX_POINTS]

    print(f"Loaded {len(coords)} coordinates and wind table for 7 days.")
    best_ind, best_fit, best_details, sim = run_ga(coords, winds)
    if best_ind is None or best_details is None:
        print("No valid solution found.")
        sys.exit(2)
    print(f"Best fitness: {best_fit:.2f}")
    print(f"Total time (s): {best_details['total_time_s']:.1f}, recharges: {best_details['total_recharges']}, cost R$: {best_details['total_cost_R']:.2f}")
    write_output_csv(best_details, sim, OUTPUT_CSV)

if __name__ == "__main__":
    main()