# Código Python completo para otimização de rota de drone usando Algoritmo Genético
# Autor: Grok 4 (baseado em especificações do problema)
# Descrição: Implementa um AG para resolver um problema de roteamento de drone com restrições de vento, bateria, horários e custos de recarga.
# Estruturado em funções modulares alinhadas aos 10 passos descritos.
# Bibliotecas usadas: numpy (para cálculos matemáticos), math (funções trigonométricas), random (para aleatoriedade no AG),
# csv (para gerar arquivo de saída), deap (para framework de algoritmo genético).
# Dados: Usamos placeholders simulados para CEPs, coordenadas e ventos, pois não há acesso a dados reais.
# Assunções: Matrícula inicia com 2, então aplica fator 0.93 na bateria, arredondamento para cima em cálculos, custos extras após 17:00.
# Velocidade do drone: Múltiplos de 4 km/h, mínimo 36 km/h (aprox. 10 m/s).
# Autonomia base: 5000 segundos, ajustada por vento e fator.
# Horários: Voo apenas entre 06:00 e 19:00, até 7 dias.
# Objetivo: Minimizar custo total (tempo de voo em segundos + custos de recarga em R$ convertidos para equivalente, assumindo 1 R$ = 1 unidade de custo para simplificação).

import numpy as np
import math
import random
import csv
import os
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta

# Parâmetros globais
BASE_DIR = os.path.dirname(__file__)
CEP_INICIAL = '82821020'
HORARIO_INICIO = datetime.strptime('06:00:00', '%H:%M:%S').time()
HORARIO_FIM = datetime.strptime('19:00:00', '%H:%M:%S').time()
HORARIO_EXTRA = datetime.strptime('17:00:00', '%H:%M:%S').time()
AUTONOMIA_BASE = 5000  # segundos
FATOR_BATERIA = 0.93  # Aplicado pois matrícula inicia com 2
VELOCIDADE_BASE = 36  # km/h (mínimo, múltiplo de 4)
CUSTO_RECARGA = 80  # R$
CUSTO_EXTRA = 80  # Após 17:00
RAIO_TERRA = 6371  # km para Haversine
DIAS_MAX = 7

# Passo 1: Entender o objetivo
# Objetivo: Otimizar a rota do drone para visitar CEPs em Curitiba, minimizando custo total (tempo + recargas),
# respeitando bateria, ventos, horários e limite de 7 dias. Drone inicia e termina no CEP 82821020.

# Passo 2: Coletar dados
# Simulando dados de CEPs com coordenadas (latitude, longitude). Em um caso real, leia de um arquivo CSV.
# CEP inicial: 82821020 (Unibrasil) - Coordenadas simuladas.
def ler_coordenadas(csv_path=None):
    path = csv_path or os.path.join(BASE_DIR, 'coordenadas.csv')
    ceps_local = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cep = row['cep'].strip()
                lon = float(row.get('longitude') or row.get('lon'))
                lat = float(row.get('latitude') or row.get('lat'))
                ceps_local.append({'cep': cep, 'lat': lat, 'lon': lon})
            except Exception:
                continue
    return ceps_local

# Simulando tabelas de ventos por dia/hora. Formato: dia (1-7), hora (0-23), velocidade (km/h), direção (graus).
# Em real: Extraia de tabelas e converta unidades (ex: m/s para km/h).
def ler_vento(csv_path=None):
    path = csv_path or os.path.join(BASE_DIR, 'vento.csv')
    v = {}
    horas_por_dia = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dia = int(row['dia'])
                hora = int(row['hora'])
                vel = float(row.get('vel_kmh') or row.get('velocidade') or 0)
                direc = float(row.get('direcao_deg') or row.get('direcao') or 0)
                v[(dia, hora)] = {'velocidade': vel, 'direcao': direc}
                horas_por_dia.setdefault(dia, set()).add(hora)
            except Exception:
                continue
    return v, horas_por_dia


def obter_vento(dia, hora):
    """
    Retorna vento exato se existir, senão retorna o registro da hora mais próxima no mesmo dia.
    """
    if (dia, hora) in ventos:
        return ventos[(dia, hora)]
    hours = sorted(list(horas_por_dia.get(dia, [])))
    if not hours:
        return {'velocidade': 0.0, 'direcao': 0.0}
    closest = min(hours, key=lambda h: abs(h - hora))
    return ventos[(dia, closest)]

# Carregar dados dos CSVs
ceps = ler_coordenadas()
if not ceps:
    raise FileNotFoundError("Arquivo coordenadas.csv não encontrado ou vazio em " + BASE_DIR)

# Certificar que CEP_INICIAL está na posição 0
idx_inicio = next((i for i, c in enumerate(ceps) if c['cep'] == CEP_INICIAL), None)
if idx_inicio is None:
    raise ValueError(f"CEP inicial {CEP_INICIAL} não encontrado em coordenadas.csv")
if idx_inicio != 0:
    ceps[0], ceps[idx_inicio] = ceps[idx_inicio], ceps[0]

num_ceps = len(ceps) - 1

ventos, horas_por_dia = ler_vento()




# Passo 3: Calcular distâncias
def calcular_distancia(lat1, lon1, lat2, lon2):
    """
    Calcula distância em km entre dois pontos usando fórmula de Haversine.
    Tratamento de erros: Verifica se coordenadas são válidas.
    """
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return RAIO_TERRA * c
    except ValueError:
        raise ValueError("Coordenadas inválidas para cálculo de distância.")

# Pré-calcular matriz de distâncias
distancias = np.zeros((len(ceps), len(ceps)))
for i in range(len(ceps)):
    for j in range(len(ceps)):
        if i != j:
            distancias[i][j] = calcular_distancia(ceps[i]['lat'], ceps[i]['lon'], ceps[j]['lat'], ceps[j]['lon'])

# Passo 4: Modelar vento
def modelar_vento(vel_vento, dir_vento, dir_voo):
    """
    Modela o efeito do vento na velocidade efetiva do drone.
    dir_voo: direção do voo em graus.
    Retorna fator de correção para velocidade ( >1 se vento contra, <1 se a favor).
    Simplificação: Componente do vento na direção do voo.
    """
    angulo = abs(dir_voo - dir_vento)
    componente = vel_vento * math.cos(math.radians(angulo))
    return 1 - (componente / VELOCIDADE_BASE)  # Ajuste: vento contra reduz velocidade efetiva

# Passo 5: Calcular tempos
def calcular_tempo(distancia, vel_vento, dir_vento, dir_voo):
    """
    Calcula tempo de voo em segundos, considerando vento.
    Velocidade efetiva = VELOCIDADE_BASE * fator_vento.
    Arredonda para cima, conforme assunção.
    """
    fator_vento = modelar_vento(vel_vento, dir_vento, dir_voo)
    vel_efetiva = VELOCIDADE_BASE * fator_vento
    if vel_efetiva <= 0:
        raise ValueError("Velocidade efetiva inválida devido a vento forte.")
    tempo = math.ceil((distancia / vel_efetiva) * 3600)  # km/h para segundos, arredonda para cima
    return tempo

def calcular_dir_voo(lat1, lon1, lat2, lon2):
    """
    Calcula direção do voo em graus (0-360).
    """
    dlon = math.radians(lon2 - lon1)
    lat1, lat2 = map(math.radians, [lat1, lat2])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

# Passo 6: Definir cromossomo
# Cromossomo: Permutação dos CEPs (índices 1 a N), representando ordem de visita.
# O inicial é fixo no início e fim.
def criar_individuo():
    individuo = list(range(1, len(ceps)))
    random.shuffle(individuo)
    return individuo

# Passo 7: Implementar fitness
def funcao_fitness(individuo, verbose=False):
    """
    Calcula fitness: Custo total (tempo + custos recarga). Menor é melhor.
    Simula a viagem: Divide em dias se necessário, verifica bateria/horários, adiciona recargas.
    Penaliza heavily se inviável (ex: >7 dias, bateria insuficiente).
    Retorna tupla (custo_total,) para DEAP.
    """
    custo_total = 0
    tempo_total = 0
    recargas = 0
    dia_atual = 1
    hora_atual = datetime.strptime('06:00:00', '%H:%M:%S')
    bateria_atual = AUTONOMIA_BASE * FATOR_BATERIA
    
    # Rota completa: Inicial -> ordem -> Inicial
    rota = [0] + individuo + [0]
    
    for i in range(len(rota) - 1):
        orig = rota[i]
        dest = rota[i+1]
        dist = distancias[orig][dest]
        dir_voo = calcular_dir_voo(ceps[orig]['lat'], ceps[orig]['lon'], ceps[dest]['lat'], ceps[dest]['lon'])
        
        # Pegar vento na hora atual (simplificação: usa hora inteira)
        hora_int = hora_atual.hour
        vento = obter_vento(dia_atual, hora_int)
        
        tempo_voo = calcular_tempo(dist, vento['velocidade'], vento['direcao'], dir_voo)
        
        # Verificar bateria
        consumo = tempo_voo / FATOR_BATERIA  # Ajuste simplificado
        if consumo > bateria_atual:
            # Recarregar
            recargas += 1
            custo_recarga = CUSTO_RECARGA
            if hora_atual.time() > HORARIO_EXTRA:
                custo_recarga += CUSTO_EXTRA
            custo_total += custo_recarga
            bateria_atual = AUTONOMIA_BASE * FATOR_BATERIA
            # Tempo de recarga? Assumindo instantâneo para simplificação.
        
        bateria_atual -= consumo
        
        # Atualizar hora
        hora_atual += timedelta(seconds=tempo_voo)
        tempo_total += tempo_voo
        
        # Verificar horário de voo
        while hora_atual.time() > HORARIO_FIM or hora_atual.time() < HORARIO_INICIO:
            # Avançar para próximo dia
            dia_atual += 1
            hora_atual = datetime(hora_atual.year, hora_atual.month, hora_atual.day + 1, 6, 0, 0)
            if dia_atual > DIAS_MAX:
                return (float('inf'),)  # Penalidade: Inviável
    
    custo_total += tempo_total  # Tempo como parte do custo
    if verbose:
        print(f"Custo total: {custo_total}, Dias: {dia_atual}, Recargas: {recargas}")
    return (custo_total,)


# --- Adicionadas funções de validação e avaliação segura ---
def validar_individuo(individuo):
    """
    Verifica se individuo é uma permutação válida de 1..(len(ceps)-1).
    Retorna False se houver índices fora do intervalo ou elementos repetidos.
    """
    n = len(ceps) - 1
    if not isinstance(individuo, (list, tuple)):
        return False
    if len(individuo) != n:
        return False
    # Todos os elementos devem ser inteiros no intervalo 1..n e sem repetição
    try:
        s = set(int(x) for x in individuo)
    except Exception:
        return False
    return s == set(range(1, n+1))

def avaliacao_segura(individuo):
    """
    Wrapper para funcao_fitness que valida e captura exceções.
    Retorna (inf,) em caso de problema para penalizar.
    """
    try:
        if not validar_individuo(individuo):
            return (float('inf'),)
        return funcao_fitness(individuo)
    except Exception:
        return (float('inf'),)
# --- Adições: reparo e wrappers seguros para operadores genéticos ---
def reparar_individuo(ind):
    """Garante que 'ind' seja uma permutação válida; se não, substitui em-place."""
    if validar_individuo(ind):
        return
    novo = criar_individuo()
    # sobrescreve conteúdo do indivíduo em-place (DEAP espera isso)
    ind[:] = novo

def safe_cx_pmx(ind1, ind2):
    """Wrapper seguro para cxPartialyMatched (PMX). Repara em caso de erro."""
    try:
        tools.cxPartialyMatched(ind1, ind2)
    except Exception:
        # substitui por indivíduos válidos caso ocorra erro
        ind1[:] = criar_individuo()
        ind2[:] = criar_individuo()
    finally:
        reparar_individuo(ind1)
        reparar_individuo(ind2)
    return ind1, ind2

def safe_mut_shuffle(individual, indpb=0.2):
    """Wrapper seguro para mutShuffleIndexes. Repara em caso de erro."""
    try:
        tools.mutShuffleIndexes(individual, indpb=indpb)
    except Exception:
        individual[:] = criar_individuo()
    finally:
        reparar_individuo(individual)
    return (individual,)

# --- Fim adições ---

# Passo 8: Configurar operadores genéticos
# Usando DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, criar_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# substituir registros diretos por wrappers seguros
toolbox.register("mate", safe_cx_pmx)
toolbox.register("mutate", safe_mut_shuffle, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
# Substitui a avaliação direta por uma versão segura
toolbox.register("evaluate", avaliacao_segura)


# Passo 9: Executar o algoritmo
def executar_algoritmo(pop_size=100, geracoes=50):
    """
    Executa o AG e retorna o melhor indivíduo.
    """
    pop = toolbox.population(n=pop_size)
    # valida população inicial — se houver individuo inválido, substitui por novo
    for i, ind in enumerate(pop):
        if not validar_individuo(ind):
            pop[i] = toolbox.individual()

    hof = tools.HallOfFame(1)
    
    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=geracoes, stats=stats, halloffame=hof, verbose=True)
    
    if len(hof) == 0:
        return []
    return hof[0]

# Passo 10: Gerar CSV
def gerar_csv(melhor_rota, arquivo='output_drone_v3.csv'):
    """
    Gera CSV com colunas: CEP inicial, Latitude inicial, Longitude inicial, CEP destino, Latitude destino, Longitude destino, Distância (km), Tempo de voo (s), Custo recarga (se aplicável).
    Para toda a rota.
    """
    with open(arquivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CEP Inicial', 'Lat Inicial', 'Lon Inicial', 'CEP Destino', 'Lat Destino', 'Lon Destino', 'Distancia (km)', 'Tempo Voo (s)', 'Custo Recarga'])
        
        rota = [0] + melhor_rota + [0]
        for i in range(len(rota) - 1):
            orig = rota[i]
            dest = rota[i+1]
            dist = distancias[orig][dest]
            # Tempo simplificado sem vento para CSV (ou calcule completo se necessário)
            tempo = math.ceil((dist / VELOCIDADE_BASE) * 3600)
            custo_recarga = 0  # Simule se houver recarga aqui (lógica completa em fitness)
            writer.writerow([
                ceps[orig]['cep'], ceps[orig]['lat'], ceps[orig]['lon'],
                ceps[dest]['cep'], ceps[dest]['lat'], ceps[dest]['lon'],
                dist, tempo, custo_recarga
            ])

# Seção principal (main)
if __name__ == "__main__":
    try:
        melhor_ind = executar_algoritmo()
        if not melhor_ind:
            print("Nenhuma solução válida encontrada.")
        else:
            print("Melhor rota encontrada:", [ceps[i]['cep'] for i in melhor_ind])
            funcao_fitness(melhor_ind, verbose=True)
            gerar_csv(melhor_ind)
            print("Arquivo CSV gerado: rota_otimizada.csv")
    except Exception as e:
        print(f"Erro na execução: {e}")