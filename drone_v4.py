import csv
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import time

# ============================================================================
# CLASSES DE DADOS
# ============================================================================

@dataclass
class CEP:
    """Representa um ponto de entrega (CEP) com coordenadas."""
    id: int
    cep_code: str
    latitude: float
    longitude: float

@dataclass
class WindData:
    """Dados de vento para um horário específico."""
    velocidade_kmh: float
    direcao_graus: float

@dataclass
class FlightSegment:
    """Representa um segmento de voo entre dois CEPs."""
    cep_origem: CEP
    cep_destino: CEP
    dia: int
    hora_partida: int
    velocidade_drone: float
    distancia_km: float
    tempo_voo_segundos: float
    consumo_bateria_segundos: float
    tempo_parada_segundos: float = 72.0
    necessita_recarga: bool = False
    hora_chegada: int = 0
    custo_recarga: float = 0.0

# ============================================================================
# CLASSE: GERENCIADOR DE COORDENADAS E DISTÂNCIAS
# ============================================================================

class CoordinateManager:
    """Gerencia distâncias e coordenadas entre CEPs."""
    
    RAIO_TERRA_KM = 6371.0
    
    def __init__(self, ceps: List[CEP]):
        self.ceps = ceps
        self.distancia_cache = {}
        self._calcular_distancias()
    
    def _calcular_distancias(self):
        """Calcula e cacheia todas as distâncias usando Haversine."""
        for i, cep1 in enumerate(self.ceps):
            for j, cep2 in enumerate(self.ceps):
                if i != j:
                    distancia = self._haversine(
                        cep1.latitude, cep1.longitude,
                        cep2.latitude, cep2.longitude
                    )
                    self.distancia_cache[(i, j)] = distancia
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distância usando fórmula de Haversine."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distancia = CoordinateManager.RAIO_TERRA_KM * c
        
        return distancia
    
    def obter_distancia(self, id1: int, id2: int) -> float:
        """Obtém distância cacheada entre dois CEPs."""
        return self.distancia_cache.get((id1, id2), 0.0)
    
    def obter_azimute(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula azimute entre dois pontos."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        azimute = math.degrees(math.atan2(y, x))
        
        return (azimute + 360) % 360

# ============================================================================
# CLASSE: GERENCIADOR DE VENTO
# ============================================================================

class WindManager:
    """Gerencia dados de vento e calcula efeitos."""
    
    def __init__(self, vento_data: Dict[int, Dict[int, WindData]]):
        self.vento_data = vento_data
    
    def obter_vento(self, dia: int, hora: int) -> WindData:
        """Obtém dados de vento para dia e hora."""
        if dia in self.vento_data and hora in self.vento_data[dia]:
            return self.vento_data[dia][hora]
        return WindData(0.0, 0.0)
    
    def calcular_velocidade_efetiva(
        self,
        velocidade_drone_kmh: float,
        azimute_voo: float,
        vento: WindData
    ) -> float:
        """Calcula velocidade efetiva considerando vento."""
        # Converte velocidades para m/s
        v_drone_ms = velocidade_drone_kmh / 3.6
        v_vento_ms = vento.velocidade_kmh / 3.6
        
        # Componentes do drone (se movimentando na direção do azimute)
        v_drone_x = v_drone_ms * math.sin(math.radians(azimute_voo))
        v_drone_y = v_drone_ms * math.cos(math.radians(azimute_voo))
        
        # Componentes do vento
        v_vento_x = v_vento_ms * math.sin(math.radians(vento.direcao_graus))
        v_vento_y = v_vento_ms * math.cos(math.radians(vento.direcao_graus))
        
        # Velocidade efetiva (soma vetorial)
        v_efetiva_x = v_drone_x + v_vento_x
        v_efetiva_y = v_drone_y + v_vento_y
        v_efetiva_ms = math.sqrt(v_efetiva_x ** 2 + v_efetiva_y ** 2)
        
        # Converte de volta para km/h
        v_efetiva_kmh = v_efetiva_ms * 3.6
        
        return max(v_efetiva_kmh, 5.0)  # Mínimo de 5 km/h

# ============================================================================
# CLASSE: GERENCIADOR DE BATERIA E AUTONOMIA
# ============================================================================

class BatteryManager:
    """Gerencia bateria e autonomia do drone."""
    
    AUTONOMIA_BASE_SEGUNDOS = 5000  # 1h 23m 20s
    FATOR_CORRECAO = 0.93  # Curitiba
    AUTONOMIA_REAL = AUTONOMIA_BASE_SEGUNDOS * FATOR_CORRECAO  # 4650 segundos
    TEMPO_PARADA = 72.0  # segundos
    VELOCIDADE_REFERENCIA = 36.0  # km/h
    
    def __init__(self):
        self.bateria_atual = self.AUTONOMIA_REAL
    
    def calcular_autonomia(self, velocidade_kmh: float) -> float:
        """Calcula autonomia para uma velocidade específica."""
        if velocidade_kmh <= self.VELOCIDADE_REFERENCIA:
            return self.AUTONOMIA_REAL
        return self.AUTONOMIA_REAL * (self.VELOCIDADE_REFERENCIA / velocidade_kmh) ** 2
    
    def calcular_consumo(
        self,
        tempo_voo_segundos: float,
        tempo_parada_segundos: float = 72.0
    ) -> float:
        """Calcula consumo de bateria total."""
        return tempo_voo_segundos + tempo_parada_segundos
    
    def resetar_bateria(self):
        """Reseta bateria para valor máximo."""
        self.bateria_atual = self.AUTONOMIA_REAL
    
    def carregar_bateria(self) -> float:
        """Carrega bateria completamente."""
        self.bateria_atual = self.AUTONOMIA_REAL
        return self.AUTONOMIA_REAL

# ============================================================================
# CLASSE: GERENCIADOR DE RECARGAS
# ============================================================================

class ChargingManager:
    """Gerencia custos e decisões de recarga."""
    
    CUSTO_POUSO = 80.0  # R$
    TAXA_NOTURNA = 80.0  # R$ extras após 17:00
    HORA_TAXA_NOTURNA = 17
    
    @staticmethod
    def calcular_custo_recarga(hora: int) -> float:
        """Calcula custo de recarga baseado na hora."""
        custo = ChargingManager.CUSTO_POUSO
        if hora >= ChargingManager.HORA_TAXA_NOTURNA:
            custo += ChargingManager.TAXA_NOTURNA
        return custo
    
    @staticmethod
    def necessita_recarga(
        autonomia_disponivel: float,
        consumo_necessario: float,
        margem_seguranca: float = 1.1
    ) -> bool:
        """Verifica se necessita recarga."""
        return autonomia_disponivel < (consumo_necessario * margem_seguranca)

# ============================================================================
# CLASSE: CALCULADORA DE VOOS
# ============================================================================

class FlightCalculator:
    """Calcula parâmetros de voo (tempo, consumo, etc)."""
    
    def __init__(
        self,
        coord_manager: CoordinateManager,
        wind_manager: WindManager,
        battery_manager: BatteryManager
    ):
        self.coord_manager = coord_manager
        self.wind_manager = wind_manager
        self.battery_manager = battery_manager
    
    def calcular_tempo_voo(
        self,
        distancia_km: float,
        velocidade_efetiva_kmh: float
    ) -> float:
        """Calcula tempo de voo em segundos."""
        if velocidade_efetiva_kmh <= 0:
            return float('inf')
        tempo_horas = distancia_km / velocidade_efetiva_kmh
        tempo_segundos = tempo_horas * 3600
        return math.ceil(tempo_segundos)
    
    def calcular_segmento(
        self,
        cep_origem: CEP,
        cep_destino: CEP,
        dia: int,
        hora_partida: int,
        velocidade_drone_kmh: float
    ) -> Optional[FlightSegment]:
        """Calcula parâmetros de um segmento de voo."""
        
        # Valida hora de partida
        if hora_partida < 6 or hora_partida >= 19:
            return None
        
        # Calcula distância
        distancia = self.coord_manager.obter_distancia(cep_origem.id, cep_destino.id)
        
        # Calcula azimute
        azimute = self.coord_manager.obter_azimute(
            cep_origem.latitude,
            cep_origem.longitude,
            cep_destino.latitude,
            cep_destino.longitude
        )
        
        # Obtém vento
        vento = self.wind_manager.obter_vento(dia, hora_partida)
        
        # Calcula velocidade efetiva
        velocidade_efetiva = self.wind_manager.calcular_velocidade_efetiva(
            velocidade_drone_kmh,
            azimute,
            vento
        )
        
        # Calcula tempo de voo
        tempo_voo = self.calcular_tempo_voo(distancia, velocidade_efetiva)
        
        # Calcula consumo de bateria
        autonomia = self.battery_manager.calcular_autonomia(velocidade_drone_kmh)
        consumo = self.battery_manager.calcular_consumo(tempo_voo, 72.0)
        
        # Verifica se necessita recarga
        necessita_recarga = self.battery_manager.bateria_atual < consumo * 1.1
        
        # Calcula hora de chegada
        tempo_total_segundos = tempo_voo + 72
        tempo_total_horas = tempo_total_segundos / 3600
        hora_chegada = int(hora_partida + tempo_total_horas)
        
        # Valida se chega dentro do horário
        if hora_chegada >= 19:
            return None
        
        # Custo de recarga se necessário
        custo_recarga = 0.0
        if necessita_recarga:
            custo_recarga = ChargingManager.calcular_custo_recarga(hora_partida)
        
        segmento = FlightSegment(
            cep_origem=cep_origem,
            cep_destino=cep_destino,
            dia=dia,
            hora_partida=hora_partida,
            velocidade_drone=velocidade_drone_kmh,
            distancia_km=distancia,
            tempo_voo_segundos=tempo_voo,
            consumo_bateria_segundos=consumo,
            tempo_parada_segundos=72.0,
            necessita_recarga=necessita_recarga,
            hora_chegada=hora_chegada,
            custo_recarga=custo_recarga
        )
        
        return segmento

# ============================================================================
# CLASSE: CROMOSSOMO (SOLUÇÃO)
# ============================================================================

class Chromosome:
    """Representa uma solução (cromossomo) do problema."""
    
    VELOCIDADES_VALIDAS = [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
    
    def __init__(
        self,
        ordem_ceps: List[int],
        dias: List[int],
        horas: List[int],
        velocidades: List[float],
        fitness_value: float = None
    ):
        self.ordem_ceps = ordem_ceps
        self.dias = dias
        self.horas = horas
        self.velocidades = velocidades
        self.fitness_value = fitness_value
        self.custo_total = 0.0
        self.segmentos = []
    
    def clone(self) -> 'Chromosome':
        """Cria uma cópia profunda do cromossomo."""
        return Chromosome(
            ordem_ceps=self.ordem_ceps.copy(),
            dias=self.dias.copy(),
            horas=self.horas.copy(),
            velocidades=self.velocidades.copy(),
            fitness_value=self.fitness_value
        )
    
    def __repr__(self) -> str:
        return (f"Chromosome(fitness={self.fitness_value:.6f}, "
                f"custo={self.custo_total:.2f}, ceps={len(self.ordem_ceps)})")

# ============================================================================
# CLASSE: SIMULADOR DE ROTEIRO
# ============================================================================

class RouteSimulator:
    """Simula um roteiro completo e calcula custos."""
    
    def __init__(
        self,
        ceps: List[CEP],
        flight_calc: FlightCalculator,
        battery_mgr: BatteryManager
    ):
        self.ceps = ceps
        self.flight_calc = flight_calc
        self.battery_mgr = battery_mgr
        self.cep_dict = {cep.id: cep for cep in ceps}
    
    def simular_roteiro(self, chromosome: Chromosome) -> Tuple[float, bool, List[FlightSegment]]:
        """
        Simula um roteiro completo.
        Retorna: (custo_total, é_válido, segmentos)
        """
        custo_total = 0.0
        segmentos = []
        penalidades = 0.0
        valido = True
        
        self.battery_mgr.resetar_bateria()
        
        # Simula cada trecho
        for i in range(len(chromosome.ordem_ceps) - 1):
            cep_origem_id = chromosome.ordem_ceps[i]
            cep_destino_id = chromosome.ordem_ceps[i + 1]
            dia = chromosome.dias[i]
            hora = chromosome.horas[i]
            velocidade = chromosome.velocidades[i]
            
            # Valida dia
            if dia < 1 or dia > 7:
                penalidades += 1e6
                valido = False
                continue
            
            # Obtém CEPs
            cep_origem = self.cep_dict.get(cep_origem_id)
            cep_destino = self.cep_dict.get(cep_destino_id)
            
            if not cep_origem or not cep_destino:
                penalidades += 1e6
                valido = False
                continue
            
            # Calcula segmento
            segmento = self.flight_calc.calcular_segmento(
                cep_origem, cep_destino, dia, hora, velocidade
            )
            
            if not segmento:
                # Tenta recarga e retry
                if self.battery_mgr.necessita_recarga(
                    self.battery_mgr.bateria_atual,
                    self.flight_calc.battery_manager.calcular_consumo(0, 72)
                ):
                    custo_total += ChargingManager.calcular_custo_recarga(hora)
                    self.battery_mgr.carregar_bateria()
                    
                    # Retry do segmento
                    segmento = self.flight_calc.calcular_segmento(
                        cep_origem, cep_destino, dia, hora, velocidade
                    )
                
                if not segmento:
                    penalidades += 1e7
                    valido = False
                    continue
            
            # Atualiza bateria
            self.battery_mgr.bateria_atual -= segmento.consumo_bateria_segundos
            
            if self.battery_mgr.bateria_atual < 0:
                penalidades += 1e7
                valido = False
                self.battery_mgr.resetar_bateria()
            
            # Adiciona custos
            custo_total += segmento.custo_recarga
            segmentos.append(segmento)
        
        custo_total += penalidades
        return custo_total, valido, segmentos

# ============================================================================
# CLASSE: POPULAÇÃO GENÉTICA
# ============================================================================

class Population:
    """Gerencia a população de cromossomos."""
    
    def __init__(
        self,
        tamanho: int,
        num_ceps: int,
        cep_inicial_id: int = 0
    ):
        self.tamanho = tamanho
        self.num_ceps = num_ceps
        self.cep_inicial_id = cep_inicial_id
        self.cromossomos: List[Chromosome] = []
    
    def gerar_populacao_inicial(self) -> List[Chromosome]:
        """Gera população inicial aleatória."""
        self.cromossomos = []
        
        for _ in range(self.tamanho):
            # Cria ordem de CEPs (mantendo primeiro e último como CEP inicial)
            ceps_meio = list(range(self.num_ceps))
            ceps_meio.remove(self.cep_inicial_id)
            random.shuffle(ceps_meio)
            ordem = [self.cep_inicial_id] + ceps_meio + [self.cep_inicial_id]
            
            # Cria dias aleatórios (1-7)
            dias = [random.randint(1, 7) for _ in range(len(ordem) - 1)]
            
            # Cria horas aleatórias (6-18)
            horas = [random.randint(6, 18) for _ in range(len(ordem) - 1)]
            
            # Cria velocidades aleatórias
            velocidades = [random.choice(Chromosome.VELOCIDADES_VALIDAS) 
                          for _ in range(len(ordem) - 1)]
            
            cromossomo = Chromosome(ordem, dias, horas, velocidades)
            self.cromossomos.append(cromossomo)
        
        return self.cromossomos
    
    def adicionar_cromossomo(self, cromossomo: Chromosome):
        """Adiciona cromossomo à população."""
        self.cromossomos.append(cromossomo)
    
    def ordenar_por_fitness(self):
        """Ordena população por fitness (melhor primeiro)."""
        self.cromossomos.sort(key=lambda x: x.fitness_value or 0, reverse=True)

# ============================================================================
# CLASSE: OPERADORES GENÉTICOS
# ============================================================================

class GeneticOperators:
    """Implementa operadores genéticos (crossover, mutação, etc)."""
    
    @staticmethod
    def pmx_crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """
        Partially Mapped Crossover (PMX) para permutações.
        """
        ordem1 = parent1.ordem_ceps[:-1]  # Remove última repetição
        ordem2 = parent2.ordem_ceps[:-1]
        
        tamanho = len(ordem1)
        p1 = random.randint(1, tamanho - 2)
        p2 = random.randint(p1 + 1, tamanho - 1)
        
        # Cria filho
        filho_ordem = [-1] * tamanho
        
        # Copia segmento de parent1
        for i in range(p1, p2):
            filho_ordem[i] = ordem1[i]
        
        # Mapeia elementos de parent2
        for i in range(tamanho):
            if i < p1 or i >= p2:
                elemento = ordem2[i]
                while elemento in filho_ordem:
                    idx = ordem2.index(elemento)
                    elemento = ordem1[idx]
                filho_ordem[i] = elemento
        
        # Adiciona CEP inicial no final
        filho_ordem.append(filho_ordem[0])
        
        # Combina outros genes
        dias = [parent1.dias[i] if random.random() < 0.5 else parent2.dias[i] 
                for i in range(len(parent1.dias))]
        horas = [parent1.horas[i] if random.random() < 0.5 else parent2.horas[i] 
                 for i in range(len(parent1.horas))]
        velocidades = [parent1.velocidades[i] if random.random() < 0.5 
                      else parent2.velocidades[i] 
                      for i in range(len(parent1.velocidades))]
        
        return Chromosome(filho_ordem, dias, horas, velocidades)
    
    @staticmethod
    def mutacao_troca(cromossomo: Chromosome, taxa: float = 0.05) -> Chromosome:
        """Mutação por troca de posições."""
        novo = cromossomo.clone()
        
        # Mutação na ordem de CEPs
        ordem = novo.ordem_ceps[:-1]  # Remove última repetição
        if random.random() < taxa and len(ordem) > 2:
            i, j = random.sample(range(1, len(ordem) - 1), 2)
            ordem[i], ordem[j] = ordem[j], ordem[i]
        novo.ordem_ceps = ordem + [ordem[0]]
        
        # Mutação em dias, horas e velocidades
        for i in range(len(novo.dias)):
            if random.random() < taxa:
                novo.dias[i] = random.randint(1, 7)
            if random.random() < taxa:
                novo.horas[i] = random.randint(6, 18)
            if random.random() < taxa:
                novo.velocidades[i] = random.choice(Chromosome.VELOCIDADES_VALIDAS)
        
        return novo
    
    @staticmethod
    def mutacao_inversao(cromossomo: Chromosome, taxa: float = 0.05) -> Chromosome:
        """Mutação por inversão de segmento."""
        novo = cromossomo.clone()
        
        if random.random() < taxa and len(novo.ordem_ceps) > 3:
            ordem = novo.ordem_ceps[:-1]
            i, j = sorted(random.sample(range(1, len(ordem) - 1), 2))
            ordem[i:j+1] = reversed(ordem[i:j+1])
            novo.ordem_ceps = ordem + [ordem[0]]
        
        return novo
    
    @staticmethod
    def selecao_torneio(
        populacao: Population,
        tamanho_torneio: int = 3
    ) -> Chromosome:
        """Seleciona cromossomo por torneio."""
        torneio = random.sample(populacao.cromossomos, 
                               min(tamanho_torneio, len(populacao.cromossomos)))
        return max(torneio, key=lambda x: x.fitness_value or 0)

# ============================================================================
# CLASSE: ALGORITMO GENÉTICO
# ============================================================================

class GeneticAlgorithm:
    """Implementa o algoritmo genético completo."""
    
    def __init__(
        self,
        tamanho_populacao: int = 50,
        num_geracoes: int = 100,
        taxa_mutacao: float = 0.1,
        taxa_crossover: float = 0.8,
        elitismo: int = 5,
        ceps: List[CEP] = None,
        vento_data: Dict = None
    ):
        self.tamanho_populacao = tamanho_populacao
        self.num_geracoes = num_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.elitismo = elitismo
        
        # Inicializa managers
        self.coord_manager = CoordinateManager(ceps)
        self.wind_manager = WindManager(vento_data)
        self.battery_manager = BatteryManager()
        self.flight_calc = FlightCalculator(
            self.coord_manager,
            self.wind_manager,
            self.battery_manager
        )
        self.route_simulator = RouteSimulator(ceps, self.flight_calc, self.battery_manager)
        
        self.ceps = ceps
        self.melhor_cromossomo: Optional[Chromosome] = None
        self.melhor_fitness = 0.0
        self.historico_fitness = []
    
    def calcular_fitness(self, cromossomo: Chromosome) -> float:
        """Calcula fitness normalizado entre 0 e 1."""
        custo_total, valido, segmentos = self.route_simulator.simular_roteiro(cromossomo)
        
        cromossomo.custo_total = custo_total
        cromossomo.segmentos = segmentos
        
        # Normaliza fitness: quanto menor o custo, melhor o fitness
        fitness = 1.0 / (1.0 + custo_total)
        cromossomo.fitness_value = fitness
        
        return fitness
    
    def executar(self, verbose: bool = True) -> Chromosome:
        """Executa o algoritmo genético."""
        if verbose:
            print("\n" + "="*70)
            print("ALGORITMO GENÉTICO - OTIMIZAÇÃO DE ROTAS DE DRONE")
            print("="*70)
        
        # Gera população inicial
        populacao = Population(
            self.tamanho_populacao,
            len(self.ceps),
            cep_inicial_id=0
        )
        populacao.gerar_populacao_inicial()
        
        # Avalia população inicial
        if verbose:
            print(f"\nAvaliando população inicial ({len(populacao.cromossomos)} indivíduos)...")
        
        for cromossomo in populacao.cromossomos:
            self.calcular_fitness(cromossomo)
        
        populacao.ordenar_por_fitness()
        self.melhor_cromossomo = populacao.cromossomos[0].clone()
        self.melhor_fitness = self.melhor_cromossomo.fitness_value
        
        if verbose:
            print(f"Melhor fitness inicial: {self.melhor_fitness:.6f}")
            print(f"Melhor custo inicial: R$ {self.melhor_cromossomo.custo_total:.2f}")
        
        # Loop de gerações
        for geracao in range(self.num_geracoes):
            if verbose and (geracao + 1) % max(1, self.num_geracoes // 10) == 0:
                print(f"\nGe geração {geracao + 1}/{self.num_geracoes}")
            
            # Elitismo - mantém melhores cromossomos
            nova_populacao = Population(self.tamanho_populacao, len(self.ceps))
            populacao.ordenar_por_fitness()
            
            for i in range(self.elitismo):
                if i < len(populacao.cromossomos):
                    nova_populacao.adicionar_cromossomo(
                        populacao.cromossomos[i].clone()
                    )
            
            # Gera novos indivíduos via crossover e mutação
            while len(nova_populacao.cromossomos) < self.tamanho_populacao:
                # Seleção
                parent1 = GeneticOperators.selecao_torneio(populacao)
                parent2 = GeneticOperators.selecao_torneio(populacao)
                
                # Crossover
                if random.random() < self.taxa_crossover:
                    filho = GeneticOperators.pmx_crossover(parent1, parent2)
                else:
                    filho = parent1.clone()
                
                # Mutação
                if random.random() < self.taxa_mutacao:
                    if random.random() < 0.5:
                        filho = GeneticOperators.mutacao_troca(filho, self.taxa_mutacao)
                    else:
                        filho = GeneticOperators.mutacao_inversao(filho, self.taxa_mutacao)
                
                nova_populacao.adicionar_cromossomo(filho)
            
            # Ajusta tamanho
            nova_populacao.cromossomos = nova_populacao.cromossomos[:self.tamanho_populacao]
            populacao = nova_populacao
            
            # Avalia nova população
            for cromossomo in populacao.cromossomos:
                self.calcular_fitness(cromossomo)
            
            # Atualiza melhor solução
            populacao.ordenar_por_fitness()
            if populacao.cromossomos[0].fitness_value > self.melhor_fitness:
                self.melhor_fitness = populacao.cromossomos[0].fitness_value
                self.melhor_cromossomo = populacao.cromossomos[0].clone()
                
                if verbose:
                    print(f"  ✓ Nova melhor solução encontrada!")
                    print(f"    Fitness: {self.melhor_fitness:.6f}")
                    print(f"    Custo: R$ {self.melhor_cromossomo.custo_total:.2f}")
            
            self.historico_fitness.append(self.melhor_fitness)
        
        if verbose:
            print("\n" + "="*70)
            print("ALGORITMO GENÉTICO FINALIZADO")
            print("="*70)
            print(f"Melhor fitness encontrado: {self.melhor_fitness:.6f}")
            print(f"Melhor custo encontrado: R$ {self.melhor_cromossomo.custo_total:.2f}")
            print(f"Total de segmentos: {len(self.melhor_cromossomo.segmentos)}")
        
        return self.melhor_cromossomo

# ============================================================================
# CLASSE: GERENCIADOR DE DADOS
# ============================================================================

class DataManager:
    """Gerencia leitura e escrita de dados."""
    
    @staticmethod
    def carregar_ceps(arquivo_csv: str) -> List[CEP]:
        """Carrega CEPs do arquivo CSV."""
        ceps = []
        with open(arquivo_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                cep = CEP(
                    id=idx,
                    cep_code=row['cep'],
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude'])
                )
                ceps.append(cep)
        return ceps
    
    @staticmethod
    def carregar_vento(arquivo_csv: str) -> Dict[int, Dict[int, WindData]]:
        """Carrega dados de vento do arquivo CSV."""
        vento_data = {}
        with open(arquivo_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dia = int(row['dia'])
                hora = int(row['hora'])
                vel = float(row['vel_kmh'])
                direcao = float(row['direcao_deg'])
                
                if dia not in vento_data:
                    vento_data[dia] = {}
                
                vento_data[dia][hora] = WindData(vel, direcao)
        
        return vento_data
    
    @staticmethod
    def salvar_resultado(
        arquivo_saida: str,
        melhor_cromossomo: Chromosome,
        cep_dict: Dict[int, CEP]
    ):
        """Salva resultado em arquivo CSV."""
        with open(arquivo_saida, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            writer.writerow([
                'CEP_inicial',
                'Latitude_inicial',
                'Longitude_inicial',
                'Dia_voo',
                'Hora_partida',
                'Velocidade_kmh',
                'CEP_final',
                'Latitude_final',
                'Longitude_final',
                'Pouso',
                'Hora_chegada',
                'Custo_recarga',
                'Tempo_voo_min'
            ])
            
            # Dados dos segmentos
            for i, segmento in enumerate(melhor_cromossomo.segmentos):
                tempo_voo_min = segmento.tempo_voo_segundos / 60.0
                pouso = "SIM" if segmento.necessita_recarga else "NÃO"
                
                writer.writerow([
                    segmento.cep_origem.cep_code,
                    f"{segmento.cep_origem.latitude:.10f}",
                    f"{segmento.cep_origem.longitude:.10f}",
                    segmento.dia,
                    f"{segmento.hora_partida:02d}:00",
                    segmento.velocidade_drone,
                    segmento.cep_destino.cep_code,
                    f"{segmento.cep_destino.latitude:.10f}",
                    f"{segmento.cep_destino.longitude:.10f}",
                    pouso,
                    f"{segmento.hora_chegada:02d}:00",
                    f"R$ {segmento.custo_recarga:.2f}",
                    f"{tempo_voo_min:.2f}"
                ])

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal."""
    import os
    
    # Diretório base
    base_dir = "/home/vinicius/drone-genetic-algorythm"
    arquivo_ceps = os.path.join(base_dir, "coordenadas.csv")
    arquivo_vento = os.path.join(base_dir, "vento.csv")
    arquivo_saida = os.path.join(base_dir, "resultado_otimizado.csv")
    
    print("\n" + "="*70)
    print("OTIMIZADOR DE ROTAS PARA DRONE COM ALGORITMO GENÉTICO")
    print("="*70)
    
    # Carrega dados
    print("\n[1/5] Carregando CEPs...")
    ceps = DataManager.carregar_ceps(arquivo_ceps)
    print(f"     ✓ {len(ceps)} CEPs carregados")
    
    print("\n[2/5] Carregando dados de vento...")
    vento_data = DataManager.carregar_vento(arquivo_vento)
    print(f"     ✓ Dados de vento carregados para {len(vento_data)} dias")
    
    # Configura algoritmo genético
    print("\n[3/5] Configurando algoritmo genético...")
    ag = GeneticAlgorithm(
        tamanho_populacao=80,
        num_geracoes=150,
        taxa_mutacao=0.15,
        taxa_crossover=0.85,
        elitismo=8,
        ceps=ceps,
        vento_data=vento_data
    )
    print("     ✓ Algoritmo genético configurado")
    
    # Executa otimização
    print("\n[4/5] Executando otimização...")
    tempo_inicio = time.time()
    melhor_solucao = ag.executar(verbose=True)
    tempo_decorrido = time.time() - tempo_inicio
    
    print(f"\n     ✓ Otimização concluída em {tempo_decorrido:.2f}s")
    
    # Salva resultado
    print("\n[5/5] Salvando resultado...")
    cep_dict = {cep.id: cep for cep in ceps}
    DataManager.salvar_resultado(arquivo_saida, melhor_solucao, cep_dict)
    print(f"     ✓ Resultado salvo em: {arquivo_saida}")
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print(f"Fitness da melhor solução:     {melhor_solucao.fitness_value:.6f}")
    print(f"Custo total da rota:          R$ {melhor_solucao.custo_total:.2f}")
    print(f"Número de segmentos de voo:   {len(melhor_solucao.segmentos)}")
    print(f"Distância total aproximada:   {sum(s.distancia_km for s in melhor_solucao.segmentos):.2f} km")
    print(f"Tempo total de voo:           {sum(s.tempo_voo_segundos for s in melhor_solucao.segmentos)/3600:.2f} horas")
    print(f"Tempo de processamento:       {tempo_decorrido:.2f} segundos")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()