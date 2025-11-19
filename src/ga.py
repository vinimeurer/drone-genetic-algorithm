# src/ga.py
import random
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Any

from .utils import LRUCache, hash_individuo
from .v_eff import build_v_eff_table
from .evaluator import avaliar_lote_numba
from .constants import *

class GeneticAlgorithm:
    def __init__(self, coord, vento, drone,
                 n_pop=DEFAULT_N_POP, n_gen=DEFAULT_N_GEN, elitismo=DEFAULT_ELITISMO,
                 taxa_mut_inicial=DEFAULT_TAXA_MUT_INI, taxa_mut_final=DEFAULT_TAXA_MUT_FIN,
                 seed=SEED, n_workers=WORKERS, max_stagnation=MAX_STAGNATION):
        self.coord = coord
        self.vento = vento
        self.drone = drone
        self.n_pop = int(n_pop)
        self.n_gen = int(n_gen)
        self.elitismo = float(elitismo)
        self.taxa_mut_inicial = taxa_mut_inicial
        self.taxa_mut_final = taxa_mut_final
        self.seed = seed
        self.n_workers = n_workers
        self.max_stagnation = max_stagnation
        self.log_interval = LOG_INTERVAL

        random.seed(seed)
        np.random.seed(seed)

        self.base = [i for i in coord.ids if i != 1]
        self.N = len(coord.ids)
        self.L = len(self.base) + 2

        self.dist_matrix = np.ascontiguousarray(coord.dist_matrix.astype(np.float32))
        self.az_matrix = np.ascontiguousarray(coord.az_matrix.astype(np.float32))
        self.vento_array = np.ascontiguousarray(vento.vento_array.astype(np.float32))

        self.v_eff_table = build_v_eff_table(
            np.array(drone.velocidades, dtype=np.float32),
            self.vento_array
        )

        self.vento_idx_map = np.arange(self.vento_array.shape[0] * 24, dtype=np.int32)
        self.fitness_cache = LRUCache(FIT_CACHE_MAX_ENTRIES)

    def inicializar_populacao(self) -> List[Tuple[List[int], List[int]]]:
        pop = []
        base = self.base
        vels = self.drone.velocidades
        weights = [0.25] * (len(vels) - 2) + [0.75] * 2
        for _ in range(self.n_pop):
            perm = random.sample(base, len(base))
            rota = [1] + perm + [1]
            velocidades = random.choices(vels, weights=weights, k=len(rota) - 1)
            pop.append((rota, velocidades))
        return pop

    def _avaliar_lote(self, individuos):
        B = len(individuos)
        rotas_batch = np.zeros((B, self.L), dtype=np.uint16)
        vels_batch = np.zeros((B, self.L - 1), dtype=np.uint8)
        hashes = [None] * B

        for b, (rota, vels) in enumerate(individuos):
            if len(rota) != self.L:
                rota = (rota + [1] * self.L)[:self.L]
            for i, rid in enumerate(rota):
                rotas_batch[b, i] = self.coord.idx_map.get(int(rid), 0)
            for i, v in enumerate(vels):
                vels_batch[b, i] = self.drone.vel_to_idx.get(int(v), 0)

            key = hash_individuo(rotas_batch[b].tobytes(), vels_batch[b].tobytes(), digest_size=FIT_CACHE_DIGEST_BYTES)
            hashes[b] = key

        cached_results = [None] * B
        uncached_idx = []
        for i, key in enumerate(hashes):
            cached = self.fitness_cache.get(key)
            if cached is not None:
                cached_results[i] = cached
            else:
                uncached_idx.append(i)

        if uncached_idx:
            sub_rotas = rotas_batch[uncached_idx]
            sub_vels = vels_batch[uncached_idx]
            fits, dists, custos, recs = avaliar_lote_numba(
                sub_rotas, sub_vels,
                self.dist_matrix, self.az_matrix,
                self.v_eff_table, self.vento_idx_map,
                np.float32(self.drone.autonomia_real),
                np.int32(self.drone.tempo_pouso_seg),
                np.int32(self.vento_array.shape[0])
            )
            for i_local, idx in enumerate(uncached_idx):
                fit = float(fits[i_local])
                res = (fit, (individuos[idx][0], individuos[idx][1], float(dists[i_local]), float(custos[i_local]), int(recs[i_local]), []))
                self.fitness_cache.set(hashes[idx], res)
                cached_results[idx] = res

        return cached_results

    def avaliar_populacao(self, populacao):
        batch_size = max(1, len(populacao) // max(1, self.n_workers))
        resultados = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = []
            for i in range(0, len(populacao), batch_size):
                batch = populacao[i:i + batch_size]
                futures.append(ex.submit(self._avaliar_lote, batch))
            for f in futures:
                resultados.extend(f.result())
        return resultados

    def selecionar_pais(self, avaliacoes, gen):
        k = 5 if gen < self.n_gen * 0.25 else 3
        contestants = random.sample(avaliacoes, min(k, len(avaliacoes)))
        return max(contestants, key=lambda x: x[0])[1]

    def executar(self, verbose=True):
        populacao = self.inicializar_populacao()
        start_time = time.time()
        avaliacoes = self.avaliar_populacao(populacao)
        avaliacoes.sort(reverse=True, key=lambda x: x[0])
        melhor_global = avaliacoes[0]
        estagnado = 0

        if verbose:
            print("=== ALGORITMO GENÉTICO INICIADO ===")

        for gen in range(self.n_gen):
            t = gen / (self.n_gen - 1) if self.n_gen > 1 else 1.0
            taxa_mut = self.taxa_mut_inicial + t * (self.taxa_mut_final - self.taxa_mut_inicial)
            elite_n = max(1, int(self.elitismo * self.n_pop))
            nova_pop = [avaliacoes[i][1][:2] for i in range(elite_n)]

            while len(nova_pop) < self.n_pop:
                p1 = self.selecionar_pais(avaliacoes, gen)
                p2 = self.selecionar_pais(avaliacoes, gen)
                child_rota = self.pmx_crossover(p1[0], p2[0])
                child_vels = self.crossover_velocidades(p1[1], p2[1])
                child_rota = self.mutacao_inversao(child_rota, taxa_mut)
                child_vels = self.mutacao_velocidades(child_vels, taxa_mut)
                nova_pop.append((child_rota, child_vels))

            populacao = nova_pop
            avaliacoes = self.avaliar_populacao(populacao)
            avaliacoes.sort(reverse=True, key=lambda x: x[0])

            if avaliacoes[0][0] > melhor_global[0]:
                melhor_global = avaliacoes[0]
                estagnado = 0
            else:
                estagnado += 1

            if estagnado > 40:
                if verbose:
                    print(f"\n>>> REINÍCIO RÁPIDO (G{gen+1})")
                elite = [x[1][:2] for x in avaliacoes[:max(1, int(0.05 * self.n_pop))]]
                nova_pop = elite[:]
                while len(nova_pop) < self.n_pop:
                    perm = random.sample(self.base, len(self.base))
                    rota = [1] + perm + [1]
                    rota = self.mutacao_inversao(rota, 0.75)
                    vels = random.choices(self.drone.velocidades, k=len(rota) - 1)
                    nova_pop.append((rota, vels))
                populacao = nova_pop
                avaliacoes = self.avaliar_populacao(populacao)
                avaliacoes.sort(reverse=True, key=lambda x: x[0])
                estagnado = 0

            if estagnado > self.max_stagnation:
                if verbose:
                    print(f"\nEstagnação crítica (G{gen+1}) — interrompendo.")
                break

            if verbose and (gen % self.log_interval == 0 or gen == self.n_gen - 1):
                best = avaliacoes[0]
                elapsed = time.time() - start_time
                print(f"G{gen+1:4d} | Fit: {best[0]:.5f} | Dist: {best[1][2]:.1f}km | T:~{best[1][3]/40:.0f}min | Rec: {best[1][4]} | Stag: {estagnado} | Elap: {elapsed:.1f}s")

        if verbose:
            print("=== FIM DO ALGORITMO GENÉTICO ===")
        return melhor_global[1], melhor_global[0]

    # operadores
    def pmx_crossover(self, p1, p2):
        size = len(p1)
        if size < 4:
            return p1[:]
        a, b = sorted(random.sample(range(1, size - 1), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        mapping = {p2[i]: p1[i] for i in range(a, b) if p2[i] != p1[i]}
        used = set(child[a:b])
        for i in range(1, size - 1):
            if a <= i < b:
                continue
            gene = p2[i]
            while gene in mapping and gene not in used:
                gene = mapping[gene]
            if gene not in used:
                child[i] = gene
                used.add(gene)
        for i in range(1, size - 1):
            if child[i] is None:
                for g in p1[1:-1]:
                    if g not in used:
                        child[i] = g
                        used.add(g)
                        break
        child[0] = child[-1] = 1
        return child

    def crossover_velocidades(self, v1, v2):
        if len(v1) != len(v2):
            return v1[:]
        a, b = sorted(random.sample(range(len(v1)), 2))
        return v1[:a] + v2[a:b] + v1[b:]

    def mutacao_inversao(self, rota, taxa):
        if random.random() > taxa:
            return rota
        i, j = sorted(random.sample(range(1, len(rota) - 1), 2))
        return rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]

    def mutacao_velocidades(self, vels, taxa):
        v = vels[:]
        for i in range(len(v)):
            if random.random() < taxa:
                v[i] = random.choice(self.drone.velocidades)
        return v
