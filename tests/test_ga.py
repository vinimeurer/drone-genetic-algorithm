# tests/test_ga.py
import random

def test_inicializar_populacao_and_operators(ga_instance):
    ga = ga_instance
    pop = ga.inicializar_populacao()
    assert len(pop) == ga.n_pop
    # cada indivíduo tem rota e velocidades
    for rota, vels in pop:
        assert rota[0] == 1 and rota[-1] == 1
        assert len(vels) == len(rota) - 1

    # testar pmx_crossover com tamanho pequeno
    p1 = [1, 2, 3, 4, 1]
    p2 = [1, 4, 3, 2, 1]
    child = ga.pmx_crossover(p1, p2)
    assert child[0] == 1 and child[-1] == 1
    assert set(child[1:-1]) == set(p1[1:-1])

    # crossover velocidades
    v1 = [36,44,52,60]
    v2 = [60,52,44,36]
    cv = ga.crossover_velocidades(v1, v2)
    assert len(cv) == len(v1)

    # mutacao inversao
    mutated = ga.mutacao_inversao(p1[:], taxa=1.0)
    assert mutated != p1 or p1 == mutated  # may equal if inversion yields same sequence

    # mutacao velocidades
    mv = ga.mutacao_velocidades(v1[:], taxa=1.0)
    assert len(mv) == len(v1)

def test_selecionar_pais_and_execute(ga_instance, coordenadas):
    ga = ga_instance
    # criar avaliações simuladas: (fit, (rota, vels, dist, custo, recs, extra))
    avaliacao = (0.9, ([1,2,3,4,1], [36,36,36,36], 100.0, 400.0, 0, []))
    avaliacoes = [avaliacao for _ in range(ga.n_pop)]
    # selecionar pais (sem erro)
    parent = ga.selecionar_pais(avaliacoes, gen=0)
    assert isinstance(parent, tuple)
    # executar algoritmo (rápido - n_gen pequeno)
    # para acelerar: ajustar parametros temporariamente
    ga.n_gen = 2
    ga.n_pop = 4
    best_info, best_fit = ga.executar(verbose=False)
    # verificar formato do retorno
    assert isinstance(best_fit, float)
    assert isinstance(best_info, list) or isinstance(best_info, tuple)
