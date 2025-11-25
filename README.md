# Documentação do Projeto - Drone Genetic Algorithm

## Visão Geral

Este projeto implementa um **Algoritmo Genético** para otimizar rotas de drones, considerando fatores como distância, tempo de voo, velocidade do vento e necessidade de recargas. O sistema lê dados de coordenadas e velocidade do vento, executa a otimização e gera uma rota otimizada em CSV.

## Objetivos

- Otimizar rotas de drones usando algoritmo genético
- Minimizar distância total percorrida
- Considerar fatores ambientais (vento)
- Gerenciar necessidade de recargas da bateria
- Gerar rotas em formato CSV para análise

## Estrutura do Projeto

```
drone-genetic-algorithm/
├── src/
│   ├── constants.py
│   ├── coordenadas.py
│   ├── drone.py
│   ├── evaluator.py
│   ├── ga.py
│   ├── io_csv.py
│   ├── utils.py
│   ├── v_eff.py
│   └── vento.py
│
├── tests/
│   ├── conftest.py
│   ├── test_constants.py
│   ├── test_coordenadas.py
│   ├── test_drone.py
│   ├── test_vento.py
│   ├── test_v_eff.py
│   ├── test_evaluator.py
│   ├── test_utils.py
│   ├── test_io_csv.py
│   └── test_ga.py
│
├── run.py
├── coordenadas.csv
├── vento.csv
└── requirements.txt
```

## Componentes Principais

### `src/coordenadas.py`
Gerencia dados de coordenadas geográficas dos pontos de interesse.
- Carrega o arquivo CSV `coordenadas.csv` que contém ceps, latitudes e logitudes, no padrão abaixo:
    ```csv
    cep,longitude,latitude
    82821020,-49.2160678044742,-25.4233146347775
    81350686,-49.3400481020638,-25.4936598469491
    ```
  
- Calcula distâncias entre pontos
- Valida dados de entrada

### `src/drone.py`
Define as características e comportamento do drone.
- Velocidade máxima
- Capacidade de bateria
- Consumo de energia

### `src/vento.py`
Gerencia dados de velocidade e direção do vento.
- Carrega dados de vento por hora do arwuivo CSV `vento.csv`, que apresenta o formato:
    ```csv
    hora,velocidade_media,direcao
    06:00,5.2,270
    06:30,5.1,265
    ```
- Fornece velocidade efetiva considerando o vento

### `src/ga.py`
Implementa o **Algoritmo Genético**.
- População inicial
- Seleção
- Crossover e mutação
- Critério de parada

### `src/evaluator.py`
Avalia a qualidade de cada rota (fitness).
- Distância total
- Tempo de voo
- Número de recargas
- Velocidade efetiva com vento

### `src/utils.py`
Funções utilitárias gerais.
- Cálculos matemáticos
- Manipulação de dados

### `src/io_csv.py`
Leitura e escrita de dados em CSV.
- Carregamento de dados
- Escrita de resultados



## Como executar

1. **Crie um ambiente virtual (recomendado)**

    - Linux/Mac:
        ```bash
        python -m venv venv
        ```

    - Windows:
        ```bash
        python -m venv venv
        ```

2. **Ative o ambiente virtual:**

    - Linux/Mac:
        ```bash
        source venv/bin/activate
        ```

    - Windows:
        ```bash
        venv\Scripts\activate
        ```



3. **Instalar as dependências**

    Execute o cmando abaixo para instalar as dependências necessárias para o projeto: 

    ```bash
    pip install -r requirements.txt
    ```


4. **Executar o script principal**

    Na raiz do projeto, execute o comando:

    ```bash
    python run.py
    ```

5. **Verifique o output** 
   Após o término da exeução, será mostrado o log

   ```bash
   ======================================================================
   MELHOR SOLUÇÃO ENCONTRADA
   ======================================================================
   Fitness: 0.90643
   Distância: 412.40 km
   Tempo: ~425 min
   Recargas: 5
   ======================================================================

   Arquivo gerado: rota.csv
   Distância total: 412.40 km
   ```

   Verifique a rota gerada na raiz no projeto, no arquivo `rota.csv`. O arquivo possui a ordem para a rota otimizada e possui o formato abaixo:
    ```csv
    CEP_inicial,Latitude_inicial,Longitude_inicial,Dia_do_voo,Hora_inicial,Velocidade,CEP_final,Latitude_final,Longitude_final,Pouso,Hora_final
    82821020,-25.4233146347775,-49.2160678044742,1,06:00,92,82821016,-25.4270763750322,-49.209505500185,NÃO,06:00
    ```

### Fluxo de Execução

1. **Carregamento de dados**: Lê `coordenadas.csv` e `vento.csv`
2. **Inicialização**: Cria instâncias de Drone, Coordenadas e Vento
3. **Algoritmo Genético**: Executa otimização iterativa
4. **Reavaliação**: Valida melhor solução encontrada
5. **Saída**: Gera `rota.csv` com a rota otimizada

## Testes

Testes disponíveis:
- `test_coordenadas.py` - Validação de coordenadas
- `test_drone.py` - Comportamento do drone
- `test_vento.py` - Dados de vento
- `test_evaluator.py` - Cálculo de fitness
- `test_ga.py` - Algoritmo genético
- `test_io_csv.py` - Leitura/escrita de CSV
- `test_utils.py` - Funções utilitárias
- `test_v_eff.py` - Velocidade efetiva

Execute os testes unitários com pytest, medindo a cobertura:

```bash
pytest --cov=src --cov-report term-missing
```

Ou execute testes específicos:

```bash
pytest tests/test_coordenadas.py -v
pytest tests/test_drone.py -v
pytest tests/test_evaluator.py -v
```

**IMPORTANTE:** Para o funcionamento adequado, é necessário que todas as dependências estrjam corretamente instaladas. Portanto, certifique-se de ter executado o comando `pip install -r requirements.txt` antes de rodar os testes.


## Visualização

Caso queira visualizar a rota otimizada do arquivo `rota.csv` de maneira gráfica, execute o comando abaixo:

```bash
python plot.py
```

Será gerada uma guia com o gráfico das rotas.


## Configuração

Caso queira executar o código com parâmetros diferentes do padrão (numero diferente de gerações, população, taxa de mutação, etc) edite o arquivo `constants.py` localizado no diretório `/src`.

## 🧬 Algoritmo Genético

Embora existam vários operadores possíveis em Algoritmos Genéticos, este projeto utiliza **somente os métodos que mostraram maior estabilidade, performance e adequação ao problema de rotas com velocidades associadas**. Abaixo está um resumo **do que realmente foi implementado** no código e **por que essas escolhas foram feitas**:

### Operadores Genéticos Utilizados no Projeto

#### Seleção - Torneio

O código utiliza seleção por torneio (k=5 no começo, k=3 depois).

**Motivo da escolha:**

- É simples, rápido e funciona bem mesmo quando os valores de fitness têm escalas diferentes.
- Mantém pressão seletiva controlada, evitando convergência prematura.
- Menos sensível a problemas de normalização do fitness, ao contrário da roleta.

#### Crossover para rotas - PMX (Partially-Mapped Crossover)

Implementado em `pmx_crossover`.

**Motivo da escolha:**

- Preserva estrutura de permutação, essencial para o problema do TSP (não cria cidades duplicadas).
- Mantém blocos de rota estáveis entre pais, o que ajuda a preservar subrotas boas.
- Mais robusto que OX para cruzamentos onde os pais têm padrões muito diferentes.


#### Crossover para velocidades - Segment Swap

Implementado em `crossover_velocidades`. Troca de um segmento entre os vetores de velocidade dos pais.

**Motivo da escolha:**

- É simples e coerente com a rota (mantém tamanho e ordem).
- Mantém alguma herança entre pais sem impor demasiada correlação com a rota — importante porque velocidade é um parâmetro contínuo/discreto independente do caminho.

#### Mutação de rota - Inversão

Implementado em `mutacao_inversao`

**Motivo da escolha:**

- É um dos melhores operadores de mutação para problemas do tipo TSP/Tour.
- Tende a reduzir distância ao remover cruzamentos na rota.
- Baixa probabilidade de gerar soluções totalmente aleatórias — mantém estabilidade.


#### Mutação de velocidades - Alteração pontual

Implementado em `mutacao_velocidades`

**Motivo da escolha:**

- Permite explorar diferentes velocidades sem modificar a estrutura da rota.
- Controle simples e direto via taxa de mutação.
- Flexível para ajustar consumo e tempo conforme o vento.

#### Critério de Parada - Gerações + Estagnação

- Número máximo de gerações
- Estagnação da população com reinício rápido

**Motivo da escolha:**

- Evita desperdício computacional quando o algoritmo deixa de melhorar.
- Permite explorar mais o espaço de busca quando preso em mínimos locais.
- Combinação simples e eficiente para problemas complexos como roteamento com vento e recarga.

## Conceitos Técnicos

### Velocidade Efetiva
A velocidade efetiva do drone é calculada considerando:
- Velocidade base do drone
- Velocidade e direção do vento
- Ângulo entre trajetória e vento

### Fitness
A função de fitness minimiza:
- Distância total percorrida
- Tempo de voo
- Número de recargas necessárias

### Recargas
O drone precisa recarregar quando:
- Bateria atinge limite crítico
- Distância restante > autonomia

## Troubleshooting

**Arquivo não encontrado:**
- Certifique-se que `coordenadas.csv` e `vento.csv` existem no diretório raiz

**Erros de performance:**
- Ajuste tamanho da população e gerações em `src/constants.py`
- Reduza número de pontos para testes iniciais

**Resultados inconsistentes:**
- O GA é estocástico; execute múltiplas vezes
- Ajuste parâmetros de seleção e mutação

**Testes falhando:**
- Verifique se as dependências estão instaladas: `pip install -r requirements.txt`
- Execute em um ambiente Python 3.8+
- Limpe cache: `pytest --cache-clear`

## Estrutura de um Teste

Exemplo de teste unitário:

```python
import pytest
from src import Drone

def test_drone_velocidade():
    drone = Drone()
    assert drone.velocidade_maxima > 0
    assert drone.autonomia > 0

def test_drone_bateria():
    drone = Drone()
    assert drone.bateria_maxima > 0
```

## Referências

HOLLAND, John H. *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence.* Ann Arbor: University of Michigan Press, 1975.

LAWLER, Eugene L. et al. (Org.). *The Traveling Salesman Problem: A Guided Tour of Combinatorial Optimization.* New York: Wiley, 1985.

GOLDEN, Bruce L.; RAGHAVAN, S.; WASIL, Edward A. (Org.). *The Vehicle Routing Problem: Latest Advances and New Challenges.* New York: Springer, 2008.

## Licença

Este código está sob a licença MIT. Você pode usar, copiar, modificar e distribuir este projeto livremente, desde que mantenha o aviso de copyright e a licença incluídos. Para mais detalhes, consulte o arquivo LICENSE.

