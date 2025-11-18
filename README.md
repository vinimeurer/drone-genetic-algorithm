# 📦 **GA Drone Routing — Otimização de Rotas com Algoritmo Genético**

Este projeto implementa um **Algoritmo Genético (AG)** de alta performance para otimizar rotas de drones considerando:

* Distâncias geográficas reais (Haversine)
* Azimutes entre pontos
* Condições de vento reais por dia e hora
* Diferentes velocidades de voo
* Autonomia, tempo de pouso e efeitos aerodinâmicos no cálculo de velocidade efetiva
* Penalidades por pousos forçados e limitações de horário de operação

O objetivo é encontrar a **melhor rota possível**, minimizando custo total e penalidades, e gerar um **CSV final detalhando o plano de voo**. Mais detahes sobre o funcionamento do projeto estão descritos no [arquivo PDF](drone-ga.pdf).

---

# 📁 Estrutura do Projeto

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


# 🚀 **Como funciona o projeto**

## 🔹 **1. Leitura dos dados**

* `coordenadas.csv` contém CEP, latitude e longitude dos pontos a serem visitados.
* `vento.csv` contém velocidade e direção do vento por dia e hora.

A classe `Coordenadas`(`src/coordenadas.py`) constrói:

* Matriz de distâncias Haversine
* Matriz de azimutes entre pares de pontos



A classe `Vento`(`src/vento.py`):

* Gera uma matriz `[dias × horas × (velocidade, direção)]`


## 🔹 **2. Modelagem do drone (`src/drone.py`)**

A classe `Drone` define:

* Autonomia
* Fator de correção local
* Velocidades disponíveis
* Tempo de pouso




## 🔹 **3. Tabela de velocidade efetiva (vento + direção)**

O módulo `v_eff.py` computa, via **Numba**, o impacto do vento:

> velocidade efetiva = velocidade própria + componente do vento

Gerando uma tabela indexada por:

* velocidade
* faixa de azimutes
* dia × hora


## 🔹 **4. Avaliação de rotas (`src/evaluator.py`)**

A função acelerada `avaliar_lote_numba` calcula:

* Distância total
* Penalidades
* Autonomia e pousos forçados
* Tempo total
* Fitness da solução

## 🔹 **5. Algoritmo Genético (`src/ga.py`)**

* Geração inicial de população
* Crossover PMX para rotas
* Mutação por inversão
* Mutação de velocidades
* Elitismo
* Reavaliação com cache LRU
* Restart automático
* Execução paralela com ThreadPool

---

## 🔹 **6. Reavaliação precisa & geração do CSV final**

Após o AG encontrar a melhor solução, ocorre:

1. **Reavaliação precisa** sem discretizar vento
   (`reavaliar_preciso`)

2. **Geração da rota final detalhada**
   (`gerar_csv_final`) contendo:

   * CEP início / fim
   * lat/lon
   * dia e hora
   * velocidade
   * tempo de voo
   * marcação de pouso forçado

Arquivo gerado: `rota.csv`.

---

# 🛠️ Instalação

### 1. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

---

# ▶️ Execução

Basta executar:

```bash
python run.py
```

Certifique-se de que os arquivos:

* `coordenadas.csv`
* `vento.csv`

estão no diretório raiz do projeto.

---

# 🧪 Testes

A suíte de testes abrange absolutamente todas as partes do sistema.

Rodar com:

```bash
pytest -v
```

ou, para medir cobertura:

```bash
pytest --cov=src --cov-report term-missing
```

---

# 📤 Formato dos arquivos de entrada

## **coordenadas.csv**

```
cep,latitude,longitude
80000-000,-25.43,-49.27
...
```

## **vento.csv**

```
dia,hora,vel_kmh,direcao_deg
1,0,3.5,270
1,1,4.1,265
...
```

---

# 📄 Saída gerada

O arquivo `rota.csv` contém:

| Coluna            | Descrição                     |
| ----------------- | ----------------------------- |
| CEP_inicial       | CEP do ponto de origem        |
| Latitude_inicial  | Latitude do ponto inicial     |
| Longitude_inicial | Longitude do ponto inicial    |
| Dia_do_voo        | Dia do plano de voo           |
| Hora_inicial      | Hora de saída                 |
| Velocidade        | Velocidade do drone no trecho |
| CEP_final         | CEP destino                   |
| Latitude_final    | Latitude do destino           |
| Longitude_final   | Longitude do destino          |
| Pouso             | Indica se houve pouso forçado |
| Hora_final        | Hora estimada de chegada      |

---

# 🤖 Tecnologias utilizadas

* **Python 3.10+**
* **NumPy**
* **Pandas**
* **Numba (aceleração JIT)**
* **PyTest**
* **ThreadPoolExecutor**
* Estratégias avançadas de AG (PMX, elitismo, reinício, cache LRU)

---

# 📌 Objetivo Científico / Prático

Este projeto pode ser aplicado a:

* Logística de entregas com drones
* Simulação de rotas sensíveis ao clima
* Otimização NP-Difícil em grafos completos
* Estudos de impacto aerodinâmico por vento em veículos aéreos autônomos

---

# 📬 Contato

Caso precise de auxílio, otimização adicional ou documentação expandida, posso gerar:

* Diagramas UML
* Fluxos de execução
* Documentação API
* Tutoriais de uso

Basta solicitar!
