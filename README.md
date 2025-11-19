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
- Carrega coordenadas de CSV
- Calcula distâncias entre pontos
- Valida dados de entrada

### `src/drone.py`
Define as características e comportamento do drone.
- Velocidade máxima
- Capacidade de bateria
- Consumo de energia

### `src/vento.py`
Gerencia dados de velocidade e direção do vento.
- Carrega dados de vento por hora
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

4. **Configurar parâmetros (opcional)**

    - Caso queira executar o código com parâmetros diferentes do padrão (numero diferente de gerações, população, taxa de mutação, etc) edite o arquivo `constants.py` localizado no diretório `/src`.

5. **Executar o script principal**

    Na raiz do projeto, execute o comando:

    ```bash
    python run.py
    ```

6. **Verifique o output** 
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

   Verifique a rota gerada na raiz no projeto, no arquivo `rota.csv`.

7. **Plotar o gráfico (opcional)**

   Caso queira visualizar a rota de maneira gráfica, xecute o comando abaixo:

   ```bash
   python plot.py
   ```

   Será gerada uma guia com o gráfico das rotas.

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

### Fluxo de Execução

1. **Carregamento de dados**: Lê `coordenadas.csv` e `vento.csv`
2. **Inicialização**: Cria instâncias de Drone, Coordenadas e Vento
3. **Algoritmo Genético**: Executa otimização iterativa
4. **Reavaliação**: Valida melhor solução encontrada
5. **Saída**: Gera `rota.csv` com a rota otimizada

## 📊 Formato dos Dados

### `coordenadas.csv`
```csv
cep,longitude,latitude
82821020,-49.2160678044742,-25.4233146347775
81350686,-49.3400481020638,-25.4936598469491
```

### `vento.csv`
```csv
hora,velocidade_media,direcao
06:00,5.2,270
06:30,5.1,265
```

### `rota.csv` (Saída)
```csv
CEP_inicial,Latitude_inicial,Longitude_inicial,Dia_do_voo,Hora_inicial,Velocidade,CEP_final,Latitude_final,Longitude_final,Pouso,Hora_final
82821020,-25.4233146347775,-49.2160678044742,1,06:00,92,82821016,-25.4270763750322,-49.209505500185,NÃO,06:00
```



## 📈 Visualização

Para visualizar os dados:

```bash
python plot.py
```

## 🔧 Configuração

As constantes do projeto estão em `src/constants.py`:
- Parâmetros do algoritmo genético
- Limites do drone
- Configurações de otimização

## 🧬 Algoritmo Genético

**Operadores Genéticos:**
- **Seleção**: Seleção por torneio ou roleta
- **Crossover**: Recombinação de rotas (Ex: Order Crossover - OX)
- **Mutação**: Inversão, inserção ou troca de pontos

**Critério de Convergência:**
- Número máximo de gerações
- Estagnação da população
- Melhor fitness encontrado

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

## 🔍 Estrutura de um Teste

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

## 📚 Referências

- Algoritmos Genéticos: Holland (1975)
- Problema do Caixeiro Viajante (TSP)
- Otimização de rotas com restrições

## 💡 Dicas de Uso

1. **Tuning de Parâmetros**: Modifique `src/constants.py` para ajustar o comportamento do GA
2. **Dados Reais**: Use seus próprios dados em `coordenadas.csv` e `vento.csv`
3. **Debug**: Adicione prints em `src/ga.py` para acompanhar a evolução
4. **Performance**: Para muitos pontos, aumente `MAX_GENERACOES` e tamanho da população

## 👨‍💻 Autor

Projeto de otimização de rotas de drones usando Algoritmo Genético.

## 📄 Licença

Este projeto é fornecido como está para fins educacionais e de pesquisa.

---

**Última atualização:** Novembro 2025
