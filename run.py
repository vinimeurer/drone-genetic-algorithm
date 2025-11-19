# run_ga.py
import os
from src import Coordenadas, Vento, Drone, GeneticAlgorithm, reavaliar_preciso, gerar_csv_final

def main():
    arquivo_coord = "coordenadas.csv"
    arquivo_vento = "vento.csv"

    if not os.path.exists(arquivo_coord):
        raise FileNotFoundError(f"Arquivo de coordenadas não encontrado: {arquivo_coord}")
    if not os.path.exists(arquivo_vento):
        raise FileNotFoundError(f"Arquivo de vento não encontrado: {arquivo_vento}")

    coord = Coordenadas(arquivo_coord)
    vento = Vento(arquivo_vento)
    drone = Drone()

    ga = GeneticAlgorithm(coord, vento, drone)
    melhor_info, melhor_fit = ga.executar()

    print("\n" + "=" * 70)
    print("MELHOR SOLUÇÃO ENCONTRADA")
    print("=" * 70)
    print(f"Fitness: {melhor_fit:.5f}")
    print(f"Distância: {melhor_info[2]:.2f} km")
    print(f"Tempo: ~{melhor_info[3] / 40:.0f} min")
    print(f"Recargas: {melhor_info[4]}")
    print("=" * 70)

    #print("Reavaliando melhor rota...")

    fit_preciso, info_precisa = reavaliar_preciso(melhor_info, coord, vento, drone)

    # print("\n" + "=" * 70)
    # print("RESULTADO FINAL")
    # print("=" * 70)
    # print(f"Fitness: {fit_preciso:.5f}")
    # print(f"Distância: {info_precisa[2]:.2f} km")
    # print(f"Tempo: ~{info_precisa[3] / 40:.0f} min")
    # print(f"Recargas: {info_precisa[4]}")
    # print("=" * 70)

    gerar_csv_final(info_precisa, coord, vento, "rota.csv")

if __name__ == "__main__":
    main()
