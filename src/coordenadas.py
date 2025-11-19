# src/coordenadas.py
import math
from typing import Dict
import pandas as pd
import numpy as np

class Coordenadas:
    def __init__(self, arquivo_csv: str):
        df = pd.read_csv(arquivo_csv).reset_index(drop=True)
        df["ID"] = list(range(1, len(df) + 1))
        self.coordenadas = {
            int(row["ID"]): {"cep": row.get("cep", ""), "lat": float(row["latitude"]), "lon": float(row["longitude"])}
            for _, row in df.iterrows()
        }
        self.ids = sorted(self.coordenadas.keys())
        self.idx_map = {id_: i for i, id_ in enumerate(self.ids)}
        self._build_matrices()

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _azimute(lat1, lon1, lat2, lon2) -> float:
        dlon = math.radians(lon2 - lon1)
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        az = math.degrees(math.atan2(x, y))
        return (az + 360) % 360

    def _build_matrices(self):
        n = len(self.ids)
        self.dist_matrix = np.zeros((n, n), dtype=np.float32)
        self.az_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            id_i = self.ids[i]
            p_i = self.coordenadas[id_i]
            for j in range(i + 1, n):
                id_j = self.ids[j]
                p_j = self.coordenadas[id_j]
                d = self._haversine(p_i["lat"], p_i["lon"], p_j["lat"], p_j["lon"])
                az = self._azimute(p_i["lat"], p_i["lon"], p_j["lat"], p_j["lon"])
                self.dist_matrix[i, j] = self.dist_matrix[j, i] = d
                self.az_matrix[i, j] = az
                self.az_matrix[j, i] = (az + 180) % 360

    def distancia(self, id1: int, id2: int) -> float:
        return float(self.dist_matrix[self.idx_map[id1], self.idx_map[id2]])

    def azimute(self, id1: int, id2: int) -> float:
        return float(self.az_matrix[self.idx_map[id1], self.idx_map[id2]])
