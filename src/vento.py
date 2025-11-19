# src/vento.py
import numpy as np
import pandas as pd
from typing import Dict

class Vento:
    def __init__(self, arquivo_csv: str, max_dias: int = 31):
        df = pd.read_csv(arquivo_csv)
        self.max_dias = max_dias
        self.vento_array = np.zeros((max_dias, 24, 2), dtype=np.float32)
        for _, row in df.iterrows():
            dia = int(row["dia"])
            hora = int(row["hora"])
            if 1 <= dia <= max_dias and 0 <= hora <= 23:
                vel = float(row.get("vel_kmh", 0.0))
                direc = float(row.get("direcao_deg", 0.0))
                self.vento_array[dia - 1, hora, 0] = vel
                self.vento_array[dia - 1, hora, 1] = direc

    def get_vento(self, dia: int, hora: int) -> Dict[str, float]:
        if dia < 1 or dia > self.max_dias:
            return {"vel_kmh": 0.0, "direcao_deg": 0.0}
        hora = hora % 24
        return {
            "vel_kmh": float(self.vento_array[dia - 1, hora, 0]),
            "direcao_deg": float(self.vento_array[dia - 1, hora, 1])
        }
