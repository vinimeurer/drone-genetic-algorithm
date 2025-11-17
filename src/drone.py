# src/drone.py
from typing import List
from .constants import DRONE_VELOCIDADES

class Drone:
    def __init__(self):
        self.autonomia_base = 5000.0
        self.fator_curitiba = 0.93
        self.autonomia_real = self.autonomia_base * self.fator_curitiba
        self.velocidades = DRONE_VELOCIDADES
        self.tempo_pouso_seg = 72
        self.vel_to_idx = {v: i for i, v in enumerate(self.velocidades)}
