# src/__init__.py
"""
Pacote AG Drone - refatorado v5.1
"""
from .constants import *
from .utils import LRUCache, hash_individuo
from .coordenadas import Coordenadas
from .vento import Vento
from .drone import Drone
from .v_eff import build_v_eff_table
from .evaluator import avaliar_lote_numba
from .ga import GeneticAlgorithm
from .io_csv import reavaliar_preciso, gerar_csv_final

__all__ = [
    "Coordenadas", "Vento", "Drone", "GeneticAlgorithm",
    "build_v_eff_table", "avaliar_lote_numba",
    "reavaliar_preciso", "gerar_csv_final",
    "LRUCache", "hash_individuo"
]
