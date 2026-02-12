"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax (Boltzmann exploration) para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm


class Softmax(Algorithm):
    def __init__(self, k: int, temperature: float = 0.1):
        """
        Inicializa el algoritmo softmax.

        :param k: Número de brazos.
        :param temperature: Temperatura (tau) para regular exploración/explotación.
                            Valores altos exploran más; valores bajos explotan más.
        """
        assert temperature > 0, "La temperatura debe ser mayor que 0."
        super().__init__(k)
        self.temperature = temperature

    def select_arm(self) -> int:
        """
        Selecciona un brazo con probabilidad proporcional a exp(Q(a)/tau).

        :return: Índice del brazo seleccionado.
        """
        # Estabilización numérica: restar el máximo evita overflow en la exponencial.
        preferences = (self.values - np.max(self.values)) / self.temperature
        probabilities = np.exp(preferences)
        probabilities /= np.sum(probabilities)

        chosen_arm = np.random.choice(self.k, p=probabilities)
        return chosen_arm

    def __str__(self):
        return f"Softmax(temperature={self.temperature})"
