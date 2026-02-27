"""
Implementación del algoritmo Softmax para bandits de k brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm


class Softmax(Algorithm):
    def __init__(self, k: int, temperature: float):
        super().__init__(k)
        self.temperature = temperature
        self.counts = np.zeros(k)
        self.q_values = np.zeros(k)

    def reset(self):
        self.counts = np.zeros(self.k)
        self.q_values = np.zeros(self.k)

    def select_arm(self) -> int:
        # Calcular las preferencias de Gibbs (exponencial de Q / temperatura)
        # Se resta el máximo para estabilidad numérica (evitar overflow)
        z = self.q_values / self.temperature
        z_stable = z - np.max(z)
        exp_values = np.exp(z_stable)

        # Calcular probabilidades
        probabilities = exp_values / np.sum(exp_values)

        # Seleccionar brazo basado en las probabilidades
        return np.random.choice(self.k, p=probabilities)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.q_values[arm]
        # Actualización incremental de la media
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.q_values[arm] = new_value

    def __str__(self):
        return f"Softmax(tau={self.temperature})"


__all__ = ["Softmax"]
