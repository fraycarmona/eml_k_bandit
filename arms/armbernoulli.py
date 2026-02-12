"""
Module: arms/armbernoulli.py
Description: Brazo Bernoulli para recompensas binarias {0,1}.
"""

import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        :param p: Probabilidad de éxito/recompensa 1.
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."
        self.p = p

    def pull(self):
        """
        Devuelve 1 con probabilidad p y 0 con probabilidad (1-p).
        """
        return int(np.random.binomial(n=1, p=self.p))

    def get_expected_value(self) -> float:
        """
        Valor esperado de Bernoulli(p): p.
        """
        return self.p

    def __str__(self):
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.05, p_max: float = 0.95):
        """
        Genera k brazos Bernoulli con probabilidades distintas en [p_min, p_max].
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Debe cumplirse 0 <= p_min < p_max <= 1."

        p_values = np.random.uniform(p_min, p_max, size=k)
        p_values = np.round(p_values, 3)

        return [ArmBernoulli(float(p)) for p in p_values]
