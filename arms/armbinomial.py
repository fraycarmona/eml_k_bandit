"""
Module: arms/armbinomial.py
Description: Brazo Binomial para recompensas discretas en {0, ..., n}.
"""

import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        :param n: Número de ensayos por tirada del brazo.
        :param p: Probabilidad de éxito en cada ensayo.
        """
        assert n > 0, "El parámetro n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Devuelve una recompensa ~ Binomial(n, p).
        """
        return int(np.random.binomial(n=self.n, p=self.p))

    def get_expected_value(self) -> float:
        """
        Valor esperado de Binomial(n, p): n * p.
        """
        return self.n * self.p

    def __str__(self):
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(
        cls,
        k: int,
        n_min: int = 1,
        n_max: int = 10,
        p_min: float = 0.05,
        p_max: float = 0.95,
    ):
        """
        Genera k brazos Binomial con n y p aleatorios.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min > 0 and n_min <= n_max, "Debe cumplirse 0 < n_min <= n_max."
        assert 0 <= p_min < p_max <= 1, "Debe cumplirse 0 <= p_min < p_max <= 1."

        n_values = np.random.randint(n_min, n_max + 1, size=k)
        p_values = np.round(np.random.uniform(p_min, p_max, size=k), 3)

        return [ArmBinomial(int(n), float(p)) for n, p in zip(n_values, p_values)]
