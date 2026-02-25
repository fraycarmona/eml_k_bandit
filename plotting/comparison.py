"""
Herramientas para comparación automática de hiperparámetros.

Author: Adrián Rodríguez Carmona
Date: 2026-02-25
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from arms import Bandit
    from algorithms import Algorithm

# ── Importar run_experiment desde el nivel superior ──
import sys
sys.path.append('..')
from main import run_experiment  # o donde esté definida


def compute_scores(rewards: np.ndarray, criterion: str = "final_100") -> np.ndarray:
    """
    Calcula un score por algoritmo a partir de la matriz de recompensas.

    Parameters
    ----------
    rewards   : ndarray (n_algorithms, steps)
    criterion : "final_100"  → media de los últimos 100 pasos
                "cumulative" → suma total de la curva

    Returns
    -------
    scores : ndarray (n_algorithms,)
    
    Examples
    --------
    >>> rewards = np.array([[1.2, 1.5, 1.8], [0.9, 1.1, 1.3]])
    >>> compute_scores(rewards, "cumulative")
    array([4.5, 3.3])
    """
    if criterion == "final_100":
        window = min(100, rewards.shape[1])  # evitar error si steps < 100
        return rewards[:, -window:].mean(axis=1)
    if criterion == "cumulative":
        return rewards.sum(axis=1)
    raise ValueError(f"Criterio desconocido: '{criterion}'. Usa 'final_100' o 'cumulative'.")


def select_best(
    bandit: "Bandit",
    algorithms: List["Algorithm"],
    steps: int,
    runs: int,
    criterion: str,
) -> Tuple[float, np.ndarray]:
    """
    Ejecuta el experimento y devuelve (parámetro_óptimo, curva_de_recompensas).
    
    Notes
    -----
    Asume que los algoritmos tienen un atributo `.temperature` (Softmax)
    o `.epsilon` (ε-greedy). Extender para UCB si es necesario.
    """
    from main import run_experiment  # importación local para evitar circular
    
    rewards, _ = run_experiment(bandit, algorithms, steps, runs)
    scores     = compute_scores(rewards, criterion)
    best_idx   = int(np.argmax(scores))
    
    # Detectar tipo de parámetro
    param = getattr(algorithms[best_idx], 'temperature', None) or \
            getattr(algorithms[best_idx], 'epsilon', None)
    
    return param, rewards[best_idx]


def plot_best_temperatures(
    best_results: Dict[str, Tuple[float, np.ndarray]],
    steps: int,
    save_path: str = "best_temperatures_comparison.png",
) -> None:
    """
    Dibuja una curva por tipo de brazo, etiquetada con su mejor τ.
    """
    fig, ax    = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(1, steps + 1)

    for arm_name, (best_param, curve) in best_results.items():
        ax.plot(time_steps, curve, label=f"{arm_name}  (τ = {best_param:.2f})", linewidth=2)

    ax.set_xlabel("Pasos de tiempo", fontsize=13)
    ax.set_ylabel("Recompensa promedio", fontsize=13)
    ax.set_title("Mejor temperatura Softmax por tipo de brazo", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()


def compare_best_temperatures(
    bandits: Dict[str, "Bandit"],
    algorithms: List["Algorithm"],
    steps: int,
    runs: int,
    seed: int = 42,
    criterion: str = "final_100",
) -> Dict[str, Tuple[float, np.ndarray]]:
    """
    Compara la mejor τ Softmax para cada tipo de brazo en un único gráfico.
    
    Parameters
    ----------
    bandits    : dict {nombre: Bandit}
    algorithms : list de algoritmos con distinto hiperparámetro
    steps      : pasos de tiempo
    runs       : repeticiones del experimento
    seed       : semilla reproducible
    criterion  : "final_100" o "cumulative"
    
    Returns
    -------
    best_results : dict {nombre_brazo: (param_óptimo, curva)}
    
    Examples
    --------
    >>> from arms import ArmNormal, Bandit
    >>> bandits = {"Normal": Bandit(arms=ArmNormal.generate_arms(10))}
    >>> algorithms = [Softmax(k=10, temperature=t) for t in [0.1, 0.5, 1.0]]
    >>> results = compare_best_temperatures(bandits, algorithms, 1000, 500)
    """
    np.random.seed(seed)

    best_results = {
        arm_name: select_best(bandit, algorithms, steps, runs, criterion)
        for arm_name, bandit in bandits.items()
    }

    for arm_name, (best_param, _) in best_results.items():
        print(f"[{arm_name:10s}]  mejor τ = {best_param:.2f}")

    plot_best_temperatures(best_results, steps)
    return best_results
