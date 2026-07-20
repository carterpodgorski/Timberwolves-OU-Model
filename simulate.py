"""
Monte Carlo simulation of Timberwolves point differentials under the
fitted Ornstein-Uhlenbeck model.

Uses the exact discrete-time transition (no Euler discretization error).
"""

from __future__ import annotations

import numpy as np

from ou_model import OUParams


def simulate_paths(params: OUParams,
                   x0: float,
                   n_steps: int,
                   n_paths: int = 10_000,
                   dt: float = 1.0,
                   seed: int | None = 42) -> np.ndarray:
    """
    Returns an (n_paths, n_steps) array of simulated point differentials.
    Each path starts from x0 and evolves for n_steps games.
    """
    rng = np.random.default_rng(seed)
    e = np.exp(-params.theta * dt)
    var_eps = params.sigma**2 * (1.0 - np.exp(-2.0 * params.theta * dt)) / (2.0 * params.theta)
    sd_eps = np.sqrt(var_eps)

    paths = np.empty((n_paths, n_steps))
    cur = np.full(n_paths, float(x0))
    for t in range(n_steps):
        cur = cur * e + params.mu * (1.0 - e) + rng.normal(0, sd_eps, size=n_paths)
        paths[:, t] = cur
    return paths


if __name__ == "__main__":
    p = OUParams(mu=5.0, theta=0.35, sigma=11.0, method="demo")
    paths = simulate_paths(p, x0=2.0, n_steps=30, n_paths=5_000)
    print("paths shape:", paths.shape)
    print("mean final diff:", paths[:, -1].mean())
