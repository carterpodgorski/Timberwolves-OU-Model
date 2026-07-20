"""
Win-probability accounting from Monte Carlo OU paths.

By definition of the data, an NBA team wins game g iff its point
differential for that game is > 0. So the principled mapping from a
simulated point-differential path to wins is just:

    wins_per_path = sum( simulated_diff > 0  for each game )

The 10,000-path distribution of total wins gives the win probability
distribution directly -- no separate logistic calibration needed
(and indeed any logistic fit on (diff, win) suffers from perfect
separation since diff > 0 ⇔ win).

We also report, as a useful intermediate quantity, the per-game
"theoretical" win probability under the OU model's stationary
(or per-step) distribution:

    P(diff > 0 | OU)  =  Phi( m_t / sqrt(v_t) )

where m_t, v_t are the conditional mean and variance of the next game
given the current state.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ou_model import OUParams, stationary_std


def wins_from_paths(paths: np.ndarray) -> np.ndarray:
    """Return total wins per simulated season-tail path. Shape (n_paths,)."""
    return (paths > 0).sum(axis=1)


def stationary_win_probability(params: OUParams) -> float:
    """P(X > 0) under the stationary distribution of the OU process."""
    sd = stationary_std(params)
    return float(1.0 - norm.cdf(0.0, loc=params.mu, scale=sd))


def per_step_win_probability(params: OUParams,
                             x0: float,
                             n_steps: int,
                             dt: float = 1.0) -> np.ndarray:
    """
    Closed-form P(X_t > 0 | X_0=x0) for each future game t = 1..n_steps.
    Useful as a smooth analytical complement to the Monte Carlo histogram.
    """
    t = np.arange(1, n_steps + 1) * dt
    e = np.exp(-params.theta * t)
    m = x0 * e + params.mu * (1.0 - e)
    v = params.sigma**2 * (1.0 - np.exp(-2.0 * params.theta * t)) / (2.0 * params.theta)
    return 1.0 - norm.cdf(0.0, loc=m, scale=np.sqrt(v))


if __name__ == "__main__":
    p = OUParams(mu=5.0, theta=0.4, sigma=11.0, method="demo")
    print("Stationary P(win):", stationary_win_probability(p))
    print("First 5 P(win) starting from 0:", per_step_win_probability(p, 0.0, 5))
