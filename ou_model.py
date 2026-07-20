"""
Ornstein-Uhlenbeck parameter estimation.

Continuous-time SDE:
    dX_t = theta * (mu - X_t) dt + sigma dW_t

Exact discrete-time transition with step dt:
    X_{t+dt} | X_t  ~  N( m, v )
        m = X_t * exp(-theta*dt) + mu * (1 - exp(-theta*dt))
        v = sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)

Equivalent AR(1) form:
    X_{t+dt} = a + b * X_t + e,    e ~ N(0, s^2)
        b = exp(-theta*dt)
        a = mu * (1 - b)
        s^2 = sigma^2 * (1 - b^2) / (2*theta)

We provide TWO estimators:

  * fit_ols() : closed-form regression on the AR(1) form.
                Robust to b near 0 (weak autocorrelation) and b <= 0.
                In those cases we use the stationary-variance fallback
                Var(X) = sigma^2 / (2*theta).

  * fit_mle() : maximizes the exact discrete-time conditional log-likelihood,
                bounded so theta and sigma stay positive.

Both will be reported in the project for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class OUParams:
    mu: float        # long-run mean ("true talent")
    theta: float     # mean-reversion speed (consistency)
    sigma: float     # volatility ("chaos factor")
    method: str = ""
    loglik: float = float("nan")

    def __repr__(self) -> str:
        return (f"OUParams(method={self.method!r}, "
                f"mu={self.mu:.3f}, theta={self.theta:.4f}, "
                f"sigma={self.sigma:.3f}, loglik={self.loglik:.2f})")


def _gauss_logpdf(x, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def _conditional_loglik(x: np.ndarray, mu: float, theta: float, sigma: float,
                        dt: float = 1.0) -> float:
    if theta <= 0 or sigma <= 0:
        return -np.inf
    b = np.exp(-theta * dt)
    var_eps = sigma**2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
    if var_eps <= 0:
        return -np.inf
    x_t = x[:-1]
    x_tp1 = x[1:]
    mean_eps = x_t * b + mu * (1.0 - b)
    return float(np.sum(_gauss_logpdf(x_tp1, mean_eps, var_eps)))


# ----------------------- OLS / regression estimator ------------------------- #

def fit_ols(x: np.ndarray, dt: float = 1.0) -> OUParams:
    """
    AR(1) regression  X_{t+1} = a + b X_t + e.

    Cases:
      * b in (0, 1) : standard OU-mean-reverting → invert directly.
      * b <= 0 or b ~ 0 : weak/no autocorrelation. In OU continuous time
        this corresponds to fast reversion (large theta). We fix
        theta = -ln(b_eff)/dt with b_eff = max(b, b_floor) and recover
        sigma so the *stationary* variance Var(X) matches sample variance:
            sigma = sqrt( 2 * theta * Var(X) ).
        This avoids degenerate sigma values from the AR(1) mapping when
        b is tiny.
      * b >= 1 : non-stationary. Clipped just below 1 (very slow reversion).
    """
    x = np.asarray(x, dtype=float)
    x_t = x[:-1]
    x_tp1 = x[1:]

    # Plain OLS for slope/intercept
    X = np.column_stack([np.ones_like(x_t), x_t])
    beta, *_ = np.linalg.lstsq(X, x_tp1, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    sample_mean = float(np.mean(x))
    sample_var = float(np.var(x, ddof=1))

    b_floor = 0.05  # corresponds to theta ~= 3.0 / dt
    b_cap = 0.999

    if b >= b_cap:
        b_eff = b_cap
        theta = -np.log(b_eff) / dt
        # near random-walk; fall back to mean = sample mean
        mu = sample_mean
        sigma = float(np.sqrt(max(sample_var * 2 * theta, 1e-9)))
    elif b <= b_floor:
        b_eff = b_floor
        theta = -np.log(b_eff) / dt
        mu = sample_mean
        sigma = float(np.sqrt(max(sample_var * 2 * theta, 1e-9)))
    else:
        theta = -np.log(b) / dt
        mu = a / (1.0 - b)
        resid = x_tp1 - (a + b * x_t)
        s2 = float(np.var(resid, ddof=2))
        denom = 1.0 - b**2
        sigma2 = s2 * 2.0 * theta / denom if denom > 1e-9 else s2
        sigma = float(np.sqrt(max(sigma2, 1e-9)))

    ll = _conditional_loglik(x, mu, theta, sigma, dt=dt)
    return OUParams(mu=float(mu), theta=float(theta), sigma=float(sigma),
                    method="OLS", loglik=float(ll))


# ----------------------------- MLE estimator -------------------------------- #

def _neg_loglik(params, x, dt):
    log_theta, mu, log_sigma = params
    theta = np.exp(log_theta)
    sigma = np.exp(log_sigma)
    if not np.isfinite(theta) or not np.isfinite(sigma):
        return 1e12
    b = np.exp(-theta * dt)
    var_eps = sigma**2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
    if var_eps <= 0 or not np.isfinite(var_eps):
        return 1e12
    x_t = x[:-1]
    x_tp1 = x[1:]
    mean_eps = x_t * b + mu * (1.0 - b)
    ll = np.sum(_gauss_logpdf(x_tp1, mean_eps, var_eps))
    return -ll


def fit_mle(x: np.ndarray, dt: float = 1.0,
            init: OUParams | None = None) -> OUParams:
    """Maximum-likelihood with bounds (via log-parameters) using Nelder-Mead."""
    x = np.asarray(x, dtype=float)
    if init is None:
        init = fit_ols(x, dt=dt)

    # Initial guess in log space; bound theta in [1e-3, 50] effectively
    init_theta = float(np.clip(init.theta, 1e-3, 50.0))
    init_sigma = float(np.clip(init.sigma, 1e-3, 1e3))
    x0 = [np.log(init_theta), init.mu, np.log(init_sigma)]

    res = minimize(_neg_loglik, x0, args=(x, dt), method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 10_000})
    log_theta, mu, log_sigma = res.x
    theta = float(np.exp(log_theta))
    sigma = float(np.exp(log_sigma))
    return OUParams(mu=float(mu), theta=theta, sigma=sigma,
                    method="MLE", loglik=float(-res.fun))


# ----------------------------- Convenience ---------------------------------- #

def fit_both(x: np.ndarray, dt: float = 1.0) -> dict:
    ols = fit_ols(x, dt=dt)
    mle = fit_mle(x, dt=dt, init=ols)
    return {"OLS": ols, "MLE": mle}


def stationary_std(p: OUParams) -> float:
    """SD of stationary distribution of OU: sigma / sqrt(2 theta)."""
    return float(p.sigma / np.sqrt(2.0 * p.theta))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_mu, true_theta, true_sigma, n = 5.0, 0.4, 11.0, 500
    x = np.zeros(n); x[0] = true_mu
    for t in range(1, n):
        b = np.exp(-true_theta)
        var = true_sigma**2 * (1 - np.exp(-2*true_theta)) / (2*true_theta)
        x[t] = x[t-1]*b + true_mu*(1-b) + rng.normal(0, np.sqrt(var))
    fits = fit_both(x)
    print("True:", true_mu, true_theta, true_sigma)
    print(fits["OLS"])
    print(fits["MLE"])
    print("stationary std (MLE):", stationary_std(fits["MLE"]))
