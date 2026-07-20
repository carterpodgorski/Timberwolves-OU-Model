# Modeling the Minnesota Timberwolves via the Ornstein–Uhlenbeck Process

Code accompanying the stochastic-processes class project. Models the team's
game-by-game point differential as an OU process, fits parameters, and runs
10,000 Monte Carlo simulations to project a full 82-game season.

## Files

| file | purpose |
|---|---|
| `data_loader.py` | Pulls 2023-24 & 2024-25 game logs from `nba_api` (with synthetic fallback). |
| `ou_model.py` | OU parameter estimation: OLS regression **and** exact-likelihood MLE. |
| `simulate.py` | Exact discrete-time Monte Carlo simulation of OU paths. |
| `win_probability.py` | Win-probability accounting (win iff diff > 0). |
| `analysis.py` | End-to-end pipeline: data → fit → simulate → plots → report. |
| `notebook.ipynb` | Walk-through notebook for the report. |
| `outputs/` | Generated plots, `summary.json`, `report.md`. |
| `data/` | Cached game-log CSV. |

## Quick start

```bash
pip install nba_api scipy matplotlib pandas numpy
python analysis.py
```

This produces `outputs/01_pointdiff_with_mean.png`, the simulation paths,
the win-total histogram, the calibration plots, and `outputs/report.md`
with all numbers.

## Model

Continuous-time SDE:

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

- **μ** = "true talent" (long-run mean point differential).
- **θ** = mean-reversion speed (consistency: large θ = streaks die fast).
- **σ** = volatility (random "chaos" per √game).

Discrete one-game step (`dt = 1`):

$$
X_{t+1} \mid X_t \sim \mathcal N\!\Big(X_t e^{-\theta} + \mu(1 - e^{-\theta}),\;
   \sigma^{2}\frac{1-e^{-2\theta}}{2\theta}\Big).
$$

This is exact (no Euler error) and is what we use for both MLE and the
Monte Carlo simulator.

## Win probability

Since a team wins a game iff its point differential is positive, we don't
need a separate logistic — for each simulated path we just count games with
`diff > 0`. The 10,000-path histogram of total wins **is** the predicted
win-total distribution. We additionally report the closed-form

$$
P(X_t > 0 \mid X_0 = x_0) = \Phi\!\left(\frac{m_t}{\sqrt{v_t}}\right)
$$

per game and the stationary win probability $\Phi(\mu / (\sigma/\sqrt{2\theta}))$.

## A note on identifiability

NBA game-to-game point differentials have very weak autocorrelation, so the
discrete AR(1) slope $b = e^{-\theta}$ is near zero. In that regime θ and σ
are individually weakly identified — you can trade them off — but the
**stationary** distribution $\mathcal N(\mu,\,\sigma^2/(2\theta))$ is well
identified. That is what governs win predictions, and it is what both the
OLS and MLE fits agree on in our results.
