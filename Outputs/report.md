# Minnesota Timberwolves — Ornstein–Uhlenbeck model: results

## Data summary

| SEASON | games | wins | avg_diff | std_diff |
|---|---|---|---|---|
| 2023-24 | 82 | 56 | 6.451 | 13.476 |
| 2024-25 | 82 | 49 | 5.000 | 13.009 |

## OU parameter estimates (pooled 2023-24 + 2024-25)

| Method | μ | θ | σ | stationary SD | log-lik |
|---|---:|---:|---:|---:|---:|
| OLS | 5.726 | 2.9957 | 32.369 | 13.224 | -652.65 |
| MLE | 5.779 | 32.1071 | 105.827 | 13.206 | -651.94 |

*Interpretation:*  μ is the team's long-run point-differential (a positive μ means above-average true talent). θ controls how fast deviations decay back to μ (large θ = streaks die quickly). σ scales the random shock per √game. The stationary standard deviation σ/√(2θ) is the natural game-to-game dispersion implied by the fit.

## Per-season fits

| season | method | mu | theta | sigma | stat_sd | loglik |
|---|---|---|---|---|---|---|
| 2023-24 | OLS | 6.451 | 2.996 | 32.986 | 13.476 | -326.378 |
| 2023-24 | MLE | 6.568 | 32.887 | 108.958 | 13.435 | -325.359 |
| 2024-25 | OLS | 5.169 | 2.475 | 29.149 | 13.102 | -322.040 |
| 2024-25 | MLE | 5.169 | 2.475 | 28.787 | 12.939 | -322.028 |

## Projection: 10,000 simulated 82-game seasons

- Starting state x₀ = **11.00** (last observed game).
- Stationary P(win) under fitted OU = **0.669**, so stationary expected wins over 82 games ≈ **54.9**.
- Monte-Carlo total wins: mean **54.9**, median **55**, 5%–95% interval **[48, 62]**.

## Plots

- `00_acf.png` — autocorrelation of game-to-game point differential
- `01_pointdiff_with_mean.png` — game-by-game diff vs OU mean
- `02_simulated_paths.png` — Monte Carlo paths for an 82-game season
- `03_win_distribution.png` — distribution of total season wins
- `04_per_game_pwin.png` — closed-form P(win) per future game