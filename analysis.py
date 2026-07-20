"""
End-to-end pipeline:

  1. Load MIN game logs (2023-24 & 2024-25).
  2. Fit OU on pooled data and per season -> mu, theta, sigma  (OLS + MLE).
  3. Monte Carlo 10,000 paths simulating an 82-game *next* season.
  4. Compute total-win distribution (win iff simulated point diff > 0).
  5. Save plots + summary.json + report.md.

Run with:
    python analysis.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_games, SEASONS
from ou_model import fit_both, OUParams, stationary_std
from simulate import simulate_paths
from win_probability import (
    wins_from_paths, stationary_win_probability, per_step_win_probability,
)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

GAMES_PER_SEASON = 82


def _save_fig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------- Plotting helpers ------------------------------- #

def plot_pointdiff_with_mean(df: pd.DataFrame, params: OUParams):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = {"2023-24": "#236192", "2024-25": "#78BE20"}
    for season in SEASONS:
        sub = df[df.SEASON == season]
        ax.plot(sub.GAME_DATE, sub.POINT_DIFF, "o-", lw=1.0, ms=3.5,
                color=colors.get(season, "gray"), label=season, alpha=0.85)
    ax.axhline(params.mu, color="crimson", lw=2,
               label=f"OU mean μ = {params.mu:.2f}")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_title(f"Minnesota Timberwolves: game-by-game point differential\n"
                 f"OU fit (MLE):  μ={params.mu:.2f},  θ={params.theta:.3f},  "
                 f"σ={params.sigma:.2f}  (stationary SD = {stationary_std(params):.2f})")
    ax.set_xlabel("Game date")
    ax.set_ylabel("Point differential (MIN − OPP)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    _save_fig(fig, "01_pointdiff_with_mean.png")


def plot_acf(x: np.ndarray, max_lag: int = 20):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = x - x.mean()
    n = len(x)
    acf = [1.0] + [
        float(np.dot(x[:-k], x[k:]) / np.dot(x, x)) for k in range(1, max_lag + 1)
    ]
    lags = np.arange(len(acf))
    conf = 1.96 / np.sqrt(n)
    ax.bar(lags, acf, width=0.6, color="#236192")
    ax.axhline(conf, ls="--", color="red", lw=0.8, label=f"95% CI ±{conf:.2f}")
    ax.axhline(-conf, ls="--", color="red", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Autocorrelation of game-to-game point differential")
    ax.set_xlabel("Lag (games)")
    ax.set_ylabel("ACF")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, "00_acf.png")


def plot_sample_paths(paths: np.ndarray, params: OUParams, x0: float,
                      n_show: int = 60):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i in range(min(n_show, paths.shape[0])):
        ax.plot(paths[i], lw=0.6, alpha=0.35, color="#236192")
    mean_path = paths.mean(axis=0)
    q05 = np.quantile(paths, 0.05, axis=0)
    q95 = np.quantile(paths, 0.95, axis=0)
    ax.plot(mean_path, color="crimson", lw=2, label="Simulated mean")
    ax.fill_between(np.arange(len(mean_path)), q05, q95,
                    color="crimson", alpha=0.15, label="5%–95% band")
    ax.axhline(params.mu, ls="--", color="black", lw=0.8, label=f"μ = {params.mu:.2f}")
    ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax.set_title(f"Monte Carlo: {paths.shape[0]:,} simulated paths "
                 f"({n_show} shown). Start x0 = {x0:.2f}, horizon = {paths.shape[1]} games.")
    ax.set_xlabel("Games into future")
    ax.set_ylabel("Simulated point differential")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, "02_simulated_paths.png")


def plot_win_distribution(total_wins: np.ndarray, n_games: int,
                          stationary_p_win: float):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(total_wins.min(), total_wins.max() + 2) - 0.5
    ax.hist(total_wins, bins=bins, color="#78BE20", edgecolor="white", alpha=0.85)
    mean = total_wins.mean()
    p05, p50, p95 = np.percentile(total_wins, [5, 50, 95])
    ax.axvline(mean, color="crimson", lw=2, label=f"Mean = {mean:.1f}")
    ax.axvline(p05, color="black", lw=1, ls="--", label=f"5% = {p05:.0f}")
    ax.axvline(p95, color="black", lw=1, ls="--", label=f"95% = {p95:.0f}")
    ax.axvline(stationary_p_win * n_games, color="purple", lw=1.5, ls=":",
               label=f"Stationary E[wins] = {stationary_p_win*n_games:.1f}")
    ax.set_title(f"Projected total wins over {n_games} games\n"
                 f"({len(total_wins):,} Monte Carlo simulations of an OU season)")
    ax.set_xlabel("Total wins")
    ax.set_ylabel("Number of simulations")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, "03_win_distribution.png")
    return {"mean": float(mean), "p05": float(p05),
            "p50": float(p50), "p95": float(p95)}


def plot_per_game_pwin(params: OUParams, x0: float, n_games: int):
    p_t = per_step_win_probability(params, x0=x0, n_steps=n_games)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, n_games + 1), p_t, color="#236192", lw=2,
            label="Conditional P(win) at game t")
    p_inf = stationary_win_probability(params)
    ax.axhline(p_inf, color="crimson", ls="--",
               label=f"Stationary P(win) = {p_inf:.3f}")
    ax.set_xlabel("Game index t (from start)")
    ax.set_ylabel("P(point diff > 0)")
    ax.set_title(f"Closed-form P(win) by game (start x0 = {x0:.2f})")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, "04_per_game_pwin.png")


# ----------------------------- Per-season fits ------------------------------ #

def per_season_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        x = df.loc[df.SEASON == season, "POINT_DIFF"].to_numpy(dtype=float)
        if len(x) < 5:
            continue
        for name, p in fit_both(x).items():
            rows.append({"season": season, "method": name,
                         "mu": p.mu, "theta": p.theta, "sigma": p.sigma,
                         "stat_sd": stationary_std(p),
                         "loglik": p.loglik})
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Tiny replacement for DataFrame.to_markdown so we don't need tabulate."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------- Main -------------------------------------- #

def main():
    print("==> Loading game data ...")
    df = load_games(use_cache=True)
    by_season = df.groupby("SEASON").agg(
        games=("WIN", "count"), wins=("WIN", "sum"),
        avg_diff=("POINT_DIFF", "mean"),
        std_diff=("POINT_DIFF", "std"),
    )
    print(by_season)

    # --- Fit OU on pooled history ----------------------------------------
    x_all = df["POINT_DIFF"].to_numpy(dtype=float)
    fits_all = fit_both(x_all)
    print("\n==> OU fit (pooled, both seasons):")
    print(" ", fits_all["OLS"])
    print(" ", fits_all["MLE"])
    best = fits_all["MLE"]
    print(f"  stationary std (MLE) = sigma/sqrt(2 theta) = {stationary_std(best):.3f}")

    # --- Per-season fits for the report ---------------------------------
    per_season = per_season_table(df)
    per_season.to_csv(os.path.join(OUT, "per_season_fits.csv"), index=False)
    print("\n==> Per-season fits:")
    print(per_season.to_string(index=False))

    # --- Project an 82-game season --------------------------------------
    # We always project a fresh 82-game season starting from the current
    # state (last observed point differential, or mu if no data).
    last_diff = float(df["POINT_DIFF"].iloc[-1])
    print(f"\n==> Simulating an 82-game season starting from x0 = "
          f"{last_diff:.2f} (last observed point diff).")

    paths = simulate_paths(best, x0=last_diff, n_steps=GAMES_PER_SEASON,
                           n_paths=10_000, seed=42)
    total_wins = wins_from_paths(paths)
    p_inf = stationary_win_probability(best)

    print(f"  stationary P(win) = {p_inf:.3f}  -> stationary expected wins "
          f"over 82 = {p_inf*GAMES_PER_SEASON:.1f}")
    print(f"  simulated mean wins = {total_wins.mean():.2f}  "
          f"(5%-95% = [{np.percentile(total_wins, 5):.0f}, "
          f"{np.percentile(total_wins, 95):.0f}])")

    # --- Plots -----------------------------------------------------------
    print("\n==> Plots ...")
    plot_acf(x_all, max_lag=20)
    plot_pointdiff_with_mean(df, best)
    plot_sample_paths(paths, best, x0=last_diff)
    win_stats = plot_win_distribution(total_wins, GAMES_PER_SEASON, p_inf)
    plot_per_game_pwin(best, x0=last_diff, n_games=GAMES_PER_SEASON)

    # --- Save summary ----------------------------------------------------
    summary = {
        "data": {
            "games_loaded": int(len(df)),
            "by_season": by_season.reset_index().to_dict(orient="records"),
        },
        "fit_pooled": {
            "OLS": vars(fits_all["OLS"]),
            "MLE": vars(fits_all["MLE"]),
            "stationary_std_mle": stationary_std(best),
        },
        "fit_per_season": per_season.to_dict(orient="records"),
        "projection_82_game_season": {
            "x0": last_diff,
            "stationary_P_win": p_inf,
            "stationary_expected_wins": p_inf * GAMES_PER_SEASON,
            "monte_carlo_total_wins": win_stats,
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {os.path.join(OUT, 'summary.json')}")

    # --- Markdown report -------------------------------------------------
    md = []
    md.append("# Minnesota Timberwolves — Ornstein–Uhlenbeck model: results\n")
    md.append("## Data summary\n")
    md.append(df_to_markdown(by_season.reset_index(), floatfmt=".3f"))
    md.append("\n## OU parameter estimates (pooled 2023-24 + 2024-25)\n")
    md.append("| Method | μ | θ | σ | stationary SD | log-lik |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in ("OLS", "MLE"):
        p = fits_all[name]
        md.append(f"| {name} | {p.mu:.3f} | {p.theta:.4f} | {p.sigma:.3f} | "
                  f"{stationary_std(p):.3f} | {p.loglik:.2f} |")
    md.append("\n*Interpretation:*  μ is the team's long-run point-differential "
              "(a positive μ means above-average true talent). θ controls how "
              "fast deviations decay back to μ (large θ = streaks die quickly). "
              "σ scales the random shock per √game. The stationary standard "
              "deviation σ/√(2θ) is the natural game-to-game dispersion implied "
              "by the fit.\n")
    md.append("## Per-season fits\n")
    md.append(df_to_markdown(per_season, floatfmt=".3f"))
    md.append("\n## Projection: 10,000 simulated 82-game seasons\n")
    md.append(f"- Starting state x₀ = **{last_diff:.2f}** (last observed game).")
    md.append(f"- Stationary P(win) under fitted OU = **{p_inf:.3f}**, so "
              f"stationary expected wins over 82 games ≈ "
              f"**{p_inf*GAMES_PER_SEASON:.1f}**.")
    md.append(f"- Monte-Carlo total wins: mean **{win_stats['mean']:.1f}**, "
              f"median **{win_stats['p50']:.0f}**, 5%–95% interval "
              f"**[{win_stats['p05']:.0f}, {win_stats['p95']:.0f}]**.")
    md.append("\n## Plots\n")
    md.append("- `00_acf.png` — autocorrelation of game-to-game point differential")
    md.append("- `01_pointdiff_with_mean.png` — game-by-game diff vs OU mean")
    md.append("- `02_simulated_paths.png` — Monte Carlo paths for an 82-game season")
    md.append("- `03_win_distribution.png` — distribution of total season wins")
    md.append("- `04_per_game_pwin.png` — closed-form P(win) per future game")
    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {os.path.join(OUT, 'report.md')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
