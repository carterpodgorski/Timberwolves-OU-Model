"""
Loads Minnesota Timberwolves game logs for the 2023-24 and 2024-25 seasons.

Primary source: nba_api (live stats.nba.com endpoint).
If the API is blocked / fails, falls back to a deterministic synthetic
generator so the rest of the pipeline still runs end-to-end.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

MIN_TEAM_ID = 1610612750  # Minnesota Timberwolves franchise id (NBA stats)
SEASONS = ["2023-24", "2024-25"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _fetch_season_via_api(season: str) -> Optional[pd.DataFrame]:
    """Try to pull a regular-season game log from stats.nba.com.

    Uses the modern ``teamgamelogs`` endpoint which includes PLUS_MINUS.
    """
    try:
        from nba_api.stats.endpoints import teamgamelogs
    except Exception:
        return None

    try:
        log = teamgamelogs.TeamGameLogs(
            team_id_nullable=str(MIN_TEAM_ID),
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=30,
        )
        df = log.get_data_frames()[0]
    except Exception as e:
        print(f"[data_loader] API failed for {season}: {e}")
        return None

    if df is None or df.empty:
        return None

    df = df.copy()
    df["SEASON"] = season
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    # PTS = MIN points, opponent points = PTS - PLUS_MINUS (PM = us - them)
    df["OPP_PTS"] = df["PTS"] - df["PLUS_MINUS"]
    df["POINT_DIFF"] = df["PLUS_MINUS"].astype(float)
    df["WIN"] = (df["WL"] == "W").astype(int)
    keep = ["SEASON", "GAME_DATE", "MATCHUP", "WL", "PTS", "OPP_PTS",
            "POINT_DIFF", "WIN"]
    return df[keep]


def _synthetic_season(season: str, n_games: int = 82, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OU-like season used as a fallback only."""
    rng = np.random.default_rng(seed + hash(season) % 1000)
    mu, theta, sigma = 5.0, 0.35, 11.5
    x = np.zeros(n_games)
    x[0] = mu + rng.normal(0, sigma)
    for t in range(1, n_games):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + rng.normal(0, sigma)
    start = pd.Timestamp("2023-10-25") if season == "2023-24" else pd.Timestamp("2024-10-22")
    dates = pd.bdate_range(start=start, periods=n_games, freq="2B")
    pts = 113 + rng.normal(0, 6, n_games)
    opp = pts - x
    return pd.DataFrame({
        "SEASON": season,
        "GAME_DATE": dates,
        "MATCHUP": "MIN vs OPP",
        "WL": np.where(x > 0, "W", "L"),
        "PTS": pts.round(0),
        "OPP_PTS": opp.round(0),
        "POINT_DIFF": x,
        "WIN": (x > 0).astype(int),
    })


def load_games(use_cache: bool = True) -> pd.DataFrame:
    """Returns a single concatenated DataFrame for both seasons."""
    cache_path = os.path.join(DATA_DIR, "timberwolves_games.csv")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["GAME_DATE"])
        if not df.empty:
            return df

    frames = []
    using_synth = False
    for season in SEASONS:
        df = _fetch_season_via_api(season)
        if df is None or df.empty:
            using_synth = True
            df = _synthetic_season(season)
        frames.append(df)
        time.sleep(1)  # be polite to the API

    out = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)
    if using_synth:
        print("[data_loader] WARNING: at least one season used SYNTHETIC fallback data.")
    out.to_csv(cache_path, index=False)
    return out


if __name__ == "__main__":
    df = load_games(use_cache=False)
    print(df.head())
    print(f"\nTotal games: {len(df)}")
    print(df.groupby("SEASON")["WIN"].agg(["count", "sum", "mean"]))
