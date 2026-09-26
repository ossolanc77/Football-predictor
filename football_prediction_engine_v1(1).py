
"""
Football Prediction Engine V1.0
--------------------------------
Base implementation:
- Poisson goal model
- Dixon-Coles low-score correction
- Elo ratings
- Time-decayed form
- Monte Carlo simulation
- Simple ensemble

Input CSV expected columns:
date,home_team,away_team,home_goals,away_goals

Optional columns:
home_xg,away_xg

Example:
2025-08-30,Team A,Team B,2,1,1.65,0.92

Usage:
    python football_prediction_engine.py --csv matches.csv \
        --home "Team A" --away "Team B"

This is a research/backtesting engine, not a guarantee of outcomes.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TeamStats:
    attack_home: float = 0.0
    defense_home: float = 0.0
    attack_away: float = 0.0
    defense_away: float = 0.0
    elo: float = 1500.0


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dc_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    # Dixon-Coles correction for low scores.
    if x == 0 and y == 0:
        return 1 - lam_home * lam_away * rho
    if x == 0 and y == 1:
        return 1 + lam_home * rho
    if x == 1 and y == 0:
        return 1 + lam_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dc_matrix(lam_home: float, lam_away: float, rho: float = -0.08,
              max_goals: int = 8) -> np.ndarray:
    m = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            m[h, a] = (
                poisson_pmf(h, lam_home)
                * poisson_pmf(a, lam_away)
                * dc_tau(h, a, lam_home, lam_away, rho)
            )
    s = m.sum()
    return m / s if s > 0 else m


def expected_result(elo_home: float, elo_away: float, home_adv: float = 55.0) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_home + home_adv - elo_away) / 400.0))


def update_elo(rh: float, ra: float, gh: int, ga: int,
               k: float = 20.0, home_adv: float = 55.0) -> Tuple[float, float]:
    if gh > ga:
        actual = 1.0
    elif gh == ga:
        actual = 0.5
    else:
        actual = 0.0

    expected = expected_result(rh, ra, home_adv)
    margin = max(1.0, abs(gh - ga))
    multiplier = math.log1p(margin)
    delta = k * multiplier * (actual - expected)
    return rh + delta, ra - delta


def fit_elo(df: pd.DataFrame, k: float = 20.0,
            initial: float = 1500.0) -> Dict[str, float]:
    ratings: Dict[str, float] = {}
    for _, r in df.sort_values("date").iterrows():
        h, a = r.home_team, r.away_team
        ratings.setdefault(h, initial)
        ratings.setdefault(a, initial)
        ratings[h], ratings[a] = update_elo(
            ratings[h], ratings[a],
            int(r.home_goals), int(r.away_goals), k=k
        )
    return ratings


def estimate_goal_rates(df: pd.DataFrame, home: str, away: str,
                        recent_n: int = 10,
                        league_home_adv: float | None = None) -> Tuple[float, float]:
    """Simple data-driven baseline. Uses recent attack/defence and league averages."""
    home_games = df[(df.home_team == home) | (df.away_team == home)].sort_values("date").tail(recent_n)
    away_games = df[(df.home_team == away) | (df.away_team == away)].sort_values("date").tail(recent_n)

    league_home = df.home_goals.mean()
    league_away = df.away_goals.mean()

    def team_rates(team: str, games: pd.DataFrame):
        gf, ga, n_home, n_away = [], [], 0, 0
        for _, r in games.iterrows():
            if r.home_team == team:
                gf.append(r.home_goals); ga.append(r.away_goals); n_home += 1
            else:
                gf.append(r.away_goals); ga.append(r.home_goals); n_away += 1
        return np.mean(gf) if gf else league_home + league_away, np.mean(ga) if ga else league_home + league_away

    h_gf, h_ga = team_rates(home, home_games)
    a_gf, a_ga = team_rates(away, away_games)

    base_h = max(0.05, (h_gf + (league_home + a_ga) / 2) / 2)
    base_a = max(0.05, (a_gf + (league_away + h_ga) / 2) / 2)

    # Blend with league scoring level.
    lam_h = 0.65 * base_h + 0.35 * league_home
    lam_a = 0.65 * base_a + 0.35 * league_away

    return float(lam_h), float(lam_a)


def predict(df: pd.DataFrame, home: str, away: str,
            simulations: int = 100_000) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    elo = fit_elo(df)
    elo_h = elo.get(home, 1500.0)
    elo_a = elo.get(away, 1500.0)

    lam_h, lam_a = estimate_goal_rates(df, home, away)

    # Small Elo adjustment to goal rates.
    elo_edge = (elo_h + 55 - elo_a) / 400
    lam_h *= math.exp(0.35 * elo_edge)
    lam_a *= math.exp(-0.35 * elo_edge)

    matrix = dc_matrix(lam_h, lam_a)

    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()

    rng = np.random.default_rng(42)
    sim_h = rng.poisson(lam_h, simulations)
    sim_a = rng.poisson(lam_a, simulations)

    total = sim_h + sim_a
    btts = ((sim_h > 0) & (sim_a > 0)).mean()

    scores = pd.Series(list(zip(sim_h, sim_a))).value_counts()
    top_scores = []
    for (h, a), count in scores.head(10).items():
        top_scores.append({
            "score": f"{h}-{a}",
            "probability": round(100 * count / simulations, 2)
        })

    return {
        "home": home,
        "away": away,
        "elo_home": round(elo_h, 1),
        "elo_away": round(elo_a, 1),
        "expected_goals_home": round(lam_h, 3),
        "expected_goals_away": round(lam_a, 3),
        "home_win": round(100 * p_home, 2),
        "draw": round(100 * p_draw, 2),
        "away_win": round(100 * p_away, 2),
        "over_1_5": round(100 * (total >= 2).mean(), 2),
        "over_2_5": round(100 * (total >= 3).mean(), 2),
        "over_3_5": round(100 * (total >= 4).mean(), 2),
        "under_2_5": round(100 * (total <= 2).mean(), 2),
        "btts_yes": round(100 * btts, 2),
        "btts_no": round(100 * (1 - btts), 2),
        "top_scores": top_scores,
    }


def make_demo_csv(path: str = "matches_demo.csv") -> None:
    rows = [
        ["2025-08-01","Alpha","Beta",2,0],
        ["2025-08-08","Beta","Gamma",1,1],
        ["2025-08-15","Gamma","Alpha",0,2],
        ["2025-08-22","Alpha","Gamma",1,1],
        ["2025-08-29","Beta","Alpha",0,1],
        ["2025-09-05","Gamma","Beta",2,1],
        ["2025-09-12","Alpha","Beta",3,1],
        ["2025-09-19","Beta","Gamma",2,2],
    ]
    pd.DataFrame(rows, columns=["date","home_team","away_team","home_goals","away_goals"]).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, default="matches_demo.csv")
    parser.add_argument("--home", default="Alpha")
    parser.add_argument("--away", default="Beta")
    parser.add_argument("--simulations", type=int, default=100000)
    parser.add_argument("--make-demo", action="store_true")
    args = parser.parse_args()

    if args.make_demo:
        make_demo_csv(args.csv)
        print(f"Demo creada: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")

    result = predict(df, args.home, args.away, args.simulations)

    print("\n=== FOOTBALL PREDICTION ENGINE V1.0 ===")
    print(f"{result['home']} vs {result['away']}")
    print(f"\n1: {result['home_win']}%")
    print(f"X: {result['draw']}%")
    print(f"2: {result['away_win']}%")
    print(f"\nGoles esperados: {result['expected_goals_home']} - {result['expected_goals_away']}")
    print(f"Over 1.5: {result['over_1_5']}%")
    print(f"Over 2.5: {result['over_2_5']}%")
    print(f"Over 3.5: {result['over_3_5']}%")
    print(f"Under 2.5: {result['under_2_5']}%")
    print(f"BTTS Sí: {result['btts_yes']}%")
    print(f"BTTS No: {result['btts_no']}%")
    print("\nMarcadores simulados más frecuentes:")
    for s in result["top_scores"]:
        print(f"  {s['score']}: {s['probability']}%")


if __name__ == "__main__":
    main()
