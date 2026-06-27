"""
FIFA World Cup 2026 Prediction Model
=====================================
Standard Linear Dixon-Coles Model
- Learns latent Attack and Defense strengths purely from historical results.
- Incorporates a Home Advantage parameter.
- Includes the classic Dixon-Coles `rho` parameter for low-scoring match dependence.
- Uses exponential time decay to weigh recent matches more heavily.
"""

import math
import os
import random
import warnings
from collections import defaultdict
from functools import cmp_to_key
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# 1. TOURNAMENT STRUCTURE  (2026 FIFA WC)
# ─────────────────────────────────────────────

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

NAME_MAP = {
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Curaçao": "Curacao",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Czechoslovakia": "Czech Republic",
    "Dahomey": "Benin",
    "Upper Volta": "Burkina Faso",
    "Netherlands Antilles": "Curacao",
    "Bohemia": "Czech Republic",
    "Bohemia and Moravia": "Czech Republic",
    "Representation of Czechs and Slovaks": "Czech Republic",
    "Belgian Congo": "DR Congo",
    "Congo-Léopoldville": "DR Congo",
    "Congo-Kinshasa": "DR Congo",
    "Zaïre": "DR Congo",
    "French Somaliland": "Djibouti",
    "United Arab Republic": "Egypt",
    "Swaziland": "Eswatini",
    "Gold Coast": "Ghana",
    "Portuguese Guinea": "Guinea-Bissau",
    "British Guiana": "Guyana",
    "Dutch East Indies": "Indonesia",
    "Mandatory Palestine": "Israel",
    "Nyasaland": "Malawi",
    "Malaya": "Malaysia",
    "Burma": "Myanmar",
    "Macedonia": "North Macedonia",
    "Ireland": "Northern Ireland",
    "Irish Free State": "Republic of Ireland",
    "Éire": "Republic of Ireland",
    "Soviet Union": "Russia",
    "CIS": "Russia",
    "Western Samoa": "Samoa",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Ceylon": "Sri Lanka",
    "Dutch Guyana": "Suriname",
    "Tanganyika": "Tanzania",
    "New Hebrides": "Vanuatu",
    "Northern Rhodesia": "Zambia",
    "Southern Rhodesia": "Zimbabwe",
    "German DR": "Germany",
    "Yugoslavia": "Serbia",
    "Vietnam Republic": "Vietnam",
    "North Vietnam": "Vietnam",
    "South Yemen": "Yemen",
    "Yemen DPR": "Yemen",
}

ALL_TEAMS = sorted({t for teams in GROUPS.values() for t in teams})

REFERENCE_DATE = pd.Timestamp("2026-06-10")
DECAY_HALF_LIFE_DAYS = 365 * 3  # 3-year half-life

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 60,
    "Confederations Cup": 50,
    "AFC Asian Cup": 50,
    "Africa Cup of Nations": 50,
    "UEFA Euro": 50,
    "Copa America": 50,
    "CONCACAF Gold Cup": 40,
    "UEFA Nations League": 40,
    "Friendly": 20,
}
DEFAULT_TOURNAMENT_WEIGHT = 30


def get_tournament_weight(tournament: str) -> float:
    if "qualif" in str(tournament).lower():
        return 40
    for key, val in TOURNAMENT_WEIGHTS.items():
        if key.lower() in str(tournament).lower():
            return float(val)
    return float(DEFAULT_TOURNAMENT_WEIGHT)


# ─────────────────────────────────────────────
# 2. DATA LOADING & PREP
# ─────────────────────────────────────────────


def load_and_prepare(csv_path: str = "results.csv"):
    print("Loading results.csv …")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "..", "data", "results.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])

    df = df.sort_values("date").reset_index(drop=True)

    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)

    return df


def time_weight(
    date: pd.Timestamp,
    tournament: str = "",
    home_team: str = "",
    away_team: str = "",
    ref_date: pd.Timestamp = REFERENCE_DATE,
) -> float:
    days_ago = (ref_date - date).days
    days_ago = max(days_ago, 0)
    lam = math.log(2) / DECAY_HALF_LIFE_DAYS
    decay = math.exp(-lam * days_ago)
    return decay * get_tournament_weight(tournament)


# ─────────────────────────────────────────────
# 3. DIXON-COLES MODEL
# ─────────────────────────────────────────────


class DixonColesModel(nn.Module):
    def __init__(self, num_teams):
        super().__init__()
        self.attack = nn.Embedding(num_teams, 1)
        self.defense = nn.Embedding(num_teams, 1)
        self.home_adv = nn.Parameter(torch.tensor(0.2))
        self.rho = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.attack.weight, mean=0, std=0.1)
        nn.init.normal_(self.defense.weight, mean=0, std=0.1)

    def forward(self, home_idx, away_idx, neutral_venue_flag):
        att_h = self.attack(home_idx).squeeze(-1)
        def_h = self.defense(home_idx).squeeze(-1)
        att_a = self.attack(away_idx).squeeze(-1)
        def_a = self.defense(away_idx).squeeze(-1)

        lambda_h = torch.exp(att_h + def_a + self.home_adv * (1 - neutral_venue_flag))
        lambda_a = torch.exp(att_a + def_h)

        return lambda_h, lambda_a


def dixon_coles_loss(lambda_h, lambda_a, rho, goals_h, goals_a, weights):
    nll_h = lambda_h - goals_h * torch.log(lambda_h + 1e-8)
    nll_a = lambda_a - goals_a * torch.log(lambda_a + 1e-8)

    tau = torch.ones_like(goals_h)

    m_00 = (goals_h == 0) & (goals_a == 0)
    m_01 = (goals_h == 0) & (goals_a == 1)
    m_10 = (goals_h == 1) & (goals_a == 0)
    m_11 = (goals_h == 1) & (goals_a == 1)

    # Using clamp to ensure we don't end up with negative tau from extreme rho
    tau[m_00] = 1 - lambda_h[m_00] * lambda_a[m_00] * rho
    tau[m_01] = 1 + lambda_h[m_01] * rho
    tau[m_10] = 1 + lambda_a[m_10] * rho
    tau[m_11] = 1 - rho

    tau = torch.clamp(tau, min=1e-8)

    log_likelihood = -nll_h - nll_a + torch.log(tau)
    # we want to maximize log_likelihood, so we minimize negative weighted LL
    loss = -torch.mean(log_likelihood * weights)
    return loss


def train_dixon_coles(
    df_train, num_teams, team_idx, epochs=3000, lr=0.01, print_freq=500
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Prepare Training Data ---
    hi_train, ai_train, venue_train, hs_train, aw_train, w_train = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for _, row in df_train.iterrows():
        h, a = row["home_team"], row["away_team"]
        hi_train.append(team_idx[h])
        ai_train.append(team_idx[a])
        venue_train.append(1.0 if bool(row.get("neutral", False)) else 0.0)
        hs_train.append(float(row["home_score"]))
        aw_train.append(float(row["away_score"]))
        w_train.append(time_weight(row["date"], str(row.get("tournament", "")), h, a))

    hi_t = torch.tensor(hi_train, dtype=torch.long, device=device)
    ai_t = torch.tensor(ai_train, dtype=torch.long, device=device)
    venue_t = torch.tensor(venue_train, dtype=torch.float, device=device)
    hs_t = torch.tensor(hs_train, dtype=torch.float, device=device)
    aw_t = torch.tensor(aw_train, dtype=torch.float, device=device)
    w_t = torch.tensor(w_train, dtype=torch.float, device=device)
    w_t = w_t / (w_t.mean() + 1e-8)

    model = DixonColesModel(num_teams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        lambda_h, lambda_a = model(hi_t, ai_t, venue_t)
        loss = dixon_coles_loss(lambda_h, lambda_a, model.rho, hs_t, aw_t, w_t)
        loss.backward()

        with torch.no_grad():
            mean_att = model.attack.weight.mean()
            model.attack.weight.sub_(mean_att)
            model.defense.weight.add_(mean_att)
            model.rho.clamp_(min=-0.2, max=0.2)

        opt.step()

        if epoch % print_freq == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | train_loss: {loss.item():.4f} | Home Adv: {model.home_adv.item():.3f} | Rho: {model.rho.item():.3f}"
            )

    return model


# ─────────────────────────────────────────────
# 4. MATCH SIMULATION
# ─────────────────────────────────────────────


def precompute_match_probabilities(model, team_idx):
    """Precompute all expected goals for fast Monte Carlo."""
    print("Precomputing expected goals matrix...")
    prob_matrix = {}
    teams = list(team_idx.keys())
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        hosts = ["United States", "Mexico", "Canada"]

        for ta in teams:
            for tb in teams:
                if ta == tb:
                    continue
                ia = torch.tensor([team_idx[ta]], device=device)
                ib = torch.tensor([team_idx[tb]], device=device)

                v1_neutral = 1.0
                if ta in hosts and tb not in hosts:
                    v1_neutral = (
                        0.0  # ta gets home adv in match 1, neither in match 2 (away)
                    )
                elif tb in hosts and ta not in hosts:
                    v1_neutral = 1.0

                l_h1, l_a1 = model(ia, ib, torch.tensor([v1_neutral], device=device))
                # For reverse fixture, just to average (not strictly needed but smooths out)
                # In standard DC, we don't need reverse fixture averaging for neutral matches
                # We can just compute it once directly assuming neutral unless host plays.

                if v1_neutral == 1.0:  # pure neutral
                    l_h_neut, l_a_neut = model(
                        ia, ib, torch.tensor([1.0], device=device)
                    )
                    prob_matrix[(ta, tb)] = np.array([l_h_neut.item(), l_a_neut.item()])
                else:  # host advantage
                    prob_matrix[(ta, tb)] = np.array([l_h1.item(), l_a1.item()])

    return prob_matrix


PENALTY_SHOOTOUT_PROB_EACH = 0.5


def simulate_match(ta, tb, prob_matrix, knockout=False):
    lambdas = prob_matrix.get((ta, tb), np.array([1.0, 1.0]))

    gf_a = np.random.poisson(lambdas[0])
    gf_b = np.random.poisson(lambdas[1])

    winner = ta if gf_a > gf_b else (tb if gf_b > gf_a else None)

    if knockout and winner is None:
        # In classic DC, we don't model penalties, so 50/50 or slight attack edge
        if random.random() < PENALTY_SHOOTOUT_PROB_EACH:
            winner = ta
        else:
            winner = tb
        return winner, {"gf_a": gf_a, "gf_b": gf_b, "outcome": "penalties"}

    return winner, {
        "gf_a": gf_a,
        "gf_b": gf_b,
        "outcome": "90min" if winner else "draw",
    }


# ─────────────────────────────────────────────
# 5. TOURNAMENT SIMULATION
# ─────────────────────────────────────────────


def run_group_stage(prob_matrix):
    standings = {}

    for grp, teams in GROUPS.items():
        table = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0} for t in teams}
        h2h = defaultdict(lambda: defaultdict(int))

        for ta, tb in combinations(teams, 2):
            winner, res = simulate_match(ta, tb, prob_matrix, knockout=False)
            gf_a, gf_b = res["gf_a"], res["gf_b"]

            table[ta]["gf"] += gf_a
            table[ta]["ga"] += gf_b
            table[ta]["gd"] += gf_a - gf_b
            table[tb]["gf"] += gf_b
            table[tb]["ga"] += gf_a
            table[tb]["gd"] += gf_b - gf_a

            if winner == ta:
                table[ta]["pts"] += 3
                h2h[ta][tb] += 3
            elif winner == tb:
                table[tb]["pts"] += 3
                h2h[tb][ta] += 3
            else:
                table[ta]["pts"] += 1
                table[tb]["pts"] += 1
                h2h[ta][tb] += 1
                h2h[tb][ta] += 1

        def team_cmp(t1, t2):
            if table[t1]["pts"] != table[t2]["pts"]:
                return table[t1]["pts"] - table[t2]["pts"]
            if table[t1]["gd"] != table[t2]["gd"]:
                return table[t1]["gd"] - table[t2]["gd"]
            if table[t1]["gf"] != table[t2]["gf"]:
                return table[t1]["gf"] - table[t2]["gf"]
            if h2h[t1][t2] != h2h[t2][t1]:
                return h2h[t1][t2] - h2h[t2][t1]
            return 1 if random.random() > 0.5 else -1

        ranked = sorted(teams, key=cmp_to_key(team_cmp), reverse=True)
        standings[grp] = {"table": table, "ranked": ranked}

    return standings


def pick_best_third_place(standings):
    thirds = []
    for grp, s in standings.items():
        third_team = s["ranked"][2]
        row = s["table"][third_team]
        thirds.append((grp, third_team, row["pts"], row["gd"], row["gf"]))
    thirds.sort(key=lambda x: (x[2], x[3], x[4], random.random()), reverse=True)
    return [t[1] for t in thirds[:8]]


def build_r32_bracket(standings, best_thirds):
    advancing_teams = []
    for grp, s in standings.items():
        w = s["ranked"][0]
        advancing_teams.append((w, s["table"][w]))
        r = s["ranked"][1]
        advancing_teams.append((r, s["table"][r]))
    for t in best_thirds:
        for grp, s in standings.items():
            if s["ranked"][2] == t:
                advancing_teams.append((t, s["table"][t]))
                break
    advancing_teams.sort(
        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], random.random()),
        reverse=True,
    )
    matchups = [(advancing_teams[i][0], advancing_teams[31 - i][0]) for i in range(16)]
    return matchups


def pair_winners(winners):
    half = len(winners) // 2
    return [(winners[i], winners[i + half]) for i in range(half)]


def simulate_tournament(prob_matrix, n_sims=10000):
    win_counts = defaultdict(int)
    final_counts = defaultdict(int)
    sf_counts = defaultdict(int)
    group_adv_counts = defaultdict(int)

    print(f"\nRunning {n_sims} Monte Carlo simulations ...")
    for _ in tqdm(range(n_sims)):
        standings = run_group_stage(prob_matrix)
        best_thirds = pick_best_third_place(standings)

        r32_matchups = build_r32_bracket(standings, best_thirds)
        for ta, tb in r32_matchups:
            group_adv_counts[ta] += 1
            group_adv_counts[tb] += 1

        r32_winners = [
            simulate_match(ta, tb, prob_matrix, knockout=True)[0]
            for ta, tb in r32_matchups
        ]

        r16_matchups = pair_winners(r32_winners)
        r16_winners = [
            simulate_match(ta, tb, prob_matrix, knockout=True)[0]
            for ta, tb in r16_matchups
        ]

        qf_matchups = pair_winners(r16_winners)
        qf_winners = [
            simulate_match(ta, tb, prob_matrix, knockout=True)[0]
            for ta, tb in qf_matchups
        ]

        sf_matchups = pair_winners(qf_winners)
        sf_winners = []
        for ta, tb in sf_matchups:
            winner = simulate_match(ta, tb, prob_matrix, knockout=True)[0]
            sf_winners.append(winner)
            sf_counts[ta] += 1
            sf_counts[tb] += 1

        fn_a, fn_b = sf_winners
        winner = simulate_match(fn_a, fn_b, prob_matrix, knockout=True)[0]
        final_counts[fn_a] += 1
        final_counts[fn_b] += 1
        win_counts[winner] += 1

    return win_counts, final_counts, sf_counts, group_adv_counts


def print_mc_report(win_counts, final_counts, sf_counts, n_sims):
    print(f"\n{'=' * 60}")
    print(f"  MONTE CARLO RESULTS  (n={n_sims} simulations)")
    print(f"{'=' * 60}")
    print(f"\n  {'Team':<30} {'Win%':>7} {'Final%':>8} {'SF%':>6}")
    print(f"  {'─' * 54}")

    sorted_teams = sorted(win_counts.keys(), key=lambda t: win_counts[t], reverse=True)
    for t in sorted_teams:
        wp = win_counts[t] / n_sims * 100
        fp = final_counts.get(t, 0) / n_sims * 100
        sp = sf_counts.get(t, 0) / n_sims * 100
        print(f"  {t:<30} {wp:>6.1f}%  {fp:>6.1f}%  {sp:>5.1f}%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main(csv_path="results.csv", epochs=1200, n_sims=5000):
    df = load_and_prepare(csv_path)

    all_teams_in_df = sorted(set(df["home_team"]).union(set(df["away_team"])))
    team_idx = {t: i for i, t in enumerate(all_teams_in_df)}
    num_teams = len(all_teams_in_df)

    # Train on ALL data up to 2026 to prepare for the WC Simulation
    print(f"\n{'=' * 50}")
    print("▶ Training Full Inference Model")
    print(f"{'=' * 50}")
    model = train_dixon_coles(
        df,
        num_teams,
        team_idx,
        epochs=epochs,
        lr=0.01,
        print_freq=epochs // 5 if epochs >= 5 else 1,
    )

    print("\nTop 10 Teams by Attack (higher is better):")
    model.eval()
    with torch.no_grad():
        atts = model.attack.weight.squeeze().cpu().numpy()
        defs = model.defense.weight.squeeze().cpu().numpy()
    att_sorted = sorted(
        [(t, atts[idx]) for t, idx in team_idx.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    for i, (t, a) in enumerate(att_sorted, 1):
        print(f"  {i}. {t:<20} {a:.3f}")

    print("\nTop 10 Teams by Defense (lower is better, meaning fewer goals conceded):")
    def_sorted = sorted(
        [(t, defs[idx]) for t, idx in team_idx.items()], key=lambda x: x[1]
    )[:10]
    for i, (t, d) in enumerate(def_sorted, 1):
        print(f"  {i}. {t:<20} {d:.3f}")

    import json
    # Run Monte Carlo
    prob_matrix = precompute_match_probabilities(model, team_idx)
    win_counts, final_counts, sf_counts, group_adv_counts = simulate_tournament(
        prob_matrix, n_sims=n_sims
    )
    print_mc_report(win_counts, final_counts, sf_counts, n_sims)

    # Save results to JSON for website
    out_data = {"teams": {}, "prob_matrix": {}, "n_sims": n_sims}
    ALL_TEAMS = [t for g in GROUPS.values() for t in g]
    for t in ALL_TEAMS:
        out_data["teams"][t] = {
            "win_prob": win_counts.get(t, 0) / n_sims,
            "final_prob": final_counts.get(t, 0) / n_sims,
            "sf_prob": sf_counts.get(t, 0) / n_sims,
            "group_adv_prob": group_adv_counts.get(t, 0) / n_sims,
            "group": next((g for g, teams in GROUPS.items() if t in teams), None),
        }

    for (t1, t2), probs in prob_matrix.items():
        key = f"{t1}::{t2}"
        out_data["prob_matrix"][key] = [float(probs[0]), float(probs[1])]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "predictions_dc.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/kaggle/input/datasets/martj42/international-football-results-from-1872-to-2017results.csv",
        help="Path to results.csv",
    )
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_known_args()[0]

    main(csv_path=args.csv, epochs=args.epochs, n_sims=args.sims)
