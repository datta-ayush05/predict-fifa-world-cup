"""
Wrapper script to simulate the World Cup from a specific knockout stage onwards.
"""

import argparse
from collections import defaultdict
import torch
from tqdm import tqdm
import os
import csv

import wc_predict_gnn

# ==============================================================================
# CONFIGURABLE MATCHUPS
# Edit these lists with the actual matchups as they are decided in the tournament.
# Make sure team names match those in `wc_predict_gnn.NAME_MAP` / `results.csv`.
# ==============================================================================

MATCHUPS_R32 = [
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("Belgium", "Senegal"),
    ("USA", "Bosnia and Herzegovina"),
    ("Spain", "Austria"),
    ("Portugal", "Croatia"),
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
    ("Australia", "Egypt"),
    ("Argentina", "Cape Verde")
]

MATCHUPS_R16 = [
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
]

MATCHUPS_QF = [
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
    ("TBD", "TBD"),
]

MATCHUPS_SF = [
    ("TBD", "TBD"),
    ("TBD", "TBD"),
]

MATCHUPS_FINAL = [
    ("TBD", "TBD")
]

MATCHUPS_THIRD_PLACE = [
    ("TBD", "TBD")
]

MATCHUPS_BY_STAGE = {
    "r32": MATCHUPS_R32,
    "r16": MATCHUPS_R16,
    "qf": MATCHUPS_QF,
    "sf": MATCHUPS_SF,
    "third_place": MATCHUPS_THIRD_PLACE,
    "final": MATCHUPS_FINAL,
}

# ==============================================================================

def simulate_from_stage(
    stage: str,
    matchups: list,
    model,
    graph,
    team_idx,
    current_elos,
    n_sims=10000
):
    # Load locked winners from knockouts.csv
    locked_winners_map = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    knockouts_path = os.path.join(script_dir, "..", "..", "data", "knockouts.csv")
    if os.path.exists(knockouts_path):
        with open(knockouts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t1 = wc_predict_gnn.NAME_MAP.get(row["team1"], row["team1"])
                t2 = wc_predict_gnn.NAME_MAP.get(row["team2"], row["team2"])
                winner = wc_predict_gnn.NAME_MAP.get(row["winner"], row["winner"])
                if winner:
                    locked_winners_map[f"{t1}::{t2}"] = winner
                    locked_winners_map[f"{t2}::{t1}"] = winner
    device = next(model.parameters()).device
    graph = graph.to(device)

    win_counts = defaultdict(int)
    final_counts = defaultdict(int)
    sf_counts = defaultdict(int)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph)

    prob_matrix = wc_predict_gnn.precompute_match_probabilities(
        model, embeddings, team_idx, current_elos
    )

    print(f"\n{'=' * 60}")
    print(f"  Running {n_sims} Monte Carlo simulations starting from {stage.upper()} …")
    print(f"{'=' * 60}")

    for sim in tqdm(range(n_sims), desc="Simulating"):
        current_matchups = list(matchups)
        current_stage = stage

        # Step through the remaining stages based on where we start
        stages = ["r32", "r16", "qf", "sf", "final"]
        start_idx = stages.index(current_stage)

        for i in range(start_idx, len(stages)):
            stage_name = stages[i]
            next_winners = []
            
            for match in current_matchups:
                ta = match[0]
                tb = match[1]
                locked_winner = locked_winners_map.get(f"{ta}::{tb}")

                # If we are in SF, keep track of who played for reporting
                if stage_name == "sf":
                    sf_counts[ta] += 1
                    sf_counts[tb] += 1
                
                if locked_winner:
                    next_winners.append(locked_winner)
                else:
                    winner, res = wc_predict_gnn.simulate_match(
                        model, embeddings, ta, tb, team_idx, current_elos,
                        "neutral", knockout=True, prob_matrix=prob_matrix
                    )
                    next_winners.append(winner)

            if stage_name == "final":
                # The winner of the final is the tournament winner
                win_counts[next_winners[0]] += 1
                # Both finalists get recorded
                final_counts[current_matchups[0][0]] += 1
                final_counts[current_matchups[0][1]] += 1
            else:
                # Pair the winners for the next round (adjacent pairs)
                current_matchups = [(next_winners[i], next_winners[i+1]) for i in range(0, len(next_winners), 2)]

    return win_counts, final_counts, sf_counts, prob_matrix

def sync_knockouts_to_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    knockouts_path = os.path.join(data_dir, "knockouts.csv")
    results_path = os.path.join(data_dir, "results.csv")
    
    if not os.path.exists(knockouts_path):
        return
        
    existing_results = set()
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_results.add((row["date"], row["home_team"], row["away_team"]))
            
    new_rows = []
    with open(knockouts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["date"], row["team1"], row["team2"])
            if key not in existing_results:
                new_rows.append(
                    f"{row['date']},{row['team1']},{row['team2']},{row['score1']},{row['score2']},FIFA World Cup,{row['city']},{row['country']},{row['neutral']}\n"
                )
                existing_results.add(key)
                
    if new_rows:
        with open(results_path, "a", encoding="utf-8") as f:
            f.writelines(new_rows)
        print(f"> Synced {len(new_rows)} new matches from knockouts.csv to results.csv")

def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="Predict from a specific stage")
    parser.add_argument(
        "--stage", 
        type=str, 
        choices=["r32", "r16", "qf", "sf", "final"], 
        required=True,
        help="The stage to start simulating from"
    )
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations")
    args = parser.parse_args()

    sync_knockouts_to_results()

    matchups = MATCHUPS_BY_STAGE[args.stage]
    
    # Map country names to canonical names
    mapped_matchups = []
    for match in matchups:
        ta = match[0]
        tb = match[1]
        ta_mapped = wc_predict_gnn.NAME_MAP.get(ta, ta)
        tb_mapped = wc_predict_gnn.NAME_MAP.get(tb, tb)
        mapped_matchups.append((ta_mapped, tb_mapped))
    matchups = mapped_matchups
    
    # Check if matchups are provided
    if not matchups:
        print(f"Error: No matchups defined for stage '{args.stage}'. Please edit the script to add them.")
        return

    # Check validity of matchups count
    expected_matches = {"r32": 16, "r16": 8, "qf": 4, "sf": 2, "final": 1}
    if len(matchups) != expected_matches[args.stage]:
        print(f"Warning: Expected {expected_matches[args.stage]} matches for {args.stage}, but found {len(matchups)}.")

    print(f"\n> Training model and preparing up to date ELOs...")
    print(f"  Note: This skips the default full simulation to prepare data quickly.")
    
    # Monkey patch original script to silence its own dummy simulation output
    original_sim = wc_predict_gnn.simulate_tournament
    wc_predict_gnn.simulate_tournament = lambda *args, **kwargs: (defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), kwargs.get("n_sims", 1))
    original_report = wc_predict_gnn.print_mc_report
    wc_predict_gnn.print_mc_report = lambda *args, **kwargs: None

    # Load model and ELOs by running main with n_sims=1
    model, graph, team_idx, current_elos = wc_predict_gnn.main(
        epochs=40,
        n_sims=1
    )
    
    # Restore original functions just in case
    wc_predict_gnn.simulate_tournament = original_sim
    wc_predict_gnn.print_mc_report = original_report

    # Run the stage simulation
    win_counts, final_counts, sf_counts, prob_matrix = simulate_from_stage(
        stage=args.stage,
        matchups=matchups,
        model=model,
        graph=graph,
        team_idx=team_idx,
        current_elos=current_elos,
        n_sims=args.sims
    )

    # Print Report
    print(f"\n{'=' * 60}")
    print(f"  PREDICTION RESULTS FROM {args.stage.upper()} ONWARDS  (n={args.sims})")
    print(f"{'=' * 60}")
    print(f"\n  {'Team':<30} {'Win%':>7} {'Final%':>8} {'SF%':>6}")
    print(f"  {'-' * 54}")

    # Gather all teams involved in the starting bracket
    teams_in_bracket = set()
    for match in matchups:
        teams_in_bracket.add(match[0])
        teams_in_bracket.add(match[1])

    sorted_teams = sorted(teams_in_bracket, key=lambda t: win_counts.get(t, 0), reverse=True)
    
    for t in sorted_teams:
        wp = win_counts.get(t, 0) / args.sims * 100
        fp = final_counts.get(t, 0) / args.sims * 100
        sp = sf_counts.get(t, 0) / args.sims * 100
        
        # Override values if we started at or past these stages
        if args.stage in ["sf", "final"]:
            sp = 100.0
        if args.stage == "final":
            fp = 100.0
            
        print(f"  {t:<30} {wp:>6.1f}%  {fp:>6.1f}%  {sp:>5.1f}%")
        
    # Save results to JSON
    import json
    out_data = {"teams": {}, "prob_matrix": {}, "n_sims": args.sims, "starting_stage": args.stage}
    for t in wc_predict_gnn.ALL_TEAMS:
        wp_frac = win_counts.get(t, 0) / args.sims
        fp_frac = final_counts.get(t, 0) / args.sims
        sp_frac = sf_counts.get(t, 0) / args.sims
        
        if args.stage in ["sf", "final"] and t in teams_in_bracket:
            sp_frac = 1.0
        if args.stage == "final" and t in teams_in_bracket:
            fp_frac = 1.0
            
        out_data["teams"][t] = {
            "win_prob": wp_frac,
            "final_prob": fp_frac,
            "sf_prob": sp_frac,
            "group_adv_prob": 1.0 if t in teams_in_bracket else 0.0,
            "group": next((g for g, teams in wc_predict_gnn.GROUPS.items() if t in teams), None),
        }

    for (t1, t2), probs in prob_matrix.items():
        key = f"{t1}::{t2}"
        out_data["prob_matrix"][key] = [float(probs[0]), float(probs[1])]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved updated predictions to {out_path}")

if __name__ == "__main__":
    main()
