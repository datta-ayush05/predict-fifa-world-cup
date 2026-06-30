import csv
import json
import os
import math

def calculate_match_probs(lambda_h, lambda_a):
    win = 0.0
    draw = 0.0
    loss = 0.0
    for i in range(10):
        for j in range(10):
            p = (math.exp(-lambda_h) * (lambda_h**i) / math.factorial(i)) * \
                (math.exp(-lambda_a) * (lambda_a**j) / math.factorial(j))
            if i > j:
                win += p
            elif i == j:
                draw += p
            else:
                loss += p
    total = win + draw + loss
    if total == 0:
        return {"win": 0, "draw": 0, "loss": 0}
    return {"win": win/total, "draw": draw/total, "loss": loss/total}

# Load prob_matrix from predictions.json
predictions_path = os.path.join("data", "predictions.json")
with open(predictions_path, "r") as f:
    pred_data = json.load(f)
prob_matrix = pred_data.get("prob_matrix", {})

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "predict_wc"))
try:
    from predict_stage import MATCHUPS_BY_STAGE
except ImportError as e:
    print(f"Failed to import: {e}")
    MATCHUPS_BY_STAGE = {}

stage_names = {
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarter-Final",
    "sf": "Semi-Final",
    "third_place": "Third Place Play-off",
    "final": "Final"
}

def normalize_name(name):
    if name == "USA": return "United States"
    if name == "Curaçao": return "Curacao"
    return name

knockout_lookup = {}
for stage_key, matchups in MATCHUPS_BY_STAGE.items():
    s_name = stage_names.get(stage_key, stage_key.upper())
    for m in matchups:
        t1_norm = normalize_name(m[0])
        t2_norm = normalize_name(m[1])
        knockout_lookup[f"{t1_norm}::{t2_norm}"] = s_name
        knockout_lookup[f"{t2_norm}::{t1_norm}"] = s_name

out_dir = os.path.join("src", "web", "frontend", "src", "data")
out_path = os.path.join(out_dir, "fixtures.json")

# Load existing fixtures to preserve predictions
existing_fixtures = {}
if os.path.exists(out_path):
    with open(out_path, "r", encoding="utf-8") as f:
        try:
            old_data = json.load(f)
            for m in old_data.get("matches", []):
                existing_fixtures[m["match_id"]] = m
        except:
            pass

fixtures = []

fixtures = []

# 1. Parse Group Stage from results.csv
results_csv_path = os.path.join("data", "results.csv")
with open(results_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["date"].startswith("2026") and row["tournament"] == "FIFA World Cup":
            t1 = normalize_name(row["home_team"])
            t2 = normalize_name(row["away_team"])
            score1 = int(row["home_score"]) if row["home_score"] else None
            score2 = int(row["away_score"]) if row["away_score"] else None
            
            stage_name_for_match = knockout_lookup.get(f"{t1}::{t2}", "Group Stage")
            match_id_prefix = "group" if stage_name_for_match == "Group Stage" else "knockout"
            match_id = f"{match_id_prefix}_{row['date']}_{t1}_{t2}".replace(" ", "")
            
            if match_id in existing_fixtures and existing_fixtures[match_id]["result"].get("status") != "Upcoming":
                fixture = existing_fixtures[match_id]
                fixture["result"]["score1"] = score1
                fixture["result"]["score2"] = score2
                fixture["result"]["status"] = "Finished" if score1 is not None else "Upcoming"
                fixtures.append(fixture)
            else:
                # Find prediction
                key1 = f"{t1}::{t2}"
                key2 = f"{t2}::{t1}"
                
                xg1, xg2 = 0.0, 0.0
                if key1 in prob_matrix:
                    xg1, xg2 = prob_matrix[key1]
                elif key2 in prob_matrix:
                    xg2, xg1 = prob_matrix[key2]
                
                probs = calculate_match_probs(xg1, xg2)
                
                fixtures.append({
                    "match_id": match_id,
                    "stage": stage_name_for_match,
                    "team1": t1,
                    "team2": t2,
                    "date": row["date"],
                    "prediction": {
                        "win_prob": probs["win"],
                        "draw_prob": probs["draw"],
                        "loss_prob": probs["loss"],
                        "xG1": xg1,
                        "xG2": xg2
                    },
                    "result": {
                        "score1": score1,
                        "score2": score2,
                        "status": "Finished" if score1 is not None else "Upcoming"
                    }
                })

# 2. Add Knockout matches from predict_stage.py
# (Only those that were not already added from results.csv)

for stage_key, matchups in MATCHUPS_BY_STAGE.items():
    stage_name = stage_names.get(stage_key, stage_key.upper())
    
    for i, match_tuple in enumerate(matchups):
        t1 = normalize_name(match_tuple[0])
        t2 = normalize_name(match_tuple[1])
        
        # Check if already added from results.csv
        already_added = False
        for f_existing in fixtures:
            if f_existing["team1"] in [t1, t2] and f_existing["team2"] in [t1, t2]:
                already_added = True
                break
        
        if already_added:
            continue
            
        match_id = f"{stage_key}_{i}_{t1}_{t2}".replace(" ", "")
        
        # If the match already exists and is NOT upcoming (i.e. played), preserve its prediction
        if match_id in existing_fixtures and existing_fixtures[match_id]["result"].get("status") != "Upcoming":
            fixtures.append(existing_fixtures[match_id])
        else:
            key1 = f"{t1}::{t2}"
            key2 = f"{t2}::{t1}"
            
            xg1, xg2 = 0.0, 0.0
            if key1 in prob_matrix:
                xg1, xg2 = prob_matrix[key1]
            elif key2 in prob_matrix:
                xg2, xg1 = prob_matrix[key2]
            else:
                xg1, xg2 = 1.0, 1.0
                
            probs = calculate_match_probs(xg1, xg2)
            
            is_finished = len(match_tuple) > 2 and match_tuple[2] not in [None, "TBD", ""]
            
            fixtures.append({
                "match_id": match_id,
                "stage": stage_name,
                "team1": t1,
                "team2": t2,
                "date": "TBD",
                "prediction": {
                    "win_prob": probs["win"],
                    "draw_prob": probs["draw"],
                    "loss_prob": probs["loss"],
                    "xG1": xg1,
                    "xG2": xg2
                },
                "result": {
                    "score1": None,
                    "score2": None,
                    "status": "Finished" if is_finished else "Upcoming",
                    "winner": normalize_name(match_tuple[2]) if is_finished else None
                }
            })

# Output to frontend data folder
os.makedirs(out_dir, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"matches": fixtures}, f, indent=2)

print(f"Generated {len(fixtures)} fixtures and saved to {out_path}")
