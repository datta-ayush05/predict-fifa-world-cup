from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="WC Predict API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "predictions.json")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Prediction data not found. Please run the model simulation first.")
    with open(DATA_PATH, "r") as f:
        return json.load(f)

@app.get("/api/teams")
def get_teams():
    data = load_data()
    # return sorted list of teams
    return sorted(list(data.get("teams", {}).keys()))

@app.get("/api/all_stats")
def get_all_stats():
    data = load_data()
    teams = data.get("teams", {})
    # Return as list of objects, sorted by win_prob descending
    team_list = []
    for name, stats in teams.items():
        team_list.append({
            "name": name,
            "win_prob": stats.get("win_prob", 0),
            "final_prob": stats.get("final_prob", 0),
            "sf_prob": stats.get("sf_prob", 0),
        })
    return sorted(team_list, key=lambda x: x["win_prob"], reverse=True)

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
    # Normalize (since we cut off at 10)
    total = win + draw + loss
    return {"win": win/total, "draw": draw/total, "loss": loss/total}

@app.get("/api/team/{team_name}")
def get_team_stats(team_name: str):
    data = load_data()
    teams = data.get("teams", {})
    if team_name not in teams:
        raise HTTPException(status_code=404, detail="Team not found")
    
    stats = teams[team_name]
    
    # gather match probs
    prob_matrix = data.get("prob_matrix", {})
    matches = []
    
    group = stats.get("group")
    
    # Just a sample, getting probabilities against teams in same group
    for t2, t2_stats in teams.items():
        if t2 != team_name and t2_stats.get("group") == group:
            key1 = f"{team_name}::{t2}"
            key2 = f"{t2}::{team_name}"
            if key1 in prob_matrix:
                probs = prob_matrix[key1]
                match_p = calculate_match_probs(probs[0], probs[1])
                matches.append({
                    "opponent": t2,
                    "xG_for": probs[0],
                    "xG_against": probs[1],
                    "win_prob": match_p["win"],
                    "draw_prob": match_p["draw"],
                    "loss_prob": match_p["loss"]
                })
            elif key2 in prob_matrix:
                probs = prob_matrix[key2]
                match_p = calculate_match_probs(probs[1], probs[0])
                matches.append({
                    "opponent": t2,
                    "xG_for": probs[1], # reversed
                    "xG_against": probs[0],
                    "win_prob": match_p["win"],
                    "draw_prob": match_p["draw"],
                    "loss_prob": match_p["loss"]
                })
                
    return {
        "stats": stats,
        "group_matches": matches,
        "n_sims": data.get("n_sims", 0)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
