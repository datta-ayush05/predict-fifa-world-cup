import os
import csv
import subprocess

def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, shell=True, check=True, env=env)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
results_path = os.path.join(data_dir, "results.csv")
fixtures_path = os.path.join(script_dir, "src", "web", "frontend", "src", "data", "fixtures.json")
python_exe = os.path.join(script_dir, ".venv", "Scripts", "python.exe")

# 1. Read all rows from results.csv
all_rows = []
with open(results_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        all_rows.append(row)

# Separate into pre-WC and WC matches
pre_wc = []
wc_2026 = []

for row in all_rows:
    if row[0].startswith("2026") and row[5] == "FIFA World Cup":
        wc_2026.append(row)
    else:
        pre_wc.append(row)

# Get unique dates in WC
wc_dates = sorted(list(set([row[0] for row in wc_2026])))

# 2. Do NOT delete fixtures.json, so we preserve the correct historical predictions for days prior to June 30
# if os.path.exists(fixtures_path):
#     os.remove(fixtures_path)

# 3. Process day by day (Resuming from June 30)
current_matches = [row for row in wc_2026 if row[0] < "2026-06-30"]
wc_dates = [d for d in wc_dates if d >= "2026-06-30"]

for date in wc_dates:
    print(f"\n{'='*40}")
    print(f"  Processing Date: {date}")
    print(f"{'='*40}")
    
    # Write results.csv with matches strictly BEFORE date
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(pre_wc)
        writer.writerows(current_matches)
        
    # Run GNN to get pre-match predictions (n_sims=1 is fast)
    print("> Running GNN to compute pure pre-match probabilities...")
    run_cmd(f'"{python_exe}" src/predict_wc/wc_predict_gnn.py --sims 1')
    
    # Get matches for the CURRENT date
    matches_today = [row for row in wc_2026 if row[0] == date]
    current_matches.extend(matches_today)
    
    # Write results.csv with matches UP TO AND INCLUDING date
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(pre_wc)
        writer.writerows(current_matches)
        
    # Run generate_fixtures.py to lock in today's predictions
    print(f"> Generating fixtures for {date}...")
    run_cmd(f'"{python_exe}" generate_fixtures.py')

# 4. Final pass for the future (Upcoming matches)
print(f"\n{'='*40}")
print("  Running final prediction pass for upcoming matches")
print(f"{'='*40}")
run_cmd(f'"{python_exe}" src/predict_wc/predict_stage.py --stage r32 --sims 10000')
run_cmd(f'"{python_exe}" generate_fixtures.py')

print("\nHistorical backfill complete! fixtures.json is fully restored with true pre-match probabilities.")
