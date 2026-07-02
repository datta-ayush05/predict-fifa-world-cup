import os
import subprocess

def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    
    print("=" * 50)
    print("  Daily Update Script")
    print("=" * 50)
    
    # 1. Lock in the true pre-match odds for today's newly finished matches
    print("\n[1/3] Locking in pre-match odds for today's results...")
    run_cmd(f'"{python_exe}" generate_fixtures.py')
    
    # 2. Sync to results.csv, update Elo, and run simulations for the future
    print("\n[2/3] Updating Elo ratings and running 10,000 simulations...")
    run_cmd(f'"{python_exe}" src/predict_wc/predict_stage.py --stage r32 --sims 10000')
    
    # 3. Update the frontend JSON with the new odds for tomorrow's upcoming matches
    print("\n[3/3] Updating frontend odds for upcoming matches...")
    run_cmd(f'"{python_exe}" generate_fixtures.py')
    
    print("\nDone! Frontend is perfectly up to date without temporal leakage.")
