import json
import os

fixtures_path = os.path.join("src", "web", "frontend", "src", "data", "fixtures.json")

if os.path.exists(fixtures_path):
    with open(fixtures_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Keep only Group Stage matches (which start with 'group_')
    cleaned_matches = [m for m in data.get("matches", []) if str(m.get("match_id", "")).startswith("group_")]
    
    with open(fixtures_path, "w", encoding="utf-8") as f:
        json.dump({"matches": cleaned_matches}, f, indent=2)
        
    print(f"Cleaned {len(data.get('matches', [])) - len(cleaned_matches)} knockout matches from fixtures.json.")
    print("fixtures.json is now ready for a clean rebuild starting from June 28!")
else:
    print("fixtures.json not found.")
