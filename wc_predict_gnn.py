"""
FIFA World Cup 2026 Prediction Model
=====================================
Graph Neural Network with:
- Continuous time-decay weighting (exponential decay favoring recent matches)
- Rolling ELO ratings integrated as node/edge features
- Latent Roster Strengths extracted from multiple EA FC player datasets (position-weighted)
- Expected Goals directly output via Softplus activation and Poisson NLL training
- Native Poisson sampling for match goal simulations and outcomes
- Home / Away / Neutral venue adjustment
- Penalty shootout probability (moderate penalty)
- Official 2026 format: 12 groups → R32 → R16 → QF → SF → Final
- Third-place best-of-8 qualification for R32

Dependencies:
    pip install torch torch-geometric pandas numpy tqdm

Datasets:
    1. results.csv from Kaggle "International Soccer Results from 1872"
       https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    2. Multiple EA FC Player Ratings datasets (e.g., FIFA 22 - FC 26) mapped chronologically to walk-forward training windows.
"""

import math
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
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

# Official R32 bracket structure (FIFA 2026)
# Winners / Runners-up → fixed slots; 3rd place → filled by 8 best from 12 groups
# Pathway A: groups A-D   Pathway B: groups E-H   Pathway C: groups I-L
# (simplified bracket; exact slots depend on which 8 third-place teams advance)
R32_FIXED_MATCHUPS = [
    # (slot_desc, winner_group, runner_group)  — playoff phase
    ("1A", "A", "B"),  # 1st A vs 2nd B
    ("1B", "B", "A"),  # 1st B vs 2nd A
    ("1C", "C", "D"),
    ("1D", "D", "C"),
    ("1E", "E", "F"),
    ("1F", "F", "E"),
    ("1G", "G", "H"),
    ("1H", "H", "G"),
    ("1I", "I", "J"),
    ("1J", "J", "I"),
    ("1K", "K", "L"),
    ("1L", "L", "K"),
    # 8 third-place slots filled dynamically
]

# Name normalisation map (results.csv ↔ our group list)
NAME_MAP = {
    # Manual overrides
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
    # Mappings from former_names.csv
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
    # Graph-density historical merges (split/merged nations)
    "German DR": "Germany",
    "Yugoslavia": "Serbia",
    "Vietnam Republic": "Vietnam",
    "North Vietnam": "Vietnam",
    "South Yemen": "Yemen",
    "Yemen DPR": "Yemen",
}

ALL_TEAMS = sorted({t for teams in GROUPS.values() for t in teams})


# ─────────────────────────────────────────────
# 2. ELO RATING ENGINE
# ─────────────────────────────────────────────


class EloSystem:
    """
    Rolling ELO system with:
    - K-factor based on match importance (WC > qualifier > friendly)
    - Home advantage offset
    - Temporal snapshots stored for feature extraction
    """

    BASE_ELO = 1500
    K_FACTORS = {
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
    DEFAULT_K = 35
    HOME_ADV = 100  # ELO points added to home side expected score

    def __init__(self):
        self.ratings: dict[str, float] = defaultdict(lambda: self.BASE_ELO)
        self.history: dict[str, list] = defaultdict(list)  # team → [(date, elo)]

    def _k(self, tournament: str) -> float:
        for key, val in self.K_FACTORS.items():
            if key.lower() in tournament.lower():
                return val
        return self.DEFAULT_K

    def _expected(
        self, elo_a: float, elo_b: float, home_side: str, team_a: str
    ) -> float:
        """Expected score for team A."""
        delta = elo_a - elo_b
        if home_side == team_a:
            delta += self.HOME_ADV
        elif home_side not in ("", "neutral", None):
            delta -= self.HOME_ADV
        return 1 / (1 + 10 ** (-delta / 400))

    def update(self, row: pd.Series):
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aw = int(row["away_score"])
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", ""))
        date = row["date"]

        ra = self.ratings[home]
        rb = self.ratings[away]

        venue = "" if neutral else home
        ea = self._expected(ra, rb, venue, home)
        eb = 1 - ea

        if hs > aw:
            sa, sb = 1.0, 0.0
        elif hs < aw:
            sa, sb = 0.0, 1.0
        else:
            sa = sb = 0.5

        k = self._k(tournament)
        self.ratings[home] = ra + k * (sa - ea)
        self.ratings[away] = rb + k * (sb - eb)

        self.history[home].append((date, self.ratings[home]))
        self.history[away].append((date, self.ratings[away]))

    def get_elo(self, team: str) -> float:
        return self.ratings[team]

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)

    def snapshot_at(self, date: pd.Timestamp) -> dict[str, float]:
        snap = {}
        for team, hist in self.history.items():
            last_elo = self.BASE_ELO
            for d, e in hist:
                if d < date:
                    last_elo = e
                else:
                    break
            snap[team] = last_elo
        return snap


# ─────────────────────────────────────────────
# 3. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────


def load_and_prepare(csv_path: str = "results.csv"):
    print("Loading results.csv …")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise names
    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)

    return df


def build_elo_series(df: pd.DataFrame) -> tuple[EloSystem, pd.DataFrame]:
    """Run ELO updates in chronological order; attach pre-match ELO to each row."""
    elo = EloSystem()
    pre_home, pre_away = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building ELO"):
        pre_home.append(elo.get_elo(row["home_team"]))
        pre_away.append(elo.get_elo(row["away_team"]))
        elo.update(row)

    df = df.copy()
    df["elo_home"] = pre_home
    df["elo_away"] = pre_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return elo, df


def parse_ea_fc_dataset(csv_path: str) -> dict:
    """Loads EA FC datasets of varying formats, computes position-weighted scores,
    selects top 23 players per nation, and returns normalized strengths.
    """
    print(f"\n-> Extracting Latent Roster Strengths from {csv_path}...")

    # Read first row to determine columns
    try:
        header_df = pd.read_csv(csv_path, nrows=0)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return {}
    cols = header_df.columns.tolist()

    # Mapping dynamic columns to unified schema
    col_map = {}
    for c in cols:
        cl = str(c).strip().lower()
        if cl in ["overallrating", "overall", "ovr"]:
            col_map[c] = "overallRating"
        elif cl in ["position", "best position"]:
            col_map[c] = "position"
        elif cl in ["nationality", "nation", "national team name"]:
            col_map[c] = "nationality"
        elif cl in ["pac", "pace total"]:
            col_map[c] = "pac"
        elif cl in ["sho", "shooting total"]:
            col_map[c] = "sho"
        elif cl in ["pas", "passing total"]:
            col_map[c] = "pas"
        elif cl in ["dri", "dribbling total"]:
            col_map[c] = "dri"
        elif cl in ["def", "defending total", "defending"]:
            col_map[c] = "def"
        elif cl in ["phy", "physicality total", "physicality"]:
            col_map[c] = "phy"

    df = pd.read_csv(csv_path, usecols=list(col_map.keys()), low_memory=False)
    df = df.rename(columns=col_map)

    df["primary_pos"] = df["position"].str.strip().str.upper()
    pos_map = {
        "GK": "GK",
        "CB": "DEF",
        "LB": "DEF",
        "RB": "DEF",
        "LWB": "DEF",
        "RWB": "DEF",
        "CM": "MID",
        "CDM": "MID",
        "CAM": "MID",
        "LM": "MID",
        "RM": "MID",
        "ST": "FWD",
        "CF": "FWD",
        "LW": "FWD",
        "RW": "FWD",
    }
    df["pos_group"] = df["primary_pos"].map(pos_map)
    df = df.dropna(subset=["pos_group"])

    numeric_cols = ["overallRating", "pac", "sho", "pas", "dri", "def", "phy"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that are missing any of the critical stats to avoid arbitrary 0s
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])

    # Position-specific weights: [Pace, Shooting, Passing, Dribbling, Defending, Physic]
    weights = {
        "FWD": np.array([0.20, 0.40, 0.10, 0.20, 0.00, 0.10]),
        "MID": np.array([0.10, 0.15, 0.40, 0.20, 0.05, 0.10]),
        "DEF": np.array([0.15, 0.00, 0.10, 0.05, 0.45, 0.25]),
    }

    def compute_weighted_rating(row):
        group = row["pos_group"]
        if group == "GK":
            return float(row["overallRating"])
        stats = np.array(
            [row["pac"], row["sho"], row["pas"], row["dri"], row["def"], row["phy"]]
        )
        return (row["overallRating"] * 0.7) + (np.dot(stats, weights[group]) * 0.3)

    df["custom_rating"] = df.apply(compute_weighted_rating, axis=1)

    squad_strengths = {}
    grouped = df.groupby("nationality")
    roster_slots = {"GK": 3, "DEF": 7, "MID": 7, "FWD": 6}

    for nation, group in grouped:
        squad_pool = []
        for pos, count in roster_slots.items():
            squad_pool.append(
                group[group["pos_group"] == pos].nlargest(count, "custom_rating")
            )
        squad_df = pd.concat(squad_pool)
        if len(squad_df) < 15:
            continue

        top_11 = squad_df.nlargest(11, "custom_rating")
        top_11_avg = top_11["custom_rating"].mean()
        bench = squad_df.drop(top_11.index)
        bench_avg = bench["custom_rating"].mean() if len(bench) > 0 else 0
        squad_strengths[nation] = (top_11_avg * 0.75) + (bench_avg * 0.25)

    if not squad_strengths:
        return {}

    min_v, max_v = min(squad_strengths.values()), max(squad_strengths.values())

    # Map back to tournament naming convention
    normalized_squads = {}
    for nation, score in squad_strengths.items():
        standard_name = NAME_MAP.get(nation, nation)
        normalized_squads[standard_name] = (score - min_v) / (max_v - min_v)

    return normalized_squads


# ─────────────────────────────────────────────
# 4. TIME-DECAY WEIGHTING
# ─────────────────────────────────────────────

REFERENCE_DATE = pd.Timestamp("2026-06-10")
DECAY_HALF_LIFE_DAYS = 365 * 3  # 3-year half-life


def time_weight(date: pd.Timestamp, ref_date: pd.Timestamp = REFERENCE_DATE) -> float:
    """Exponential decay: matches from 3 years ago get half the weight."""
    days_ago = (ref_date - date).days
    days_ago = max(days_ago, 0)
    lam = math.log(2) / DECAY_HALF_LIFE_DAYS
    return math.exp(-lam * days_ago)


# ─────────────────────────────────────────────
# 5. GRAPH CONSTRUCTION
# ─────────────────────────────────────────────


def build_graph(
    df: pd.DataFrame,
    current_elos: dict[str, float],
    ref_date: pd.Timestamp = REFERENCE_DATE,
) -> Data:
    """
    Nodes  = all historical teams
    Edges  = historical match-ups (weighted by time-decay)
    Node features: [elo, avg_goals_scored, avg_goals_conceded, win_rate, draw_rate]
    Edge features: [elo_diff, goal_diff, venue_flag, time_weight, match_count]
    """
    all_teams_in_df = sorted(set(df["home_team"]).union(set(df["away_team"])))
    team_idx = {t: i for i, t in enumerate(all_teams_in_df)}
    n = len(all_teams_in_df)

    lam = math.log(2) / DECAY_HALF_LIFE_DAYS

    def calc_tw(d):
        days_ago = max((ref_date - d).days, 0)
        return math.exp(-lam * days_ago)

    sub = df.copy()
    sub["tw"] = sub["date"].apply(calc_tw)

    # ── Node feature accumulators ──────────────────────
    stats = {
        t: {"w": 0.0, "d": 0.0, "l": 0.0, "gf": 0.0, "ga": 0.0, "tw_sum": 0.0}
        for t in all_teams_in_df
    }

    for _, row in sub.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, aw = int(row["home_score"]), int(row["away_score"])
        tw = row["tw"]

        # Calculate opponent strength multiplier (Baseline = 1500)
        # We use max (0.5, ...) to ensure we don't accidentally divide by zero or negative
        # Use historical pre-match ELO to avoid temporal leak
        strength_h = max(0.5, row["elo_home"] / 1500.0)
        strength_a = max(0.5, row["elo_away"] / 1500.0)

        for team, opp_strength, gf, ga, result in [
            (h, strength_a, hs, aw, 1 if hs > aw else (0 if hs < aw else 0.5)),
            (a, strength_h, aw, hs, 1 if aw > hs else (0 if aw < hs else 0.5)),
        ]:
            if team in stats:
                stats[team]["tw_sum"] += tw

                # Strength of Schedule Scaling
                stats[team]["gf"] += (
                    gf * opp_strength
                ) * tw  # Scoring on strong teams is rewarded
                stats[team]["ga"] += (
                    ga / opp_strength
                ) * tw  # Conceding to weak teams is penalized
                if result == 1:
                    stats[team]["w"] += (
                        opp_strength * tw
                    )  #  Beating a stronger team = higher win value
                elif result == 0.5:
                    stats[team]["d"] += (
                        opp_strength * tw
                    )  # Drawing against a strong team = good draw value
                else:
                    stats[team]["l"] += (
                        1.0 / opp_strength
                    ) * tw  # Losing to a weak team = massive penalty

    node_feats = []
    for t in all_teams_in_df:
        s = stats[t]
        total = s["tw_sum"] + 1e-9
        node_feats.append(
            [
                current_elos.get(t, 1500) / 2000,  # normalised ELO
                s["gf"] / total,  # weighted avg goals scored
                s["ga"] / total,  # weighted avg goals conceded
                s["w"] / total,  # win rate
                s["d"] / total,  # draw rate
            ]
        )

    x = torch.tensor(node_feats, dtype=torch.float)

    # ── Edge construction ──────────────────────────────
    edge_agg: dict[tuple, dict] = {}

    for _, row in sub.iterrows():
        h, a = row["home_team"], row["away_team"]
        hi, ai = team_idx[h], team_idx[a]
        hs, aw = int(row["home_score"]), int(row["away_score"])
        tw = row["tw"]
        neutral = bool(row.get("neutral", False))
        venue_flag = 0.0 if neutral else 1.0  # 1 = home team advantage present

        key = (hi, ai)
        if key not in edge_agg:
            edge_agg[key] = {
                "elo_diff_sum": 0.0,
                "gd_sum": 0.0,
                "tw_sum": 0.0,
                "count": 0.0,
                "venue_sum": 0.0,
            }
        e = edge_agg[key]
        e["elo_diff_sum"] += (row["elo_diff"]) * tw
        e["gd_sum"] += (hs - aw) * tw
        e["tw_sum"] += tw
        e["count"] += 1
        e["venue_sum"] += venue_flag * tw

    src_list, dst_list, edge_feats = [], [], []
    for (hi, ai), e in edge_agg.items():
        tw_sum = e["tw_sum"] + 1e-9
        feat = [
            e["elo_diff_sum"] / tw_sum / 400,  # normalised elo diff
            e["gd_sum"] / tw_sum,  # avg goal diff
            e["venue_sum"] / tw_sum,  # venue factor
            min(e["tw_sum"], 10) / 10,  # total time weight (capped)
            min(e["count"], 30) / 30,  # match count (capped)
        ]
        src_list.append(hi)
        dst_list.append(ai)
        edge_feats.append(feat)
        # Reverse edge
        src_list.append(ai)
        dst_list.append(hi)
        edge_feats.append([-feat[0], -feat[1], -feat[2], feat[3], feat[4]])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), team_idx


# ─────────────────────────────────────────────
# 6. GNN MODEL
# ─────────────────────────────────────────────


class MatchGNN(nn.Module):
    """
    Graph Attention Network (2 layers) that produces team embeddings,
    then a match predictor head over concatenated pair embeddings.

    Output: [lambda_home, lambda_away] (Expected Goals)
    """

    def __init__(
        self, node_feat_dim=5, edge_feat_dim=5, hidden=64, heads=4, dropout=0.3
    ):
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(
            node_feat_dim, hidden, heads=heads, edge_dim=edge_feat_dim, dropout=dropout
        )
        self.gat2 = GATConv(
            hidden * heads, hidden, heads=1, edge_dim=edge_feat_dim, dropout=dropout
        )

        # Match predictor: concat(emb_a, emb_b, elo_diff, venue, delta_s)
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2 + 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [lambda_home, lambda_away]
        )

    def encode(self, data: Data) -> torch.Tensor:
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.gat1(x, ei, ea))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, ei, ea)
        return x  # (N, hidden)

    def predict_match(
        self,
        embeddings: torch.Tensor,
        idx_a: int,
        idx_b: int,
        elo_a: float,
        elo_b: float,
        venue: str = "neutral",
        s_a: float = 0.5,
        s_b: float = 0.5,
    ) -> torch.Tensor:
        """
        venue: 'home_a' | 'home_b' | 'neutral'
        Returns Expected Goals [lambda_a, lambda_b]
        """
        ea = embeddings[idx_a]
        eb = embeddings[idx_b]

        elo_norm = (elo_a - elo_b) / 400.0
        venue_flag = 1.0 if venue == "home_a" else (-1.0 if venue == "home_b" else 0.0)
        delta_s = s_a - s_b

        extra = torch.tensor(
            [elo_norm, venue_flag, delta_s], dtype=torch.float, device=embeddings.device
        )
        inp = torch.cat([ea, eb, extra])
        raw_lambda = self.predictor(inp)

        return F.softplus(raw_lambda)


# ─────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────


def prepare_training_samples(
    df: pd.DataFrame, team_idx: dict, ref_date: pd.Timestamp = REFERENCE_DATE
) -> dict:
    """Convert historical match rows into vectorized tensors for speed."""
    (
        hi_list,
        ai_list,
        elo_a_list,
        elo_b_list,
        venue_list,
        weight_list,
        hs_list,
        aw_list,
    ) = [], [], [], [], [], [], [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in team_idx or a not in team_idx:
            continue

        neutral = bool(row.get("neutral", False))
        hs, aw = int(row["home_score"]), int(row["away_score"])
        tw = time_weight(row["date"], ref_date)

        # Forward match (Team A = home_team, Team B = away_team)
        hi_list.append(team_idx[h])
        ai_list.append(team_idx[a])
        elo_a_list.append(float(row["elo_home"]))
        elo_b_list.append(float(row["elo_away"]))
        venue_list.append(1.0 if not neutral else 0.0)
        hs_list.append(hs)
        aw_list.append(aw)
        weight_list.append(tw)

        # Reversed match (Team A = away_team, Team B = home_team)
        # This trains the model to understand venue_flag = -1.0
        hi_list.append(team_idx[a])
        ai_list.append(team_idx[h])
        elo_a_list.append(float(row["elo_away"]))
        elo_b_list.append(float(row["elo_home"]))
        venue_list.append(-1.0 if not neutral else 0.0)
        hs_list.append(aw)
        aw_list.append(hs)
        weight_list.append(tw)

    return {
        "hi": torch.tensor(hi_list, dtype=torch.long),
        "ai": torch.tensor(ai_list, dtype=torch.long),
        "elo_a": torch.tensor(elo_a_list, dtype=torch.float),
        "elo_b": torch.tensor(elo_b_list, dtype=torch.float),
        "venue": torch.tensor(venue_list, dtype=torch.float),
        "weight": torch.tensor(weight_list, dtype=torch.float),
        "hs": torch.tensor(hs_list, dtype=torch.float),
        "aw": torch.tensor(aw_list, dtype=torch.float),
    }


def train_model(
    model: MatchGNN,
    graph: Data,
    samples: dict,
    team_idx: dict,
    roster_strengths: dict,
    epochs: int = 40,
    opt=None,
    scheduler=None,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    graph = graph.to(device)

    # Move all tensors to device
    hi, ai = samples["hi"].to(device), samples["ai"].to(device)
    elo_a, elo_b = samples["elo_a"].to(device), samples["elo_b"].to(device)
    venue, weight = (
        samples["venue"].to(device),
        samples["weight"].to(device),
    )
    actual_hs = samples["hs"].to(device)
    actual_aw = samples["aw"].to(device)

    # Vectorized Train/Val split indices (Chronological split to prevent temporal leak)
    num_samples = len(venue)
    split = int(0.85 * num_samples)
    split -= split % 2  # Ensure even split so forward/reversed pairs aren't torn
    train_idx = torch.arange(split)
    val_idx = torch.arange(split, num_samples)

    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Precompute squad strength array
    max_idx = max(team_idx.values()) if team_idx else 0
    strength_array = np.zeros(max_idx + 1, dtype=np.float32)
    for team, idx in team_idx.items():
        strength_array[idx] = (
            roster_strengths.get(team, 0.5) if roster_strengths else 0.5
        )
    s_tensor = torch.tensor(strength_array, device=device)
    s_a = s_tensor[hi]
    s_b = s_tensor[ai]
    delta_s = (s_a - s_b).unsqueeze(1)  # shape (N, 1)

    print("\n🚀 Starting Fast Vectorized Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # 1. Multi-layer GNN forwarding pass (Done once per epoch!)
        embeddings = model.encode(graph)

        # 2. Batch compute match predictions vectorially
        ea = embeddings[hi]
        eb = embeddings[ai]
        elo_norm = ((elo_a - elo_b) / 400.0).unsqueeze(1)

        venue_col = venue.unsqueeze(1)
        extra = torch.cat([elo_norm, venue_col, delta_s], dim=1)
        inp = torch.cat([ea, eb, extra], dim=1)

        # Pass through the predictor head
        raw_lambda = model.predictor(inp)

        lambdas = F.softplus(raw_lambda)
        lambda_h = lambdas[:, 0]
        lambda_a = lambdas[:, 1]

        # Calculate loss only for training slice
        # Poisson Negative Log-Likelihood Loss: Loss = lambda - actual * log(lambda)
        loss_h = lambda_h[train_idx] - actual_hs[train_idx] * torch.log(
            lambda_h[train_idx] + 1e-8
        )
        loss_a = lambda_a[train_idx] - actual_aw[train_idx] * torch.log(
            lambda_a[train_idx] + 1e-8
        )

        loss_all = loss_h + loss_a

        batch_weights = weight[train_idx]
        batch_weights = batch_weights / (batch_weights.mean() + 1e-8)

        train_loss = (loss_all * batch_weights).mean()

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # Quick validation check
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_emb = model.encode(graph)
                v_ea, v_eb = val_emb[hi[val_idx]], val_emb[ai[val_idx]]
                v_inp = torch.cat([v_ea, v_eb, extra[val_idx]], dim=1)
                v_raw_lambda = model.predictor(v_inp)
                v_lambdas = F.softplus(v_raw_lambda)

                v_lambda_h = v_lambdas[:, 0]
                v_lambda_a = v_lambdas[:, 1]
                v_loss_h = v_lambda_h - actual_hs[val_idx] * torch.log(
                    v_lambda_h + 1e-8
                )
                v_loss_a = v_lambda_a - actual_aw[val_idx] * torch.log(
                    v_lambda_a + 1e-8
                )
                val_loss = (v_loss_h + v_loss_a).mean().item()

            print(
                f"  Epoch {epoch:3d} | train_loss={train_loss.item():.4f} | val_loss={val_loss:.4f}"
            )

    return model


# ─────────────────────────────────────────────
# 8. MATCH SIMULATION
# ─────────────────────────────────────────────

PENALTY_SHOOTOUT_PROB_EACH = 0.5  # 50/50 in shootout (moderate penalty factor)


def precompute_match_probabilities(
    model, embeddings, team_idx, current_elos, roster_strengths
):
    """Precompute all expected goals (lambdas) for O(1) speedup Monte Carlo."""
    print(
        "⚡ Precomputing static expected goals (Host advantages applied) for O(1) lookup..."
    )
    prob_matrix = {}
    teams = list(team_idx.keys())
    for ta in teams:
        for tb in teams:
            if ta == tb:
                continue
            ia = team_idx.get(ta)
            ib = team_idx.get(tb)
            ea = current_elos.get(ta, 1500)
            eb = current_elos.get(tb, 1500)
            s_a = roster_strengths.get(ta, 0.5) if roster_strengths else 0.5
            s_b = roster_strengths.get(tb, 0.5) if roster_strengths else 0.5

            hosts = ["United States", "Mexico", "Canada"]
            venue_1 = "neutral"
            venue_2 = "neutral"

            if ta in hosts and tb not in hosts:
                venue_1 = "home_a"
                venue_2 = "home_b"
            elif tb in hosts and ta not in hosts:
                venue_1 = "home_b"
                venue_2 = "home_a"

            with torch.no_grad():
                lambdas1 = model.predict_match(
                    embeddings, ia, ib, ea, eb, venue_1, s_a, s_b
                )
                lambdas2 = model.predict_match(
                    embeddings, ib, ia, eb, ea, venue_2, s_b, s_a
                )

                # Average the forward and reversed lambdas for stability
                gnn_lambdas = np.array(
                    [
                        (lambdas1[0].item() + lambdas2[1].item()) / 2.0,
                        (lambdas1[1].item() + lambdas2[0].item()) / 2.0,
                    ]
                )

            prob_matrix[(ta, tb)] = gnn_lambdas
    return prob_matrix


def simulate_match(
    model: MatchGNN,
    embeddings: torch.Tensor,
    team_a: str,
    team_b: str,
    team_idx: dict,
    current_elos: dict,
    roster_strengths: dict = None,
    venue: str = "neutral",
    knockout: bool = False,
    deterministic: bool = False,
    prob_matrix: dict = None,
) -> tuple[str, dict]:
    """
    Returns (winner_or_none, result_dict).
    Draws goals directly from Poisson distribution using Expected Goals.
    """
    ia = team_idx.get(team_a)
    ib = team_idx.get(team_b)
    ea = current_elos.get(team_a, 1500)
    eb = current_elos.get(team_b, 1500)

    if prob_matrix is not None and (team_a, team_b) in prob_matrix:
        final_lambdas = prob_matrix[(team_a, team_b)]
    else:
        if venue == "home_a":
            v1, v2 = "home_a", "home_b"
        elif venue == "home_b":
            v1, v2 = "home_b", "home_a"
        else:
            v1, v2 = "neutral", "neutral"

        if ia is None or ib is None:
            # fallback: pure ELO
            p_a = 1 / (1 + 10 ** (-(ea - eb) / 400))
            final_lambdas = np.array([p_a * 2.6, (1 - p_a) * 2.6])
        else:
            s_a = roster_strengths.get(team_a, 0.5) if roster_strengths else 0.5
            s_b = roster_strengths.get(team_b, 0.5) if roster_strengths else 0.5
            with torch.no_grad():
                lambdas1 = model.predict_match(embeddings, ia, ib, ea, eb, v1, s_a, s_b)
                lambdas2 = model.predict_match(embeddings, ib, ia, eb, ea, v2, s_b, s_a)

                # Average the outputs
                final_lambdas = np.array(
                    [
                        (lambdas1[0].item() + lambdas2[1].item()) / 2.0,
                        (lambdas1[1].item() + lambdas2[0].item()) / 2.0,
                    ]
                )

    if deterministic:
        gf_a = round(final_lambdas[0])
        gf_b = round(final_lambdas[1])
    else:
        gf_a = np.random.poisson(final_lambdas[0])
        gf_b = np.random.poisson(final_lambdas[1])

    winner = team_a if gf_a > gf_b else (team_b if gf_b > gf_a else None)

    if knockout and winner is None:
        # Extra time → penalties: Moderate bias toward better ELO
        elo_edge = (ea - eb) / 600
        p_a_pen = max(0.35, min(0.65, PENALTY_SHOOTOUT_PROB_EACH + elo_edge))
        winner = (
            team_a
            if (p_a_pen > 0.5 if deterministic else random.random() < p_a_pen)
            else team_b
        )
        return winner, {
            "lambdas": final_lambdas,
            "outcome": "penalties",
            "winner": winner,
            "gf_a": gf_a,
            "gf_b": gf_b,
        }

    return winner, {
        "lambdas": final_lambdas,
        "outcome": "90min" if winner else "draw",
        "winner": winner,
        "gf_a": gf_a,
        "gf_b": gf_b,
    }


# ─────────────────────────────────────────────
# 9. GROUP STAGE SIMULATION
# ─────────────────────────────────────────────


def run_group_stage(
    model,
    embeddings,
    team_idx,
    current_elos,
    roster_strengths=None,
    deterministic: bool = False,
    prob_matrix: dict = None,
):
    """
    Simulate all 72 group matches. Returns group standings dict.
    Points: W=3, D=1, L=0
    Tiebreakers: points → goal_diff → goals_for → H2H → fair play → lots
    """
    standings = {}  # group → {team: {pts, gd, gf, ga}}
    all_group_results = {}

    for grp, teams in GROUPS.items():
        table = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0} for t in teams}
        h2h = defaultdict(lambda: defaultdict(int))  # h2h[t1][t2] = pts

        for ta, tb in combinations(teams, 2):
            winner, res = simulate_match(
                model,
                embeddings,
                ta,
                tb,
                team_idx,
                current_elos,
                roster_strengths,
                "neutral",
                False,
                deterministic,
                prob_matrix,
            )
            gf_a = res["gf_a"]
            gf_b = res["gf_b"]

            # Update table
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

        # Sort: pts → gd → gf → random (lots)
        ranked = sorted(
            teams,
            key=lambda t: (
                table[t]["pts"],
                table[t]["gd"],
                table[t]["gf"],
                random.random(),
            ),
            reverse=True,
        )
        standings[grp] = {"table": table, "ranked": ranked}

    return standings


def pick_best_third_place(standings: dict) -> list[str]:
    """Select 8 best third-place finishers from 12 groups."""
    thirds = []
    for grp, s in standings.items():
        third_team = s["ranked"][2]
        row = s["table"][third_team]
        thirds.append((grp, third_team, row["pts"], row["gd"], row["gf"]))

    thirds.sort(key=lambda x: (x[2], x[3], x[4], random.random()), reverse=True)
    return [t[1] for t in thirds[:8]]


# ─────────────────────────────────────────────
# 10. KNOCKOUT STAGE
# ─────────────────────────────────────────────


def build_r32_bracket(standings: dict, best_thirds: list[str]) -> list[tuple]:
    """
    R32 bracket seeding:
    - 12 winners, 12 runners-up, 8 best thirds advance (32 teams).
    - We rank all 32 advancing teams 1 to 32 based on group stage performance (Pts, GD, GF).
    - Seed 1 plays Seed 32, Seed 2 plays Seed 31, etc.
    This guarantees Group Winners face weaker advancing teams (Thirds or weak Runners-up).
    """
    advancing_teams = []

    # Gather all 32 advancing teams with their group stage stats
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

    # Sort them by Points -> Goal Difference -> Goals For -> Random
    advancing_teams.sort(
        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], random.random()),
        reverse=True,
    )

    # Pair Seed i vs Seed 31-i
    matchups = []
    for i in range(16):
        matchups.append((advancing_teams[i][0], advancing_teams[31 - i][0]))

    return matchups


def run_knockout_round(
    matchups: list[tuple],
    model,
    embeddings,
    team_idx,
    current_elos,
    round_name: str,
    roster_strengths: dict = None,
    deterministic: bool = False,
    prob_matrix: dict = None,
) -> list[str]:
    """Simulate one knockout round. Returns list of winners."""
    print(f"\n  {'─' * 48}")
    print(f"  {round_name}")
    print(f"  {'─' * 48}")
    winners = []
    for ta, tb in matchups:
        winner, res = simulate_match(
            model,
            embeddings,
            ta,
            tb,
            team_idx,
            current_elos,
            roster_strengths=roster_strengths,
            venue="neutral",
            knockout=True,
            deterministic=deterministic,
            prob_matrix=prob_matrix,
        )
        note = ""
        if res["outcome"] == "penalties":
            note = " (pens)"
        print(f"  {ta:<30} vs {tb:<30} → {winner}{note}")
        winners.append(winner)
    return winners


def pair_winners(winners: list[str]) -> list[tuple]:
    """Cross-bracket pairing to distribute heavyweights evenly"""
    half = len(winners) // 2
    return [(winners[i], winners[i + half]) for i in range(half)]


# ─────────────────────────────────────────────
# 11. FULL TOURNAMENT SIMULATION
# ─────────────────────────────────────────────


def simulate_tournament(
    model,
    graph,
    team_idx,
    current_elos,
    roster_strengths=None,
    n_sims=10000,
    prob_matrix=None,
):
    """
    Monte Carlo simulation of the full 2026 WC.
    Returns win-probability dictionary per team.
    """
    device = next(model.parameters()).device
    graph = graph.to(device)

    win_counts = defaultdict(int)
    final_counts = defaultdict(int)
    sf_counts = defaultdict(int)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph)

    # Precompute probability matrix for fast MC simulation if not provided
    if prob_matrix is None:
        prob_matrix = precompute_match_probabilities(
            model, embeddings, team_idx, current_elos, roster_strengths
        )

    print(f"\n{'=' * 60}")
    print(f"  Running {n_sims} Monte Carlo simulations …")
    print(f"{'=' * 60}")

    for sim in tqdm(range(n_sims), desc="Simulating"):
        # Group stage
        standings = run_group_stage(
            model,
            embeddings,
            team_idx,
            current_elos,
            roster_strengths,
            prob_matrix=prob_matrix,
        )
        best_thirds = pick_best_third_place(standings)

        # R32
        r32_matchups = build_r32_bracket(standings, best_thirds)
        r32_winners = []
        for ta, tb in r32_matchups:
            winner, res = simulate_match(
                model,
                embeddings,
                ta,
                tb,
                team_idx,
                current_elos,
                roster_strengths,
                "neutral",
                True,
                False,
                prob_matrix,
            )
            r32_winners.append(winner)

        # R16 (16 winners from 32 = 8 matches)
        r16_matchups = pair_winners(r32_winners)
        r16_winners = []
        for ta, tb in r16_matchups:
            winner, res = simulate_match(
                model,
                embeddings,
                ta,
                tb,
                team_idx,
                current_elos,
                roster_strengths,
                "neutral",
                True,
                False,
                prob_matrix,
            )
            r16_winners.append(winner)

        # QF (4 matches)
        qf_matchups = pair_winners(r16_winners)
        qf_winners = []
        for ta, tb in qf_matchups:
            winner, res = simulate_match(
                model,
                embeddings,
                ta,
                tb,
                team_idx,
                current_elos,
                roster_strengths,
                "neutral",
                True,
                False,
                prob_matrix,
            )
            qf_winners.append(winner)

        # SF (2 matches)
        sf_matchups = pair_winners(qf_winners)
        sf_losers, sf_winners = [], []
        for ta, tb in sf_matchups:
            winner, res = simulate_match(
                model,
                embeddings,
                ta,
                tb,
                team_idx,
                current_elos,
                roster_strengths,
                "neutral",
                True,
                False,
                prob_matrix,
            )
            loser = ta if winner == tb else tb
            sf_winners.append(winner)
            sf_losers.append(loser)
            sf_counts[ta] += 1
            sf_counts[tb] += 1

        # Final
        fn_a, fn_b = sf_winners
        winner, res = simulate_match(
            model,
            embeddings,
            fn_a,
            fn_b,
            team_idx,
            current_elos,
            roster_strengths,
            "neutral",
            True,
            False,
            prob_matrix,
        )
        final_counts[fn_a] += 1
        final_counts[fn_b] += 1
        win_counts[winner] += 1

    return win_counts, final_counts, sf_counts, n_sims


# ─────────────────────────────────────────────
# 12. SINGLE TOURNAMENT WALKTHROUGH
# ─────────────────────────────────────────────


def run_single_walkthrough(
    model,
    graph,
    team_idx,
    current_elos,
    roster_strengths=None,
    prob_matrix=None,
):
    device = next(model.parameters()).device
    graph = graph.to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph)

    if prob_matrix is None:
        prob_matrix = precompute_match_probabilities(
            model, embeddings, team_idx, current_elos, roster_strengths
        )

    print("\n" + "=" * 60)
    print("  FIFA WORLD CUP 2026 - MAXIMUM LIKELIHOOD BRACKET")
    print("=" * 60)

    # ── Group Stage ────────────────────────────────────
    print("\n📋 GROUP STAGE")
    standings = run_group_stage(
        model,
        embeddings,
        team_idx,
        current_elos,
        roster_strengths=roster_strengths,
        deterministic=True,
        prob_matrix=prob_matrix,
    )

    for grp in sorted(standings):
        s = standings[grp]
        print(f"\n  Group {grp}")
        print(f"  {'Team':<30} {'Pts':>4} {'GD':>4} {'GF':>4} {'GA':>4}")
        print(f"  {'─' * 46}")
        for rank, team in enumerate(s["ranked"], 1):
            t = s["table"][team]
            adv = " ✅" if rank <= 2 else (" 🔘" if rank == 3 else "")
            print(
                f"  {rank}. {team:<28} {t['pts']:>4} {t['gd']:>4} "
                f"{t['gf']:>4} {t['ga']:>4}{adv}"
            )

    best_thirds = pick_best_third_place(standings)
    print(f"\n  🔘 8 Best Third-Place Teams advancing:")
    for i, t in enumerate(best_thirds, 1):
        print(f"     {i}. {t}")

    # ── R32 ───────────────────────────────────────────
    r32_matchups = build_r32_bracket(standings, best_thirds)
    r32_winners = run_knockout_round(
        r32_matchups,
        model,
        embeddings,
        team_idx,
        current_elos,
        "⚡ ROUND OF 32",
        roster_strengths=roster_strengths,
        deterministic=True,
        prob_matrix=prob_matrix,
    )

    # ── R16 ───────────────────────────────────────────
    r16_matchups = pair_winners(r32_winners)
    r16_winners = run_knockout_round(
        r16_matchups,
        model,
        embeddings,
        team_idx,
        current_elos,
        "⚡ ROUND OF 16",
        roster_strengths=roster_strengths,
        deterministic=True,
        prob_matrix=prob_matrix,
    )

    # ── QF ────────────────────────────────────────────
    qf_matchups = pair_winners(r16_winners)
    qf_winners = run_knockout_round(
        qf_matchups,
        model,
        embeddings,
        team_idx,
        current_elos,
        "⚡ QUARTER-FINALS",
        roster_strengths=roster_strengths,
        deterministic=True,
        prob_matrix=prob_matrix,
    )

    # ── SF ────────────────────────────────────────────
    sf_matchups = pair_winners(qf_winners)
    sf_winners, sf_losers = [], []
    print(f"\n  {'─' * 48}")
    print(f"  ⚡ SEMI-FINALS")
    print(f"  {'─' * 48}")
    for ta, tb in sf_matchups:
        winner, res = simulate_match(
            model,
            embeddings,
            ta,
            tb,
            team_idx,
            current_elos,
            roster_strengths=roster_strengths,
            venue="neutral",
            knockout=True,
            deterministic=True,
            prob_matrix=prob_matrix,
        )
        loser = ta if winner == tb else tb
        note = " (pens)" if res["outcome"] == "penalties" else ""
        print(f"  {ta:<30} vs {tb:<30} → {winner}{note}")
        sf_winners.append(winner)
        sf_losers.append(loser)

    # ── 3rd Place ─────────────────────────────────────
    print(f"\n  {'─' * 48}")
    print(f"  🥉 THIRD PLACE PLAY-OFF")
    print(f"  {'─' * 48}")
    t3_winner, res = simulate_match(
        model,
        embeddings,
        sf_losers[0],
        sf_losers[1],
        team_idx,
        current_elos,
        roster_strengths=roster_strengths,
        venue="neutral",
        knockout=True,
        deterministic=True,
        prob_matrix=prob_matrix,
    )
    note = " (pens)" if res["outcome"] == "penalties" else ""
    print(f"  {sf_losers[0]:<30} vs {sf_losers[1]:<30} → {t3_winner}{note}")

    # ── Final ─────────────────────────────────────────
    print(f"\n  {'─' * 48}")
    print(f"  🏆 FINAL")
    print(f"  {'─' * 48}")
    fn_a, fn_b = sf_winners
    champion, res = simulate_match(
        model,
        embeddings,
        fn_a,
        fn_b,
        team_idx,
        current_elos,
        roster_strengths=roster_strengths,
        venue="neutral",
        knockout=True,
        deterministic=True,
        prob_matrix=prob_matrix,
    )
    note = " (pens)" if res["outcome"] == "penalties" else ""
    runner_up = fn_a if champion == fn_b else fn_b
    print(f"  {fn_a:<30} vs {fn_b:<30} → {champion}{note}")
    print(f"\n  🏆 CHAMPION:    {champion}")
    print(f"  🥈 RUNNER-UP:   {runner_up}")
    print(f"  🥉 THIRD PLACE: {t3_winner}")

    return champion


# ─────────────────────────────────────────────
# 13. REPORTING
# ─────────────────────────────────────────────


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

    print(
        f"\n  Teams not appearing in semis: "
        f"{', '.join(t for t in ALL_TEAMS if t not in sf_counts) or 'none'}"
    )


# ─────────────────────────────────────────────
# 14. MAIN
# ─────────────────────────────────────────────


def main(
    csv_path: str = "results.csv",
    epochs: int = 40,
    n_sims: int = 10000,
):

    # ── Load & ELO ────────────────────────────────────
    df = load_and_prepare(csv_path)
    elo_system, df = build_elo_series(df)
    current_elos = elo_system.snapshot()

    print(f"\n📊 Top 10 ELO Ratings (pre-tournament):")
    top = sorted(
        [(t, current_elos[t]) for t in ALL_TEAMS], key=lambda x: x[1], reverse=True
    )[:10]
    for rank, (team, elo) in enumerate(top, 1):
        print(f"  {rank:2}. {team:<30} {elo:7.1f}")

    dataset_map = {
        2022: "/kaggle/input/datasets/ayushdatta15/wc-predict/Fifa 23 Players Data (1).csv",
        2023: "/kaggle/input/datasets/ayushdatta15/wc-predict/Fifa 23 Players Data (1).csv",
        2024: "/kaggle/input/datasets/ayushdatta15/wc-predict/male_fc_24_players.csv",
        2025: "/kaggle/input/datasets/ayushdatta15/wc-predict/male_players.csv",
        2026: "/kaggle/input/datasets/ayushdatta15/wc-predict/ea_fc26_players.csv",
    }

    roster_strengths_by_year = {}
    for y, path in dataset_map.items():
        try:
            roster_strengths_by_year[y] = parse_ea_fc_dataset(path)
        except Exception as e:
            print(f"Could not load dataset for {y} from {path}: {e}")
            roster_strengths_by_year[y] = {}

    roster_strengths_26 = roster_strengths_by_year.get(2026, {})

    # ── Graph & Walk-Forward Training ─────────────────
    print("\n🔗 Initialising GNN model & Walk-Forward Training …")
    model = MatchGNN(node_feat_dim=5, edge_feat_dim=5, hidden=64, heads=4)

    start_year = 2022
    end_year = 2026

    total_epochs = epochs * (end_year - start_year)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)

    for year in range(start_year, end_year):
        window_start = pd.Timestamp(f"{year}-01-01")
        window_end = pd.Timestamp(f"{year + 1}-01-01")

        print(f"\n   {'=' * 40}")
        print(f"   ▶ Walk-forward Window: {year}")
        print(f"   {'=' * 40}")

        graph_df = df[df["date"] < window_start].copy()
        train_df = df[(df["date"] >= window_start) & (df["date"] < window_end)].copy()

        if len(train_df) == 0:
            print("   No training samples for this window. Skipping...")
            continue

        train_elos = elo_system.snapshot_at(window_start)

        print(f"   Building Training Graph (data up to {year - 1}) ...")
        train_graph, train_team_idx = build_graph(
            graph_df, train_elos, ref_date=window_start
        )
        print(
            f"   Train Nodes: {train_graph.num_nodes}  |  Train Edges: {train_graph.num_edges}"
        )

        samples = prepare_training_samples(
            train_df, train_team_idx, ref_date=window_end
        )
        print(f"   Training on {len(samples['weight'])} samples from {year} ...")

        model = train_model(
            model,
            train_graph,
            samples,
            train_team_idx,
            roster_strengths_by_year.get(year, {}),
            epochs=epochs,
            opt=opt,
            scheduler=scheduler,
        )

    # Now build the full Inference Graph for 2026 WC simulation
    print("\n🔗 Building Full Inference Graph (up to 2026) ...")
    graph, team_idx = build_graph(df, current_elos)
    print(
        f"   Inference Nodes: {graph.num_nodes}  |  Inference Edges: {graph.num_edges}"
    )

    # Precompute match probabilities once for all downstream simulations
    model.eval()
    device = next(model.parameters()).device
    graph = graph.to(device)
    with torch.no_grad():
        embeddings = model.encode(graph)
    prob_matrix = precompute_match_probabilities(
        model, embeddings, team_idx, current_elos, roster_strengths_26
    )

    # ── Single Walkthrough ────────────────────────────
    champion = run_single_walkthrough(
        model,
        graph,
        team_idx,
        current_elos,
        roster_strengths_26,
        prob_matrix=prob_matrix,
    )

    # ── Monte Carlo ───────────────────────────────────
    win_counts, final_counts, sf_counts, _ = simulate_tournament(
        model,
        graph,
        team_idx,
        current_elos,
        roster_strengths=roster_strengths_26,
        n_sims=n_sims,
        prob_matrix=prob_matrix,
    )
    print_mc_report(win_counts, final_counts, sf_counts, n_sims)

    return model, graph, team_idx, current_elos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FIFA 2026 WC GNN Predictor")
    parser.add_argument(
        "--csv",
        default="/kaggle/input/private-dataset/results.csv",
        help="Path to Kaggle results.csv",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument(
        "--sims", type=int, default=10000, help="Monte Carlo simulations"
    )

    args = parser.parse_known_args()[0]

    main(csv_path=args.csv, epochs=args.epochs, n_sims=args.sims)
