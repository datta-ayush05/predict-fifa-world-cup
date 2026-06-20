import argparse
import os

import dixon_coles_wc_predict as dc
import numpy as np
import pandas as pd
import torch
import wc_predict_gnn as gnn

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "results.csv")


def run_evaluation(csv_path):
    print("Loading data...")
    df = dc.load_and_prepare(csv_path)

    cutoff = pd.Timestamp("2018-06-01")
    end_tournament = pd.Timestamp("2018-08-01")

    # Train entirely on matches prior to June 2018
    train_df = df[df["date"] < cutoff].copy()

    # Evaluate strictly on 2018 WC matches
    test_df = df[
        (df["date"] >= cutoff)
        & (df["date"] < end_tournament)
        & (df["tournament"].str.contains("FIFA World Cup", case=False, na=False))
    ].copy()

    if len(test_df) == 0:
        print(
            "No 2018 World Cup matches found in the dataset! Make sure results.csv has the matches."
        )
        return

    print(f"Found {len(test_df)} 2018 World Cup matches to test on.")

    # --- 1. Train Dixon Coles ---
    print("\n" + "=" * 60)
    print("TRAINING DIXON-COLES (Knowledge Cutoff: 2018-06-01)")
    print("=" * 60)
    all_teams = sorted(
        set(train_df["home_team"])
        .union(set(train_df["away_team"]))
        .union(set(test_df["home_team"]))
        .union(set(test_df["away_team"]))
    )
    dc_team_idx = {t: i for i, t in enumerate(all_teams)}

    dc_model = dc.train_dixon_coles(
        train_df, len(all_teams), dc_team_idx, epochs=1500, print_freq=1000
    )
    dc_prob_matrix = dc.precompute_match_probabilities(dc_model, dc_team_idx)

    start_year = 2014
    end_year = 2018

    epochs_per_window = 50
    total_epochs = epochs_per_window * (end_year - start_year)

    # --- 2. Train GNN ---
    print("\n" + "=" * 60)
    print("TRAINING GNN (Knowledge Cutoff: 2018-06-01)")
    print("=" * 60)

    elo_system, df_gnn = gnn.build_elo_series(train_df)
    current_elos = elo_system.snapshot()

    gnn_model = gnn.MatchGNN(node_feat_dim=5, edge_feat_dim=5, hidden=64, heads=4)
    opt = torch.optim.Adam(gnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)

    for year in range(start_year, end_year):
        window_start = pd.Timestamp(f"{year}-01-01")
        window_end = pd.Timestamp(f"{year + 1}-01-01")

        print(f"\n   {'=' * 40}")
        print(f"   ▶ Walk-forward Window: {year}")
        print(f"   {'=' * 40}")

        graph_df = df_gnn[df_gnn["date"] < window_start].copy()
        train_window_df = df_gnn[
            (df_gnn["date"] >= window_start) & (df_gnn["date"] < window_end)
        ].copy()

        if len(train_window_df) == 0:
            continue

        train_elos = elo_system.snapshot_at(window_start)
        train_graph, train_team_idx = gnn.build_graph(
            graph_df, train_elos, ref_date=window_start
        )
        samples = gnn.prepare_training_samples(
            train_window_df, train_team_idx, ref_date=window_end
        )

        gnn_model = gnn.train_model(
            gnn_model,
            train_graph,
            samples,
            train_team_idx,
            epochs=epochs_per_window,
            opt=opt,
            scheduler=scheduler,
        )

    print("Building GNN Inference Graph...")
    gnn_graph, gnn_team_idx = gnn.build_graph(df_gnn, current_elos)
    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    gnn_graph = gnn_graph.to(device)
    with torch.no_grad():
        gnn_embeddings = gnn_model.encode(gnn_graph)
    gnn_prob_matrix = gnn.precompute_match_probabilities(
        gnn_model,
        gnn_embeddings,
        gnn_team_idx,
        current_elos,
    )

    # --- 3. Evaluate on 2018 WC Matches ---
    print("\n" + "=" * 80)
    print("EVALUATION ON 2018 WORLD CUP MATCHES")
    print("=" * 80)

    print(
        f"{'Date':<12} | {'Home Team':<15} | {'Away Team':<15} | {'Actual':<6} | {'GNN xG':<12} | {'DC xG':<12}"
    )
    print("-" * 83)

    gnn_total_nll = 0.0
    dc_total_nll = 0.0
    gnn_total_3way_nll = 0.0
    dc_total_3way_nll = 0.0
    dc_rho = dc_model.rho.item()

    for _, row in test_df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        h = row["home_team"]
        a = row["away_team"]
        hs = int(row["home_score"])
        aws = int(row["away_score"])

        dc_lambdas = dc_prob_matrix.get((h, a), np.array([1.0, 1.0]))
        gnn_lambdas = gnn_prob_matrix.get((h, a), np.array([1.0, 1.0]))

        import math

        def calc_nll(lambdas, act_h, act_a):
            lh, la = lambdas[0], lambdas[1]
            ln_fact_h = math.lgamma(act_h + 1)
            ln_fact_a = math.lgamma(act_a + 1)
            nll_h = lh - act_h * np.log(lh + 1e-8) + ln_fact_h
            nll_a = la - act_a * np.log(la + 1e-8) + ln_fact_a
            return nll_h + nll_a

        def calc_3way_probs(lambdas, rho=0.0):
            l1, l2 = lambdas[0], lambdas[1]
            pw, pd, pl = 0.0, 0.0, 0.0

            def poiss(k, l):
                return (l**k * np.exp(-l)) / math.factorial(k)

            for x in range(15):
                for y in range(15):
                    px = poiss(x, l1)
                    py = poiss(y, l2)
                    tau = 1.0
                    if rho != 0.0:
                        if x == 0 and y == 0:
                            tau = max(1e-8, 1 - l1 * l2 * rho)
                        elif x == 0 and y == 1:
                            tau = max(1e-8, 1 + l1 * rho)
                        elif x == 1 and y == 0:
                            tau = max(1e-8, 1 + l2 * rho)
                        elif x == 1 and y == 1:
                            tau = max(1e-8, 1 - rho)
                    p = px * py * tau
                    if x > y:
                        pw += p
                    elif x == y:
                        pd += p
                    else:
                        pl += p
            tot = pw + pd + pl
            return pw / tot, pd / tot, pl / tot

        gnn_nll = calc_nll(gnn_lambdas, hs, aws)
        dc_nll = calc_nll(dc_lambdas, hs, aws)

        gnn_total_nll += gnn_nll
        dc_total_nll += dc_nll

        act_outcome = 0 if hs > aws else (1 if hs == aws else 2)
        gnn_3way_p = calc_3way_probs(gnn_lambdas, 0.0)[act_outcome]
        dc_3way_p = calc_3way_probs(dc_lambdas, dc_rho)[act_outcome]

        gnn_total_3way_nll += -np.log(gnn_3way_p + 1e-8)
        dc_total_3way_nll += -np.log(dc_3way_p + 1e-8)

        gnn_xg_str = f"{gnn_lambdas[0]:.2f}-{gnn_lambdas[1]:.2f}"
        dc_xg_str = f"{dc_lambdas[0]:.2f}-{dc_lambdas[1]:.2f}"

        print(
            f"{date:<12} | {h:<15} | {a:<15} | {hs}-{aws:<4} | {gnn_xg_str:<12} | {dc_xg_str:<12}"
        )

    print("-" * 83)
    print("Total Negative Log-Likelihood (Lower is better):")
    print(f"GNN Scoreline NLL:            {gnn_total_nll:.2f}")
    print(f"Dixon-Coles Scoreline NLL:    {dc_total_nll:.2f}")
    print(f"GNN 3-way NLL:                {gnn_total_3way_nll:.2f}")
    print(f"Dixon-Coles 3-way NLL:        {dc_total_3way_nll:.2f}")

    scores = {
        "Dixon-Coles": dc_total_nll,
        "GNN": gnn_total_nll,
    }
    best_model = min(scores, key=scores.get)
    print(
        f"\n=> {best_model} was more accurate at predicting the exact 2018 scorelines."
    )

    scores_3way = {
        "Dixon-Coles": dc_total_3way_nll,
        "GNN": gnn_total_3way_nll,
    }
    best_model_3way = min(scores_3way, key=scores_3way.get)
    print(f"=> {best_model_3way} was more accurate at 3-way match classification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results.csv",
    )
    args = parser.parse_args()
    run_evaluation(args.csv)
