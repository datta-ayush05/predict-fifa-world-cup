# FIFA World Cup 2026 Prediction Model

<div align="center">
  <a href="https://www.kaggle.com/code/ayushdatta15/fifa-world-cup-prediction-a-gnn-model">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle (Standard GNN)" height="30">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.kaggle.com/code/ayushdatta15/predicting-fifa-world-cup-using-dixon-coles-gnn">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle (Dixon-Coles GNN)" height="30">
  </a>
</div>
<br>

A Graph Neural Network Approach Integrating Time-Decay Weighting, Elo Ratings, and Latent Roster Strengths.

## Abstract

This repository presents a sophisticated prediction framework for the FIFA World Cup 2026, leveraging a **Graph Attention Network (GAT)** to forecast match outcomes and tournament progression. By representing international football as a dynamic temporal graph, the model integrates a holistic feature set comprising historical match records, continuous Elo rating dynamics, and position-weighted roster strengths extracted from comprehensive EA FC player databases. The system outputs expected goals via a Poisson distribution, enabling statistically rigorous Monte Carlo simulations of the entire tournament structure.

## Methodology

The predictive architecture is built upon several core components designed to capture the complex, evolving nature of international football:

1. **Temporal Graph Construction**: Historical national teams constitute the nodes, while past match-ups form the weighted edges. This structure allows the model to learn complex inter-team relationships beyond direct head-to-head records.
2. **Continuous Time-Decay Weighting**: To balance historical precedent with current form, the model applies an exponential decay function to past matches. A calibrated 3-year half-life ensures recent performances exert a dominant influence on the learned embeddings.
3. **Integrated Elo Rating System**: A bespoke rolling Elo engine provides baseline strength assessments. These ratings dynamically update based on match importance (e.g., World Cup fixtures yield higher K-factors than friendlies) and are embedded directly into the node and edge feature spaces.
4. **Latent Roster Strengths**: Player-level data across multiple EA FC datasets (FIFA 22 through FC 26) is chronologically mapped and synthesized. The model calculates position-specific weighted ratings to extract overall top-11 and bench strengths, providing a granular view of a nation's current talent pool.
5. **Poisson Goal Expectancy**: The GAT architecture outputs expected goals ($\lambda$) for both teams via a Softplus activation. The network is optimized using Poisson Negative Log-Likelihood. During simulation, native Poisson sampling drives the generation of match scores and determines knockout phase progressions, including empirically derived penalty shootout probabilities.

## Tournament Structure & Simulation

The model accurately reflects the expanded 48-team FIFA World Cup 2026 format:
- 12 groups of 4 teams.
- Progression for the top 2 teams per group, alongside the 8 best third-place finishers.
- Full knockout bracket progression (Round of 32 through to the Final).

## Results and Findings

The model was rigorously validated using a walk-forward training methodology across sequential temporal windows (2022 to 2025). The finalized inference graph was then utilized to conduct 10,000 Monte Carlo simulations of the 2026 tournament.

### Dixon-Coles vs. Standard GNN: A Case for Simplicity
While the original Dixon-Coles method introduces a dependency parameter ($\rho$) to adjust for low-scoring draws in standard Poisson regression, empirical results from our walk-forward training demonstrate that the **Graph Neural Network (GNN)** natively captures these complexities without needing explicit hard-coded dependence modeling.

By leveraging rich node embeddings that learn highly non-linear, temporal team interactions, the standard GNN inherently accounts for defensive or "cagey" matchups. As shown in the final walk-forward validation losses below, the standard GNN consistently competes with (and occasionally outperforms) the explicit Dixon-Coles implementation:

| Walk-Forward Window | Standard GNN Validation Loss | Dixon-Coles Validation Loss |
| :--- | :---: | :---: |
| **2023** | **1.0480** | 1.0722 |
| **2024** | 1.4709 | **1.4649** |
| **2025** | **1.4050** | 1.4059 |

**Conclusion:** The base GNN architecture is robust enough to learn match dynamics intrinsically. Hard-coding the traditional Dixon-Coles $\rho$ parameter introduces structural complexity without yielding statistically significant improvements in predictive power. Thus, the independent Poisson GNN is highly efficient and serves as the optimal predictive model.

### Monte Carlo Probabilities (n=10,000)
The comprehensive simulation highlights the dominant favorites and their respective probabilities of tournament success:

| Nation | Win % | Final % | Semi-Final % |
| :--- | :---: | :---: | :---: |
| **Spain** | 21.8% | 32.3% | 45.9% |
| **Argentina** | 18.7% | 28.8% | 42.3% |
| **France** | 13.0% | 22.9% | 35.5% |
| **England** | 7.6% | 15.0% | 27.0% |
| **Brazil** | 4.6% | 10.2% | 20.0% |
| **Portugal** | 4.4% | 9.6% | 19.3% |
| **Mexico** | 4.0% | 9.2% | 19.4% |
| **Germany** | 3.2% | 7.5% | 16.8% |
| **Colombia** | 3.2% | 7.6% | 16.0% |
| **Ecuador** | 2.6% | 6.4% | 14.8% |

*(For the complete distribution spanning all 48 participating nations, refer to the generated simulation outputs).*

## Usage

Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

Execute the primary script to initiate the data processing, model training, and Monte Carlo simulation phases:
```bash
python wc_predict_gnn.py
```

## Data Sources
- Historical match results derived from the Kaggle "International Soccer Results from 1872" dataset.
- Player rosters and attributes sourced from EA FC (FIFA) rating databases.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
