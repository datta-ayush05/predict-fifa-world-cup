# FIFA World Cup 2026 Prediction Model

A Graph Neural Network Approach Integrating Time-Decay Weighting, Elo Ratings, and Latent Roster Strengths.

## Abstract

This repository presents a sophisticated prediction framework for the FIFA World Cup 2026, leveraging a **Graph Attention Network (GAT)** to forecast match outcomes and tournament progression. By representing international football as a dynamic temporal graph, the model integrates a holistic feature set comprising historical match records and continuous Elo rating dynamics The system outputs expected goals via a Poisson distribution, enabling statistically rigorous Monte Carlo simulations of the entire tournament structure.

## Methodology

The predictive architecture is built upon several core components designed to capture the complex, evolving nature of international football:

1. **Temporal Graph Construction**: Historical national teams constitute the nodes, while past match-ups form the weighted edges. This structure allows the model to learn complex inter-team relationships beyond direct head-to-head records.
2. **Continuous Time-Decay Weighting**: To balance historical precedent with current form, the model applies an exponential decay function to past matches. A calibrated 3-year half-life ensures recent performances exert a dominant influence on the learned embeddings.
3. **Integrated Elo Rating System**: A bespoke rolling Elo engine provides baseline strength assessments. These ratings dynamically update based on match importance (e.g., World Cup fixtures yield higher K-factors than friendlies) and are embedded directly into the node and edge feature spaces.
4. **Poisson Goal Expectancy**: The GAT architecture outputs expected goals ($\lambda$) for both teams via a Softplus activation. The network is optimized using Poisson Negative Log-Likelihood. During simulation, native Poisson sampling drives the generation of match scores and determines knockout phase progressions, including empirically derived penalty shootout probabilities.

## Tournament Structure & Simulation

The model accurately reflects the expanded 48-team FIFA World Cup 2026 format:
- 12 groups of 4 teams.
- Progression for the top 2 teams per group, alongside the 8 best third-place finishers.
- Full knockout bracket progression (Round of 32 through to the Final).

## Results and Findings

The model was rigorously validated against Dixon Coles for the first 28 matches using a walk-forward training methodology across sequential temporal windows (2022 to 2025). The finalized inference graph was then utilized to conduct 10,000 Monte Carlo simulations of the 2026 tournament.

```bash
================================================================================
EVALUATION ON LIVE 2026 WORLD CUP MATCHES
================================================================================
Date         | Home Team       | Away Team       | Actual | GNN xG       | DC xG
-----------------------------------------------------------------------------------
2026-06-11   | Mexico          | South Africa    | 2-0    | 2.41-0.51    | 1.60-0.68
2026-06-11   | South Korea     | Czech Republic  | 2-1    | 1.32-1.29    | 1.13-1.12
2026-06-12   | Canada          | Bosnia and Herzegovina | 1-1    | 1.98-0.77    | 1.67-0.88
2026-06-12   | United States   | Paraguay        | 4-1    | 1.07-1.54    | 1.08-0.99
2026-06-13   | Qatar           | Switzerland     | 1-1    | 0.47-2.55    | 0.68-2.63
2026-06-13   | Brazil          | Morocco         | 1-1    | 1.55-1.02    | 1.06-0.70
2026-06-13   | Haiti           | Scotland        | 0-1    | 0.69-2.19    | 0.70-2.27
2026-06-13   | Australia       | Turkey          | 2-0    | 0.90-1.80    | 1.15-1.22
2026-06-14   | Sweden          | Tunisia         | 5-1    | 1.40-1.16    | 1.07-0.92
2026-06-14   | Germany         | Curacao         | 7-1    | 3.00-0.36    | 3.76-0.45
2026-06-14   | Ivory Coast     | Ecuador         | 1-0    | 0.61-2.16    | 0.56-0.92
2026-06-14   | Netherlands     | Japan           | 2-2    | 1.61-1.03    | 1.40-1.02
2026-06-15   | Spain           | Cape Verde      | 0-0    | 2.77-0.36    | 2.79-0.41
2026-06-15   | Saudi Arabia    | Uruguay         | 1-1    | 0.57-2.31    | 0.46-1.51
2026-06-15   | Belgium         | Egypt           | 1-1    | 1.69-0.92    | 1.46-0.69
2026-06-15   | Iran            | New Zealand     | 2-2    | 1.60-1.07    | 1.40-0.53
2026-06-16   | Argentina       | Algeria         | 3-0    | 2.24-0.58    | 1.92-0.61
2026-06-16   | Austria         | Jordan          | 3-1    | 1.95-0.79    | 1.72-0.77
2026-06-16   | France          | Senegal         | 3-1    | 1.98-0.71    | 1.46-0.72
2026-06-16   | Iraq            | Norway          | 1-4    | 0.72-2.06    | 0.63-1.85
2026-06-17   | England         | Croatia         | 4-2    | 1.51-1.08    | 1.33-0.75
2026-06-17   | Ghana           | Panama          | 1-0    | 0.89-1.78    | 1.23-0.94
2026-06-17   | Portugal        | DR Congo        | 1-1    | 2.09-0.65    | 1.85-0.55
2026-06-17   | Uzbekistan      | Colombia        | 1-3    | 0.59-2.32    | 0.56-1.55
2026-06-18   | Mexico          | South Korea     | 1-0    | 1.94-0.80    | 1.36-0.93
2026-06-18   | Switzerland     | Bosnia and Herzegovina | 4-1    | 2.04-0.71    | 1.92-0.72
2026-06-18   | Czech Republic  | South Africa    | 1-1    | 1.71-0.90    | 1.32-0.83
2026-06-18   | Canada          | Qatar           | 6-0    | 2.49-0.52    | 2.29-0.84
-----------------------------------------------------------------------------------
Total Negative Log-Likelihood (Lower is better):
GNN Score:            40.50
Dixon-Coles Score:    40.99
```


## Usage

Execute the scripts to install the libraries, initiate the data processing, model training, and Monte Carlo simulation phases:
```bash
uv run wc_predict_gnn.py

uv run dixon_coles_wc_predict.py
```

## Data Sources
- Historical match results derived from the Kaggle "International Soccer Results from 1872" dataset.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
