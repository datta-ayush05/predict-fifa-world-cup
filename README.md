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

The model was rigorously validated against Dixon Coles for the first 48 matches using a walk-forward training methodology across sequential temporal windows (2022 to 2025). The finalized inference graph was then utilized to conduct 10,000 Monte Carlo simulations of the 2026 tournament.

```bash
================================================================================
EVALUATION ON LIVE 2026 WORLD CUP MATCHES
================================================================================
Date         | Home Team       | Away Team       | Actual | GNN xG       | DC xG
-----------------------------------------------------------------------------------
2026-06-11   | Mexico          | South Africa    | 2-0    | 2.45-0.49    | 1.60-0.68
2026-06-11   | South Korea     | Czech Republic  | 2-1    | 1.34-1.25    | 1.13-1.12
2026-06-12   | Canada          | Bosnia and Herzegovina | 1-1    | 2.02-0.74    | 1.67-0.88
2026-06-12   | United States   | Paraguay        | 4-1    | 1.05-1.53    | 1.08-0.99
2026-06-13   | Qatar           | Switzerland     | 1-1    | 0.48-2.53    | 0.68-2.63
2026-06-13   | Brazil          | Morocco         | 1-1    | 1.57-0.98    | 1.06-0.70
2026-06-13   | Haiti           | Scotland        | 0-1    | 0.69-2.16    | 0.70-2.27
2026-06-13   | Australia       | Turkey          | 2-0    | 0.91-1.77    | 1.15-1.22
2026-06-14   | Sweden          | Tunisia         | 5-1    | 1.39-1.16    | 1.07-0.92
2026-06-14   | Netherlands     | Japan           | 2-2    | 1.57-1.04    | 1.40-1.02
2026-06-14   | Germany         | Curacao         | 7-1    | 2.96-0.36    | 3.76-0.45
2026-06-14   | Ivory Coast     | Ecuador         | 1-0    | 0.57-2.22    | 0.56-0.92
2026-06-15   | Belgium         | Egypt           | 1-1    | 1.68-0.92    | 1.46-0.69
2026-06-15   | Iran            | New Zealand     | 2-2    | 1.59-1.06    | 1.40-0.53
2026-06-15   | Spain           | Cape Verde      | 0-0    | 2.81-0.35    | 2.79-0.41
2026-06-15   | Saudi Arabia    | Uruguay         | 1-1    | 0.56-2.32    | 0.46-1.51
2026-06-16   | France          | Senegal         | 3-1    | 1.98-0.70    | 1.46-0.72
2026-06-16   | Iraq            | Norway          | 1-4    | 0.73-2.03    | 0.63-1.85
2026-06-16   | Argentina       | Algeria         | 3-0    | 2.27-0.55    | 1.92-0.61
2026-06-16   | Austria         | Jordan          | 3-1    | 1.92-0.80    | 1.72-0.77
2026-06-17   | England         | Croatia         | 4-2    | 1.52-1.06    | 1.33-0.75
2026-06-17   | Ghana           | Panama          | 1-0    | 0.86-1.81    | 1.23-0.94
2026-06-17   | Uzbekistan      | Colombia        | 1-3    | 0.57-2.32    | 0.56-1.55
2026-06-17   | Portugal        | DR Congo        | 1-1    | 2.13-0.62    | 1.85-0.55
2026-06-18   | Czech Republic  | South Africa    | 1-1    | 1.70-0.90    | 1.32-0.83
2026-06-18   | Mexico          | South Korea     | 1-0    | 1.94-0.78    | 1.36-0.93
2026-06-18   | Switzerland     | Bosnia and Herzegovina | 4-1    | 2.05-0.70    | 1.92-0.72
2026-06-18   | Canada          | Qatar           | 6-0    | 2.49-0.51    | 2.29-0.84
2026-06-19   | Turkey          | Paraguay        | 0-1    | 1.15-1.41    | 1.07-1.00
2026-06-19   | Scotland        | Morocco         | 0-1    | 1.08-1.47    | 0.63-1.21
2026-06-19   | Brazil          | Haiti           | 3-0    | 2.72-0.43    | 3.80-0.41
2026-06-19   | United States   | Australia       | 2-0    | 1.65-1.01    | 1.23-1.14
2026-06-20   | Germany         | Ivory Coast     | 2-1    | 1.91-0.75    | 1.62-0.77
2026-06-20   | Ecuador         | Curacao         | 0-0    | 3.22-0.29    | 2.13-0.33
2026-06-20   | Netherlands     | Sweden          | 5-1    | 1.87-0.81    | 1.87-1.01
2026-06-20   | Tunisia         | Japan           | 0-4    | 0.93-1.69    | 0.69-1.07
2026-06-21   | Spain           | Saudi Arabia    | 4-0    | 2.89-0.34    | 2.52-0.40
2026-06-21   | Uruguay         | Cape Verde      | 2-2    | 2.25-0.56    | 1.67-0.47
2026-06-21   | Belgium         | Iran            | 0-0    | 1.68-0.95    | 1.58-0.84
2026-06-21   | New Zealand     | Egypt           | 1-3    | 1.06-1.54    | 0.49-1.15
2026-06-22   | Argentina       | Austria         | 2-0    | 2.11-0.65    | 1.60-0.62
2026-06-22   | Jordan          | Algeria         | 1-2    | 0.91-1.73    | 0.93-1.71
2026-06-22   | France          | Iraq            | 3-0    | 2.60-0.44    | 2.14-0.43
2026-06-22   | Norway          | Senegal         | 3-2    | 1.43-1.12    | 1.26-1.05
2026-06-23   | Colombia        | DR Congo        | 1-0    | 2.25-0.56    | 1.60-0.56
2026-06-23   | England         | Ghana           | 0-0    | 2.55-0.44    | 2.08-0.45
2026-06-23   | Portugal        | Uzbekistan      | 5-0    | 2.20-0.63    | 1.80-0.55
2026-06-23   | Panama          | Croatia         | 0-1    | 0.92-1.76    | 0.60-2.06
-----------------------------------------------------------------------------------
Total Negative Log-Likelihood (Lower is better):
GNN Scoreline NLL:            141.13
Dixon-Coles Scoreline NLL:    142.02
GNN 3-way NLL:                45.68
Dixon-Coles 3-way NLL:        42.07
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
