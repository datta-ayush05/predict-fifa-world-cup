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
2026-06-11   | Mexico          | South Africa    | 2-0    | 2.34-0.53    | 1.63-0.69
2026-06-11   | South Korea     | Czech Republic  | 2-1    | 1.25-1.33    | 1.14-1.19
2026-06-12   | Canada          | Bosnia and Herzegovina | 1-1    | 2.05-0.71    | 1.65-0.87
2026-06-12   | United States   | Paraguay        | 4-1    | 1.19-1.38    | 1.14-1.00
2026-06-13   | Qatar           | Switzerland     | 1-1    | 0.43-2.64    | 0.71-2.56
2026-06-13   | Brazil          | Morocco         | 1-1    | 1.62-0.94    | 1.08-0.71
2026-06-13   | Haiti           | Scotland        | 0-1    | 0.65-2.25    | 0.72-2.20
2026-06-13   | Australia       | Turkey          | 2-0    | 0.90-1.78    | 1.16-1.23
2026-06-14   | Sweden          | Tunisia         | 5-1    | 1.42-1.12    | 1.13-0.92
2026-06-14   | Netherlands     | Japan           | 2-2    | 1.61-1.01    | 1.49-1.04
2026-06-14   | Germany         | Curacao         | 7-1    | 2.99-0.35    | 3.97-0.45
2026-06-14   | Ivory Coast     | Ecuador         | 1-0    | 0.63-2.10    | 0.60-0.98
2026-06-15   | Belgium         | Egypt           | 1-1    | 1.76-0.85    | 1.45-0.71
2026-06-15   | Iran            | New Zealand     | 2-2    | 1.67-1.00    | 1.42-0.55
2026-06-15   | Spain           | Cape Verde      | 0-0    | 2.92-0.32    | 2.79-0.42
2026-06-15   | Saudi Arabia    | Uruguay         | 1-1    | 0.55-2.34    | 0.46-1.53
2026-06-16   | France          | Senegal         | 3-1    | 2.04-0.67    | 1.47-0.73
2026-06-16   | Iraq            | Norway          | 1-4    | 0.63-2.20    | 0.64-1.86
2026-06-16   | Argentina       | Algeria         | 3-0    | 2.27-0.55    | 1.87-0.65
2026-06-16   | Austria         | Jordan          | 3-1    | 1.98-0.75    | 1.81-0.77
2026-06-17   | England         | Croatia         | 4-2    | 1.56-1.02    | 1.36-0.80
2026-06-17   | Ghana           | Panama          | 1-0    | 0.90-1.75    | 1.21-0.99
2026-06-17   | Uzbekistan      | Colombia        | 1-3    | 0.55-2.35    | 0.56-1.59
2026-06-17   | Portugal        | DR Congo        | 1-1    | 2.25-0.56    | 1.86-0.53
2026-06-18   | Czech Republic  | South Africa    | 1-1    | 1.75-0.86    | 1.34-0.83
2026-06-18   | Mexico          | South Korea     | 1-0    | 1.88-0.82    | 1.45-0.95
2026-06-18   | Switzerland     | Bosnia and Herzegovina | 4-1    | 2.15-0.63    | 1.94-0.73
2026-06-18   | Canada          | Qatar           | 6-0    | 2.55-0.48    | 2.18-0.85
2026-06-19   | Turkey          | Paraguay        | 0-1    | 1.26-1.30    | 1.10-1.01
2026-06-19   | Scotland        | Morocco         | 0-1    | 1.06-1.49    | 0.62-1.23
2026-06-19   | Brazil          | Haiti           | 3-0    | 2.88-0.38    | 3.83-0.42
2026-06-19   | United States   | Australia       | 2-0    | 1.69-0.98    | 1.26-1.15
2026-06-20   | Germany         | Ivory Coast     | 2-1    | 1.94-0.73    | 1.72-0.80
2026-06-20   | Ecuador         | Curacao         | 0-0    | 3.13-0.31    | 2.26-0.34
2026-06-20   | Netherlands     | Sweden          | 5-1    | 1.90-0.78    | 1.87-1.04
2026-06-20   | Tunisia         | Japan           | 0-4    | 0.90-1.73    | 0.73-1.13
2026-06-21   | Spain           | Saudi Arabia    | 4-0    | 2.98-0.32    | 2.60-0.41
2026-06-21   | Uruguay         | Cape Verde      | 2-2    | 2.29-0.54    | 1.64-0.47
2026-06-21   | Belgium         | Iran            | 0-0    | 1.71-0.93    | 1.60-0.85
2026-06-21   | New Zealand     | Egypt           | 1-3    | 1.04-1.57    | 0.50-1.18
2026-06-22   | Argentina       | Austria         | 2-0    | 2.03-0.69    | 1.60-0.65
2026-06-22   | Jordan          | Algeria         | 1-2    | 0.90-1.72    | 0.90-1.78
2026-06-22   | France          | Iraq            | 3-0    | 2.74-0.39    | 2.22-0.44
2026-06-22   | Norway          | Senegal         | 3-2    | 1.50-1.05    | 1.23-1.06
2026-06-23   | Colombia        | DR Congo        | 1-0    | 2.33-0.52    | 1.63-0.54
2026-06-23   | England         | Ghana           | 0-0    | 2.67-0.39    | 2.13-0.46
2026-06-23   | Portugal        | Uzbekistan      | 5-0    | 2.27-0.59    | 1.81-0.55
2026-06-23   | Panama          | Croatia         | 0-1    | 0.80-1.92    | 0.63-2.11
-----------------------------------------------------------------------------------
Total Negative Log-Likelihood (Lower is better):
GNN Scoreline NLL:            140.36
Dixon-Coles Scoreline NLL:    141.48
GNN 3-way NLL:                45.29
Dixon-Coles 3-way NLL:        42.01
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
