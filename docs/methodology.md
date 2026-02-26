# Methodology: Hybrid Expected Threat

## Objective
Estimate attacking value of on-ball progression actions (passes and carries) by combining:
- a zone-based Markov-chain xT surface
- a supervised XGBoost shot-likelihood model with context features

## Data
- Source: StatsBomb open data via `statsbombpy`
- Scope: configured competition and season (`ml/config.py`)
- Event subsets:
  - Successful passes (exclude non-null `pass_outcome`)
  - Carries with valid end location
  - Shots with goal outcome label

## Feature Engineering
Per action features include:
- Start/end zone index on a 16x12 grid (192 zones)
- Start/end distance to goal center
- Start/end directional angle to goal center
- Start/end goal-mouth angle (between lines to both posts)
- Under-pressure binary flag
- Freeze-frame pressure score (sum of inverse opponent distances within 5m)
- Progressive flag (end distance <= 2/3 of start distance)
- Action type code (pass/carry)
- Action length
- Reconstructed game-state label (winning/drawing/losing)
- Game-phase bin (early/mid/late/extra)

## Markov xT
For each zone:
- `P(shot | zone)` from zone action outcomes
- `P(move | zone)` from move attempts
- transition matrix `P(next_zone | zone, move)`
- `P(goal | shot, zone)` from shots taken in zone

Iterative update:
`xT(z) = P(shot|z)*P(goal|z) + P(move|z) * Σ_z' P(z'|z,move) * xT(z')`

Run for 50 iterations.

## XGBoost Model
Target:
- `1` if same team takes a shot within next 5 actions in same possession
- `0` otherwise

Train/validation split:
- 80/20 stratified
- eval metric: validation AUC

Inference:
- Predict shot probability for end-state feature vector
- Create start-state proxy by replacing end spatial fields with start fields
- Use delta `P_shot_end - P_shot_start`

## Hybrid Action xT
Per action:
- Zone delta = `xT_surface[end_zone] - xT_surface[start_zone]`
- ML delta = `P_shot_end - P_shot_start`
- Final xT = `0.5 * zone_delta + 0.5 * ml_delta`

## Aggregation and Validation
Player metrics:
- total xT
- xT per 90
- action volume
- progressive action count
- under-pressure action rate
- goals/xG per 90

Validation:
- Pearson and Spearman correlations between `xT_per_90` and `goals_per_90`
- Pearson and Spearman correlations between `xT_per_90` and `xG_per_90`

## Limitations
- Credits only on-ball actions; off-ball movement is not captured.
- Freeze-frame pressure data is incomplete in some competitions/events.
- Team side/orientation handling is simplified in open event streams.
- Correlation with outcomes does not imply causal attacking impact.
