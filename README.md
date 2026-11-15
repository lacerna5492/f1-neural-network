# 🏁 F1 Points-Finish Predictor

**CAI2300C Introduction to NLP · Dr. Lee · In-class assignment: build your own neural network with your OWN dataset**
Ryan Lacerna

A binary neural network (TensorFlow/Keras) that predicts whether a Formula 1 driver will **score championship points** in a race,
using only information known before the race starts. Built on Dr. Lee's *Neural Network Cookbook · 02 Binary classifier* recipe.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lacerna5492/f1-neural-network/blob/main/RLacerna_F1_Neural_Network.ipynb)

## Files

| File | What it is |
|---|---|
| `RLacerna_F1_Neural_Network.ipynb` | The notebook: load → clean → split → encode → build → train → score, plus my notes, a common-sense rule check and an architecture comparison |
| `f1_points_prerace.csv` | The dataset: 25,939 rows (one per driver per race, 1950–2026), 15 pre-race inputs, target `points_finish` |
| `build_table.py` | How I built the CSV from 8 raw F1 tables (results, races, drivers, constructors, circuits, qualifying, driver and constructor standings) |
| `template/02-binary-classifier-original.ipynb` | Dr. Lee's original recipe, unchanged, for reference |

## Results (test races the model never saw)

| Approach | Accuracy |
|---|---|
| Always guess "no points" | ~67.5% |
| Rule: starts in the top 10 | ~72.5% |
| Rule: averaged top-10 finishes in last 5 races | ~71.5% |
| **Neural network (64 → 32, dropout 0.2)** | **~78%** (AUC ~0.84) |

Data leakage was avoided on purpose: finishing position, points, laps, status, race time and fastest lap are **not** inputs.
Recent-form columns only use previous races.
