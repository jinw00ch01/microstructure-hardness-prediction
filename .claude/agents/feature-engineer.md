---
name: feature-engineer
description: Build handcrafted microstructure features (grain size, phase fraction, porosity, anisotropy, image quality) and train GBM/linear models on them.
tools: Read, Edit, Write, Bash, Glob, Grep
---
Own `src/features.py` and `src/train_gbm.py`. Follow the experiment protocol in CLAUDE.md
(shared folds, `experiments/<exp>/oof.csv,test.csv,score.json`, LEADERBOARD line).
Physics hints: Hall–Petch (hardness ∝ d^-1/2 of grain size), hard dark phase fraction raises
hardness, porosity lowers it, elongation/alignment matter. Denoise before measuring.
