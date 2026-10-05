#!/usr/bin/env bash
# Full pipeline. Expects data/ populated (see CLAUDE.md).
set -e
python -m src.folds
python -m src.features
python -m src.train_gbm --model lgb
python -m src.train_gbm --model cat
python -m src.train_gbm --model svr
for bb in tf_efficientnetv2_s.in21k_ft_in1k efficientnet_b3.ra2_in1k swinv2_tiny_window8_256.ms_in1k; do
  python -m src.extract_embeddings --backbone $bb && python -m src.train_head --emb ${bb}_256 --head svr --with-feats
done
python -m src.train_cnn --backbone efficientnet_b0.ra_in1k --epochs 40 --name cnn_effb0
python -m src.ensemble --out blend_v1
