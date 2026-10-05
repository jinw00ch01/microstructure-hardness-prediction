#!/usr/bin/env bash
# Full pipeline. Expects data/ populated (see CLAUDE.md).
set -e
python -m src.folds
python -m src.features
python -m src.train_gbm --model lgb
python -m src.train_gbm --model cat
python -m src.train_gbm --model svr
for bb in convnext_small.fb_in22k_ft_in1k efficientnet_b3.ra2_in1k; do
  python -m src.extract_embeddings --backbone $bb && python -m src.train_head --emb ${bb}_256 --head svr --with-feats
done
python -m src.train_cnn --backbone convnext_tiny.fb_in22k_ft_in1k --epochs 40 --name cnn_cnxt_t
python -m src.ensemble --out blend_v1
