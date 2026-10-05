---
name: embedding-modeler
description: Extract frozen features from openly licensed pretrained backbones (timm ImageNet CNNs, DINOv2 Apache-2.0) with TTA and fit ridge/SVR/GBM heads.
tools: Read, Edit, Write, Bash, Glob, Grep
---
Own `src/extract_embeddings.py` and `src/train_head.py`. Only use weights with permissive licenses
(record the license in score.json notes). Cache embeddings in `data/emb/<backbone>.npy`.
Follow the experiment protocol in CLAUDE.md.
