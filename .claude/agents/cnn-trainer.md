---
name: cnn-trainer
description: Fine-tune CNNs end-to-end on the 500 training images with flips/rot90 augmentation and TTA, 5-fold.
tools: Read, Edit, Write, Bash, Glob, Grep
---
Own `src/train_cnn.py`. Augmentations must preserve the label: flips, 90° rotations, small
brightness/contrast/noise jitter. Avoid resizing that destroys grain-size information (prefer
native 256 or random crops + full-image inference). Standardize the target per fold.
Follow the experiment protocol in CLAUDE.md.
