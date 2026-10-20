"""README figures (docs/images/), drawn from TRAIN images only: no labels, no test images.

  dataset_samples.png   eight train images, one per structure / image-quality type
  het4_example.png      two clean train images with about the same grain count but different het4 (src.het_blocks)
  reader_training.png   a clean train image, its degrade_sim copy (tools/fsim) and a real noisy (G1) train image

The Korean labels need a Hangul font, e.g. NanumGothic (OFL, github.com/google/fonts):
  python tools/readme_figures.py /path/to/NanumGothic-Regular.ttf /path/to/NanumGothic-Bold.ttf
docs/images/leaderboard_final.png is a screenshot of the final leaderboard, not generated here.
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'fsim'))
from src.common import DATA_DIR  # noqa: E402
from src.het_blocks import GRID, grain_table, segment, shading_norm  # noqa: E402

OUT = os.path.join(ROOT, 'docs', 'images')


def load(i):
    return np.array(Image.open(DATA_DIR / 'train' / f'{i}.png').convert('L'))


def save(fig, name, mode):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, facecolor='white')
    plt.close(fig)
    Image.open(path).convert(mode).save(path, optimize=True)


def fig_samples():
    rows = [
        [('TRAIN_000022', '깨끗함 · 미세 결정립'), ('TRAIN_000291', '깨끗함 · 굵은 결정립'),
         ('TRAIN_000446', '어두운 상(2상) 많음'), ('TRAIN_000265', '기공 많음')],
        [('TRAIN_000177', '결정립 신장 · 정렬'), ('TRAIN_000087', '노이즈 중간 · 경계선 보임'),
         ('TRAIN_000206', '흐림 · 낮은 대비'), ('TRAIN_000021', '노이즈 심함 (G1) · 경계 안 보임')],
    ]
    fig, ax = plt.subplots(2, 4, figsize=(12, 7.0))
    for r, row in enumerate(rows):
        for c, (i, t) in enumerate(row):
            a = ax[r, c]
            a.imshow(load(i), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            a.set_title(t, fontsize=12, fontweight='bold', pad=4)
            a.text(0.5, -0.045, i, transform=a.transAxes, ha='center', va='top', fontsize=8.5, color='#555555')
            a.set_xticks([]), a.set_yticks([])
    fig.tight_layout(h_pad=2.8, w_pad=0.6)
    save(fig, 'dataset_samples.png', 'L')


def block_sizes(im):
    """The 16 block values of src.het_blocks.one (het4 = their sd): -2 log mean(a^-1/2) over non-pore pixels."""
    sm, ws = segment(im.astype(np.float32))
    g = grain_table(sm, ws)
    pore = ((shading_norm(g) < 0.68) & (g.area < 600)).values
    is_pore = np.zeros(ws.max() + 1, bool)
    is_pore[g.label.values[pore]] = True
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1)
    m = ~is_pore[ws]
    v = np.full(16, np.nan)
    for b in range(16):
        s = (GRID == b) & m
        if s.sum() >= 50:
            v[b] = -2 * np.log((a[s] ** -0.5).mean())
    return v


def fig_het(pair_ids=('TRAIN_000442', 'TRAIN_000340')):
    hb = pd.read_parquet(DATA_DIR / 'het_blocks_train.parquet').set_index('ID')   # python -m src.het_blocks
    pair = [(i, hb.loc[i, 'het4'], hb.loc[i, 'N_eff']) for i in pair_ids]
    vals = [block_sizes(load(i)) for i, _, _ in pair]
    allv = np.concatenate([v[~np.isnan(v)] for v in vals])
    lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 6.2))
    names = ['균일한 이미지', '불균일한 이미지']
    for k, ((i, h, n), v) in enumerate(zip(pair, vals)):
        a = ax[k]
        a.imshow(load(i), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        a.imshow(np.kron(v.reshape(4, 4), np.ones((64, 64))), cmap='coolwarm', vmin=lo, vmax=hi, alpha=0.38,
                 interpolation='nearest')
        for t in range(1, 4):
            a.axhline(t * 64 - 0.5, color='white', lw=1.2)
            a.axvline(t * 64 - 0.5, color='white', lw=1.2)
        for b in range(16):
            if not np.isnan(v[b]):
                a.text((b % 4) * 64 + 32, (b // 4) * 64 + 32, f'{np.exp(v[b]):.0f}', ha='center', va='center',
                       fontsize=10, color='black', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
        a.set_title(f'{names[k]}: het4 = {h:.2f}  (결정립 약 {n:.0f}개)', fontsize=12.5, fontweight='bold', pad=8)
        a.text(0.5, -0.025, i, transform=a.transAxes, ha='center', va='top', fontsize=8.5, color='#555555')
        a.set_xticks([]), a.set_yticks([])
    fig.text(0.5, 0.02, '숫자 = 64×64px 블록의 대표 결정립 면적(px²)   색 = 블록 사이 크기 차이 (파랑: 작음, 빨강: 큼)',
             ha='center', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'het4_example.png', 'RGB')


def fig_reader(src_id='TRAIN_000177', real_id='TRAIN_000021'):
    import degrade_sim
    clean = load(src_id)
    sim, _ = degrade_sim.degrade(clean, prior='G1hi', seed=3)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.7))
    panels = [(clean, '1) 깨끗한 train 이미지\n(블록별 het, 결정립 수를 재서 정답으로)', src_id),
              (sim, '2) 같은 이미지를 노이즈 이미지처럼 열화\n(판독기 CNN의 학습 입력)', f'{src_id} 열화 사본'),
              (load(real_id), '3) 실제 노이즈 이미지 (G1)\n(학습된 판독기가 het를 읽는 대상)', real_id)]
    for a, (im, t, cap) in zip(ax, panels):
        a.imshow(im, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        a.set_title(t, fontsize=11.5, fontweight='bold')
        a.text(0.5, -0.03, cap, transform=a.transAxes, ha='center', va='top', fontsize=8.5, color='#555555')
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout(w_pad=1.0)
    save(fig, 'reader_training.png', 'L')


if __name__ == '__main__':
    for f in sys.argv[1:]:
        font_manager.fontManager.addfont(f)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(OUT, exist_ok=True)
    fig_samples()
    fig_het()
    fig_reader()
