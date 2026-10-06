# Resume guide (updated 2026-10-07 ~04:40 KST)

Goal: public LB RMSE <= 10.

## Morning of 2026-10-07 (read this first)
- **blend_v12 scored public 11.9094** (user, 04:25 KST), worse than v10 (11.8267) and about v7 (11.8795), although its
  nested CV was 0.26 better. Not a bug: own columns are applied to the 333 high-noise test images exactly as to train
  (feat7 - feat6 differences have sd 5.2 on both). On OOF the v12 - v10 direction correlates -0.21 with v10's errors; on
  public about -0.03. The OOF gain was carried by a few images: the 10 most improved images give 73% of it, the
  10%-trimmed mean gain is under half, the median about zero. Lesson: before handing over a file, check that the gain
  is not concentrated (top-10 share, trimmed mean, median), not only folds and bootstrap.
- Next submission: blend_v11 (CNN share 0.131, no own-slope member), testing whether a larger CNN share helps public.
- Next GPU candidate: a CNN with a restored-image input channel (cnn-trainer prepares the code and laptop steps).

## Night of 2026-10-07
- **blend_v12** (nested CV 12.162 vs v11 12.407, 5/5 folds, paired bootstrap 90% of the difference [-0.41, -0.09]) is the
  next submission (delivered ~02:15 KST). New member `feat7_rawn3own_ridge_all` (weight 0.59): the feat6 restored ridge
  plus "own columns", raw v3+cal copied as n3_* only for images with raw ic_noise > 11.916 (train top-tercile cut; NaN
  elsewhere, median-imputed in fold), so the noisiest third gets its own slopes (`python -m src.features --n3own`).
  Member CV 12.53 -> 12.26; random 500-image "own" groups never help, so the gain is specific to the noise split.
  Gain is mostly in the high-noise tercile (blend 14.85 -> 14.37).
- Null that night: the high-noise specialist restorer (`src.restore --tag hn`, `data_restored_hn/`) is better on
  synthetic pairs but its features rank hardness worse on real noisy images; effnetv2-s 6 seeds and ConvNeXt-tiny add
  nothing to the blend (laptop-gpu.md section 7).
- Follow-ups that night (all fold-paired, then checked in the two-stage blend):
  - N3 own columns also help the other ridge members (feat7_rawn3own_ridge_splmean 12.365, _milspl 12.323, _add
    12.340), but next to feat7_rawn3own_ridge_all they add nothing to the blend (errors correlate 0.97-0.99).
  - Null: own slopes for the mid-noise tercile (+0.44), smooth noise interactions x*t (+0.16 / +0.27), a bigger own
    block (v4, v5), own block from an NLM view or either restorer (raw calibrated measures rank hardness best in the
    noisiest third), own slopes in the cs24 embedding ridge (1088 own columns on ~133 images per fold overfit).
  - Ties: ic_noise in the ridge's variance model (feat7_rawn3own_ridge_all_hetnz, blend 12.159) and a noise-weighted
    embedding fit (emb_..._cs24_hetnz): **blend_v13**, nested 12.136 (4/5 folds vs v12, 90% [-0.051, -0.003]) but its
    test predictions differ from v12 by sd 0.19, so it is a public-LB tie with v12. Keep it as a later option.
- Next ideas: wait for v12's public score first (does the own-slope gain carry to test?). The noisiest third is still
  14.37 vs 11.30 / 10.46. Untried: a dedicated model for the noisiest third that borrows strength from the rest
  (e.g. ridge on the shared blend prediction + own block), and noise-dependent CNN shares.
- Two-stage blend script: hardness-cache `scripts/twostage.py` (`--add` extra pool members, `--stage2` CNNs,
  `--save-nested`, `--out`). Backups of that night: `experiments-test-extra/`, `data/data_restored_hn.tar`,
  `data/restore_cache/hn/`, the new `data/features_*` files, `submissions/blend_v12-13`.

## Late 2026-10-06
- **blend_v10 scored public 11.8267** (user, 23:33 KST), the new best: 0.053 below v7 although the nested CV tied
  (12.4242 vs 12.4254). The CNN direction helps on test more than OOF shows: v10 - v7 on test has sd 0.30, so for a
  0.053 drop it must correlate about -0.19 with v7's public errors (if the public subset looks like the whole test),
  vs about zero on OOF where the CV tied. One public result, about 2-3 SE.
- Next file: **blend_v11** (`submissions/blend_v11.json`), two-stage: v7-pool NNLS, then the effnetv2-s CNN at a share
  fit by least squares on the inner nested base OOF (refit per outer fold). CNN share 0.131, nested CV 12.4074 (beats
  joint NNLS 12.4242). Handed over for the first submission after 00:00 KST 2026-10-07. On test it moves about 1.9x
  further along v10's direction (v11 - v7 sd 0.60, correlation 0.96 with v10 - v7): expect public about 11.79 if the
  CNN effect is linear in its share, about 11.86 (v7 minus the CV gain) if v10's drop was luck.
- Next GPU runs (user picks exclusive or shared GPU mode in the thread first): `notes/handoff/laptop-gpu.md` section 6.
- Where the noise error is (orchestrator, 23:50 KST): blend_v11 OOF RMSE by calibrated grain count
  (`cal_seg_count_density`, Spearman 0.02 with noise) x raw `ic_noise` tercile (train cuts 7.49 / 11.92):
  fine 7.75 / 7.89 / 11.36, mid 8.82 / 11.03 / 14.15, coarse 13.43 / 14.14 / 18.19. A fit r^2 ~ a + b/N + c*noise gives
  c = 9.0 (90% CI 5.3-13.0): about 56 of the ~151 MSE units are tied to noise, in every grain-size tercile.
  Do not stratify by `ic_acg_len50_gm` or `ic_seg_L_gm` for this: they are biased by noise (Spearman 0.43 with
  ic_noise) and make the noisy tercile look like a coarse-grain confound.
- Cloud CPU next (running): idea 1 below, a high-noise specialist restorer (cnn-trainer; outputs to
  `data_restored_hn/`, git-ignored), then restored features and a ridge screen (feature-engineer).

## End of day 2026-10-06
- Public LB so far: blend_v2 12.3454, v4 12.0329, v6 11.9248, **v7 11.8795** (best). In all four, public came in
  0.55-0.6 below the nested CV. LB #1 that day: 9.2442.
- File handed over for the remaining slot (today's last or 00:00 KST 2026-10-07): **blend_v10** = v7 pool + the two
  laptop-GPU CNNs. Nested CV 12.4242 vs v7 12.4254 (bootstrap 90% of the difference [-0.032, 0.031]), so it is a tie:
  weights cs24 0.244, feat6_rest_ridge_all 0.393, feat3 lgbs 0.166, feat5 milspl 0.089, feat2 lgbs hetN 0.038,
  effnetv2-s CNN 0.07. Test predictions differ from v7 by sd 0.30 (max 1.25), so expect public about 11.88.
  CSV in `/mnt/project-files/work/hardness-cache/submissions/`. When the user reports the score, add it to
  `experiments/LEADERBOARD.md` and memory.
- Laptop GPU run (RTX 5060, exclusive use 19:16-20:11 KST, returned with the done file; plan in
  `notes/handoff/laptop-gpu.md`), base flags `--input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30`, lr 1e-3:
  | experiment | seeds | CV | time per fold-seed | blend weight |
  |---|---|---|---|---|
  | `cnn_r18_rawnlm_degcons_gpu` (resnet18) | 3 | 13.593 | ~31 s | 0 |
  | `cnn_ev2s_rawnlm_degcons_gpu_s3` (tf_efficientnetv2_s.in21k_ft_in1k) | 3 | 13.246 | ~76 s | 0.07 |
  CNN residuals correlate 0.93 with the blend; even the best in-sample mix of resnet18 into v7 is weight 0. More seeds or
  bigger backbones of the same CNN are unlikely to move the blend. lr 3e-4 was not screened.
- The laptop session returned results as messages (2-decimal values in ID order plus checksums); the cloud copies and
  checking script are in the hardness-cache (`laptop-gpu-2026-10-06/`, `scripts/ingest.py`, `scripts/paired.py`).
- No CPU candidate beat v7 either (restored-image embeddings, restored-feature GBM, add-alongside ridge, tercile
  stacker, stretch recalibration, curated subsets).
- Where the error is: the noisy SNR tercile (blend about 14.7 vs about 11 on the other two) and coarse-grain images,
  whose extra variance tested as per-grain label noise the image does not show (every probe null; list in memory).
- Untested ideas for the next session, best first:
  1. A restorer trained only on low-SNR degradations, then restored features + ridge for the noisy tercile (restoration
     so far helps the mid tercile, not the noisy one).
  2. A CNN on raw + restored channels on the laptop GPU (needs `data_restored/` and `restore_unet.pt` on the laptop).
  3. Only if 2 gets real blend weight: more GPU seeds/backbones.

## Repo history rewrite (2026-10-06 ~11:30 UTC, user's choice)
- The user asked for the work on `main`: main was fast-forwarded to `claude/lb-under-10-7080jr`.
- Before that, 16 early `experiments/*/test.csv`, two EDA tables that copied the train labels
  (`experiments/eda/blend_v*_nested_oof.csv`) and the restorer's per-image test statistics
  (`experiments/restore/apply_change.csv`) were still tracked. They were untracked, git-ignored, and on the user's
  choice removed from every commit; main and the branch were force-pushed. Local copies:
  hardness-cache `experiments-test.tar` and `local-only-files.tar`. Pre-rewrite history: `git/lb-under-10-prepurge.bundle`.
- Old clones (the laptop) must re-sync once with `git fetch origin && git reset --hard origin/claude/lb-under-10-7080jr`.

## Cloud container lesson (pause 2026-10-06 07:05-07:38 UTC)
- The container sleeps a few minutes after the session goes idle and kills every detached job (the restorer's training
  died at step 3500 that day). Checkpoint long jobs (`src/restore.py --ckpt-every`, `train_cnn` per fold-seed caches).
- In a fresh session, re-brief agents from `notes/handoff/*.md`; agent ids from an old session do not carry over.

## Where we are (2026-10-06 evening)
| model family | best saved experiment | CV RMSE |
|---|---|---|
| OOF blend (NNLS, nested CV) | `blend_v10` (ties `blend_v7`, public 11.8795) | **12.42** |
| handcrafted features | `feat6_rest_ridge_add` / `feat6_rest_ridge_all` (restored-image features) | 12.52 / 12.53 |
| frozen embeddings | `emb_effv2s_256_gridge3_noise_csbag_rawrest` | 12.84 |
| CNN fine-tune | `cnn_ev2s_rawnlm_degcons_gpu_s3` (laptop GPU) | 13.25 |
| baseline | `feat_lgb` | 14.20 |

Target std is 17.7 (mean predictor). Submission CSVs are git-ignored; copies are in the hardness-cache `submissions/`.

## Restore a fresh cloud session (about 5 min)
```bash
cd /home/claude/microstructure-hardness-prediction        # repo, branch claude/lb-under-10-7080jr
git fetch origin claude/lb-under-10-7080jr && git checkout claude/lb-under-10-7080jr && git pull
python3 -m pip install -q -r requirements.txt             # torch comes from PyPI; PyWavelets needed by skimage
mkdir -p data && (cd data && unzip -qo /mnt/project-files/open.zip)
cp -r /mnt/project-files/work/hardness-cache/data/. data/  # folds, features v1-v3, embeddings, CNN caches
tar -xf data/data_restored.tar && rm data/data_restored.tar  # restored images -> data_restored/ (repo root)
tar -xf /mnt/project-files/work/hardness-cache/experiments-test.tar   # experiments/*/test.csv (git-ignored)
tar -xf /mnt/project-files/work/hardness-cache/figures-local.tar      # optional: figures that show competition images
mkdir -p submissions && cp /mnt/project-files/work/hardness-cache/submissions/* submissions/
```
Pretrained weights download from GitHub releases on first use (`src.common.create_timm`).
huggingface.co is blocked in the cloud sandbox.

## Cross-agent findings (read before planning)
- All families capture the same signal: residual correlations 0.87-0.93; blends gain only about 0.3-0.6.
  Getting to 10 needs a new signal, not more of the same models.
- Dark-phase fraction dominates (per-grain fraction Spearman about 0.7 on clean images). Normalise by the
  local matrix level: shading (13-19 grey levels) is comparable to the phase contrast (about 30).
- Error rises with image noise: about 12 on the cleaner half vs about 15 on the noisier half.
  Denoising the input hurts embeddings but an extra NLM channel helps the CNN.
- The target is heteroscedastic: coarse-structure images have hardness std 21.5 vs 13.5 for fine ones,
  and OOF RMSE 16.7 vs 9.3. Nothing measured so far explains the extra variance in coarse images.
  This is the biggest open question; candidates: a location-dependent label (e.g. an indent-like local
  sample), a grain-level property not yet measured, or label noise built into the generator.
- Every model regresses to the mean (RMSE about 18 in the extreme target quintiles vs about 9 in the middle).
- With 500 images, differences under about 0.1 RMSE are noise; single-fold screens are about ±0.4.
- `src/train_gbm.py` `feat_lgb` / `feat_cat` used early stopping on the validation fold (optimistic). Newer runs use fixed trees.

## Spatial-heterogeneity signal (2026-10-06 afternoon, eda-analyst; experiments/eda/NOTES.md from line 96)
- Part of the coarse-image "label noise" is model error. Images with zones of large grains next to zones of small grains
  are harder than their mean grain size predicts, and global pooling or global mean-size features average this away.
- Three per-image feature files (train + test, fixed transforms): `data/eda_feats_lledge.parquet` (block heterogeneity of
  edge energy), `data/eda_feats_ecs.parquet` (cross-cell std of g2a embeddings), `data/eda_feats_lf.parquet` (local
  grain-size field; GBMs only). Copies are in the hardness-cache `data/`.
- Fold-paired gains: ridge het 13.101 -> 12.824 (5/5 folds); lgbs het 12.991 -> 12.771 (4/5); blend about -0.18.
- Cross-fitted probes are needed for residual tests: naive OOF-residual probes are biased negative by stacking leakage.

## Noise-floor finding (2026-10-06 morning, eda-analyst) - SUPERSEDED by the section above; the floor is only an upper bound
- Residual variance follows label noise ~ b/N (N = calibrated grain count `cal_seg_count_density*6.5536`,
  b about 28k, 90% CI 21.9k-32.4k) plus a measurement term ~ c/snr. Clean images already sit at that floor
  in every grain-size tercile (orchestrator re-check: clean coarse 14.5 vs floor 14.5, mid 9.2 vs 9.7, fine 7.8 vs 7.8).
- Floor for a perfect model: train CV about 11.2-11.6, full test about 10.7-11.1; public 300 subset median 10.7
  [9.2, 12.2]. LB <= 10 is out of reach on expectation; blend_v2 is expected around public 12.6 [11.4, 13.8].
- All reducible error is in degraded (noisy/blurred) images: focus on measurement under noise/blur
  (calibration v2, NLM channel for CNNs, training on synthetic degradations). Stop adding coarse-image grain,
  location or texture features (all null). Public-LB differences under about 0.3 are noise.

## Compute notes
- Cloud sandbox: 4 CPU (Xeon with AMX), no GPU. bf16 + channels_last makes CNN training about 3x faster.
  Budget threads per agent (e.g. CNN 2, features 1, embeddings 1) when running agents in parallel.
- Laptop GPU (RTX 5060 8GB): this project's device folder is `C:\Daker\microstructure-hardness-prediction`
  (branch checked out, data copied to `data\`, `.venv` with torch 2.11+cu128). The GPU is shared with the user's
  robot project. Ask the user which mode applies before running anything on it:
  - shared: wrap every GPU command as
    `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- <command>`; on CUDA OOM lower `--bs`;
  - exclusive hand-over (as on 2026-10-06 19:16 KST): `--device cuda` directly, then return the GPU with PowerShell
    `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done` when all GPU work is done.
  Never touch the robot project otherwise.
- "Jev" (TypeSafe AI) is a remote, text-only API model and is banned by competition rule 2, so it is not used.
