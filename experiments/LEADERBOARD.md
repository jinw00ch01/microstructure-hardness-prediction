# Experiment leaderboard

| exp | CV RMSE | public LB | notes |
|---|---|---|---|
| feat_lgb | 14.202 |  | lgb on features.parquet |
| feat_ridge | 14.451 |  | ridge on features.parquet |
| feat_svr | 14.659 |  | svr on features.parquet |
| emb_r18_256_krr_feats | 13.902 |  | krr head (inner-CV hp per fold: {"gamma_mult": 0.0625, "alpha": 0.04642}; {"gamma_mult": 0.0625, "alpha": 0.04642}; {"gamma_mult": 0.0625, "alpha": 0.04642}; {"gamma_mult": 0.0625, "alpha": 0.04642}; {"gamma_mult": 0.0625, "alpha": 0.04642}) on resnet18.a1_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=mean view_std=False transform=none block_norm=False pca=0 feats=features.parquet x1.0; licenses: apache-2.0 |
| emb_r18_256_ridge_aug | 13.928 |  | ridge head (inner-CV hp per fold: {"alpha": 1000.0}; {"alpha": 1000.0}; {"alpha": 1000.0}; {"alpha": 1000.0}; {"alpha": 1000.0}) on resnet18.a1_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=aug view_std=False transform=none block_norm=False pca=0 feats=no; licenses: apache-2.0 |
| feat2_lgb | 14.226 |  | lgb on features_v2.parquet; cols=all(178) |
| feat2_ridge | 14.055 |  | ridge on features_v2.parquet; cols=all(178) |
| feat2_lgb_v1v2 | 14.127 |  | lgb on features_v2.parquet,features.parquet; cols=all(280) |
| emb_effv2s_256_ridge_aug | 13.675 |  | ridge head (inner-CV hp: alpha=562.3 x3 alpha=1000 x1 alpha=316.2 x1) on tf_efficientnetv2_s.in21k_ft_in1k (apache-2.0, 256px, 4 views); stages=2,3 pools=mean,std view_mode=aug view_std=False transform=none block_norm=False pca=0 feats=no; licenses: apache-2.0 |
| feat2_v3_lgb | 13.612 |  | lgb on features_v3.parquet; cols=all(128) |
| feat2_v3_ridge | 13.219 |  | ridge on features_v3.parquet; cols=all(128) |
| feat2_v23_lgb | 13.530 |  | lgb on features_v3.parquet,features_v2.parquet; cols=all(306) |
| feat2_v23_ridge | 13.546 |  | ridge on features_v3.parquet,features_v2.parquet; cols=all(306) |
| feat2_v3_fwd | 13.721 |  | fwd on features_v3.parquet; cols=all(128); in-fold forward selection (ridge inner-CV, max 20) |
| feat2_v123_fwd | 14.165 |  | fwd on features_v3.parquet,features_v2.parquet,features.parquet; cols=all(408); in-fold forward selection (ridge inner-CV, max 20) |
| cnn_r18_c224_e30 | 13.787 |  | resnet18.a1_in1k ep30 bs16 crop224 lr0.001 hlr1.0 wd0.01 mse pool=avg drop0.0 dp0.0 ema0.0->final seeds1 aug(b0.03 c0.1 n0.03@0.5 blur1.0@0.0) tta8; fixed schedule, no val checkpoint selection |
| blend_v1 (blend) | 13.151 |  | nested-CV, 16 models |
| emb_effv2s_256_gridge3_aug | 13.514 |  | gridge3 head (inner-CV hp: lam_view=0,lam_cell=0,lam_aug=16,alpha=3.162 x2 lam_view=4,lam_cell=0,lam_aug=16,alpha=0.3162 x1 lam_view=4,lam_cell=0,lam_aug=4,alpha=10 x2) on tf_efficientnetv2_s.in21k_ft_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=aug cells=global augs=True view_std=False transform=none block_norm=False pca=0 feats=no; licenses: apache-2.0 |
| emb_effv2s_256_gridge3_aug_featv3 | 13.254 |  | gridge3 head (inner-CV hp: lam_view=0,lam_cell=0,lam_aug=4,alpha=316.2 x2 lam_view=0,lam_cell=0,lam_aug=64,alpha=100 x1 lam_view=16,lam_cell=0,lam_aug=64,alpha=316.2 x1 lam_view=0,lam_cell=0,lam_aug=64,alpha=316.2 x1) on tf_efficientnetv2_s.in21k_ft_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=aug cells=global augs=True extra_rows=- view_std=False transform=none block_norm=False pca=0 feats=features_v3.parquet x1.0; licenses: apache-2.0 |
| feat2_v3cal_ridge | 13.177 |  | ridge on features_v3.parquet,features_cal.parquet; cols=all(157) |
| feat2_v3cal_lgbs | 13.346 |  | lgbs on features_v3.parquet,features_cal.parquet; cols=all(157) |
| feat2_v23_lgbs | 13.212 |  | lgbs on features_v3.parquet,features_v2.parquet; cols=all(312); 3 seeds |
| feat2_v3cal_ridge_het | 13.101 |  | ridge on features_v3.parquet,features_cal.parquet; cols=all(157); hetero weights ~ log-var(log ic_acg_len50_gm) in-fold |
| feat2_v23cal_lgbs_het | 12.991 |  | lgbs on features_v3.parquet,features_v2.parquet,features_cal.parquet; cols=all(335); 3 seeds; hetero weights ~ log-var(log ic_acg_len50_gm) in-fold |
| blend_v2 (blend) | 12.963 |  | nested-CV, 23 models |
| emb_effv2s_r18_256_gridge3_aug | 13.414 |  | gridge3 head (inner-CV hp: lam_view=4,lam_cell=0,lam_aug=4,alpha=100 x3 lam_view=4,lam_cell=0,lam_aug=16,alpha=100 x1 lam_view=4,lam_cell=0,lam_aug=16,alpha=31.62 x1) on tf_efficientnetv2_s.in21k_ft_in1k (apache-2.0, 256px, 4 views) + resnet18.a1_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=aug cells=global augs=True extra_rows=- view_std=False transform=none block_norm=False pca=0 feats=no; grid={"lam_view": [0, 4], "lam_aug": [4, 16, 64]}; licenses: apache-2.0 |
| emb_effv2s_256_gridge3_noise | 13.442 |  | gridge3 head (inner-CV hp: lam_view=4,lam_cell=0,lam_aug=4,alpha=10 x3 lam_view=16,lam_cell=0,lam_aug=4,alpha=3.162 x1 lam_view=4,lam_cell=0,lam_aug=4,alpha=31.62 x1) on tf_efficientnetv2_s.in21k_ft_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=aug cells=global augs=True:noise extra_rows=tf_efficientnetv2_s.in21k_ft_in1k_256_augs2 view_std=False transform=none block_norm=False pca=0 feats=no; licenses: apache-2.0 |
| cnn_r18_rawnlm_e30 | 13.837 |  | resnet18.a1_in1k in=raw+nlm norm=global ep30 bs16 crop224 lr0.001 hlr1.0 wd0.01 mse pool=avg drop0.0 dp0.0 ema0.0->final seeds2 aug(b0.03 c0.1 n0.03@0.5 blur1.0@0.0) tta8; fixed schedule, no val checkpoint selection |
| emb_effv2s_r18_256_gridge3_noise | 13.403 |  | gridge3 head (inner-CV hp: lam_view=4,lam_cell=0,lam_aug=4,alpha=100 x4 lam_view=4,lam_cell=0,lam_aug=16,alpha=100 x1) on tf_efficientnetv2_s.in21k_ft_in1k (apache-2.0, 256px, 4 views) + resnet18.a1_in1k (apache-2.0, 256px, 4 views); stages=1,2,3 pools=mean,std view_mode=aug cells=global augs=True:noise extra_rows=tf_efficientnetv2_s.in21k_ft_in1k_256_augs2,resnet18.a1_in1k_256_augs2 view_std=False transform=none block_norm=False pca=0 feats=no; grid={"lam_view": [0, 4], "lam_aug": [4, 16, 64]}; licenses: apache-2.0 |
| feat2_v3calc2_ridge_het | 13.067 |  | ridge on features_v3.parquet,features_cal.parquet,features_cal2.parquet; cols=all(180); hetero weights ~ log-var(log ic_acg_len50_gm) in-fold |
| feat2_v23cal_lgbs_hetN | 13.008 |  | lgbs on features_v3.parquet,features_v2.parquet,features_cal.parquet; cols=all(335); 3 seeds; hetero weights var = b0 + sum b/(cal_seg_count_density,ic_ridge_snr) in-fold |
| blend_v3 (blend) | 12.982 |  | nested-CV, 29 models |
