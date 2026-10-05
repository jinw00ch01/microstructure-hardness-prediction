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
