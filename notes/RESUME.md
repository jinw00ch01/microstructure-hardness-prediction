# Resume guide (updated 2026-10-18 14:54 KST)

Goal: public LB RMSE in the 8s, below 9.0 (user, 2026-10-15 12:56 KST; was <= 9 from 2026-10-10 16:11 KST, <= 10 before). Deadline 2026-10-19 10:00 KST. The final is picked
automatically: the submission with the best public score (user, 16:12 KST). There is no manual choice of two finals, so
every file submitted must be one we would accept as the final if it tops public (no public-tuned or untested probes that
could win public by noise and be worse on private). The earlier "finals v27u + v28n" rules are moot.


## 2026-10-18 14:54 KST: v39q8a4b read 9.9434 = NEW BEST; kriging29, env run_env_1018q4b.sh; pool re-judged (10-19 file 1 = lineB PALL) - read this first
- blend_v39q8a4b.csv (10-18 file 5, Q8A4B = [B8f, APf, AP4Bf]) read 9.9434377740 (user 14:48 KST): -0.0173 public MSE vs Q3A6, -0.1217
  vs Q8A; kriging28 had predicted 9.9469 (z -0.20), the full predictive 9.9462 (z -0.26). It is the best file and the automatic final.
  10-18 slots are used up; 10-19 has 5 slots from 00:00 KST until the 10:00 KST deadline.
- Public model: hardness-cache scripts/d1016/pub/forensics29.py + kriging29.py (mk_pubN.py 29 v39q8a4b=9.9434377740:cfe4f97c);
  read29.py / read29_out.txt (LOO z v39q8a4b -0.26, v39q3a6 -0.22).
- Env for every later build: d1018/run_env_1018q4b.sh "v39q8a4b 9.9434377740" (= run_env_1018a6b.sh, 24-file skip list, plus
  KRIG_LATEST -> kriging29 and the v39q8a4b best case: BEST_RUNS = its 9 runs; every builder reproduced it bit for bit). Its line A
  (BASE = B8f,APf,AP4Bf) cannot be built (AP4Bf is a g1mix11 new family); BASE per d1018/ADDENDUM_pool1019.txt as before.
- Whole pool rebuilt 14:51 KST with d1018/rebuild_pool1019.sh into tag q8a4b_k29 (record d1018/g1mix11/q8a4b_k29/README.txt). No SAFE,
  no NEAR-TIE(strict). FP-BOLD order (P(beat v39q8a4b) at x1.0): lineB PALL [B8f, AP6f, AP6Bf, AP4f, AP4Bf] 0.35 (train -0.079 3/5,
  k29 9.9460); lineAq8a PALL 0.32 (G1 rms 0.154 to lineB PALL, so one of the two); lineB P4A 0.32; lineAq8a P4A 0.23; lineAq8a P6AB
  0.18; g1mix9 lineAq8a P4B 0.17.
- Next: workflow preverify-1019-files (file side of lineB PALL, lineAq8a PALL, lineAq8a P4A, lineB P4AB; started 14:46 KST), then a
  rules + public-model verification of lineB PALL under q8a4b_k29 and its handover as 10-19 file 1 (submit after 00:00 KST). After
  each 10-19 reading: mk_pubN.py next N, a new env, rebuild_pool1019.sh with a new tag, pick again, verify, hand over.
- Card 14:44 KST (pending, d1018/CARD_boldcap1019.txt): fill the spare 10-19 slots with BOLD files (option 0, recommended) or keep
  the 2-BOLD cap. Section 28 still needs "섹션28 실행".

## 2026-10-18 14:41 KST: 10-18 file 5 handed over = blend_v39q8a4b (B8f + APf + AP4Bf, BOLD by P(beat)); env run_env_1018a6b.sh - read this first
- Section 27 complete 13:52 KST (12 runs, rc 0, no OOM). AP4Bf = b4ap_locK_fsim, _s1, _s2 (27-10..12: 25-4's B1-recipe command with
  --arch tf_efficientnet_b4.ap_in1k; walls 12:31:02-12:57:54, 12:57:54-13:24:28, 13:24:28-13:52:07 KST) ingested: parquet md5 train/test
  4dc94187/bc190f82, ebf517f9/d78c2b3d, f68a5c88/d7dcaeeb; gates 0.990/0.991/0.990; het_cnn-tables.tar 140 parquet members.
- Final builds 13:57-13:58 KST (env run_env_1018a6.sh, no ALLOW_PARTIAL): hardness-cache scripts/d1018/g1mix11/q3a6_ap4b/{lineB,
  lineAq8ab3, lineAq8a} (README.txt, judge_slots_pool.txt, judge_bold_fp_pool.txt). No SAFE; NEAR-TIE lenient only (lineAq8a P6AB).
  FP-BOLD order (P at x1.34): 1 lineAq8a P4AB [B8f, APf, AP4Bf] 0.448; 2 lineB PALL [B8f, AP6f, AP6Bf, AP4f, AP4Bf] 0.407 (train
  -0.049 4/5); 3 lineAq8a PALL [B8f, APf, AP6f, AP6Bf, AP4f, AP4Bf] 0.403; 4 lineB P4A 0.373 (the interim leader); 5 lineAq8a P4A 0.337.
- Verified 14:06-14:37 KST by workflow verify-p4ab-1018 (record d1018/verify_p4ab_1018/): all PASS_WITH_NOTES, critic DELIVER.
  **Handed over 14:40 KST as submissions/blend_v39q8a4b.csv** (test cfe4f97c, oof 0046c8d9). Told JINWOO: P(beat Q3A6) about 43%
  (band 40-52%); train-worse than Q3A6 (+0.030, 3/5 folds, drop-top-3 +0.264); the card's B8f+APf+AP6Bf was replaced after section
  27; private about neutral if it wins public; the train-better 2nd file (lineB PALL, about 38%) stays a 10-19 candidate.
- From the verification: kriging28's x1.34 sd factor does not reproduce from the 34 readings (LOO rms z 0.98; quote P at x1.0); the
  top three are near-equal (about 0.4 each) and dropping the v38q8a or v39q8ab3 reading puts lineB PALL first; 97.5% of P4AB's
  predictive variance is the AP4Bf part no reading has measured.
- Env for every later build: d1018/run_env_1018a6b.sh "v39q3a6 9.9443061756" (= run_env_1018a6.sh + v39q8a4b in SUBMITTED_EXTRA,
  24-file skip list; hash in d1018/NOTES.sha256). The runners-up stay >= 0.25 G1 rms from v39q8a4b (lineB PALL 0.279, lineAq8a PALL
  0.278, lineB P4A 0.324, lineAq8a P6AB 0.299), so they remain 10-19 candidates.
- After the v39q8a4b reading: mk_pubN.py 29 v39q8a4b=<score>:cfe4f97c -> forensics29/kriging29, read29.py; a new env (a v39q8a4b
  best case if it beats 9.9443061756: BEST_RUNS = the 9 runs of B8f, APf, AP4Bf); rebuild the pool under it; judge_slots1018.py and
  KRIG=kriging29 judge_bold_fp1018.py over the 8 pool dirs; then the 10-19 slate (5 slots until 10:00 KST): SAFE first, BOLD by
  P(beat) at most 2, NEAR-TIE(strict) at most 2, each verified before handover.
- Section 28 (AdvProp B7/B8; d1018/g1mix12) still needs "섹션28 실행" from JINWOO in the main thread; GPU idle since 13:52 KST
  (reminded 14:40 KST).

## 2026-10-18 13:18 KST: BOLD order now = full-predictive P(beat best) (user card 12:34); slot 5 leader B8f+AP4f pre-verified; waiting for AP4Bf - read this first
- 11:22-11:49 KST workflow verify-p3b-1018 (record hardness-cache scripts/d1018/verify_p3b_1018/) found that the written-rule slot 5
  pick g1mix10 line B P3B [B8f, NS3Bf] has its public score fixed by the readings (P3B = 1.5 Q8AB3 + 0.5 v38q8 - Q8AB on the test
  rows, ~9.950, P(beat Q3A6) ~0): kriging's kpred sd leaves out the candidate's own sampling term. Same for line A Q8A P6A.
- Amendment d1018/AMEND_bold_fullpred_1018.txt (11:57 KST; model hardness-cache scripts/d1016/pub/fullpred28.py, prospective check
  fullpred28_prospective.py, judge d1018/judge_bold_fp1018.py). **Card 11:58 KST, JINWOO picked "이길 확률로 고르기" at 12:34 KST**
  (record d1018/DECISION_bold_fullpred_1018.txt with an erratum; hashes in d1018/NOTES.sha256): every BOLD pick (10-18 slot 5 and
  the 10-19 BOLD slots) = the FP-BOLD file with the highest full-predictive P(beat best). FP-BOLD = not SAFE, vs-BESTfixed dMSE <=
  +0.42, latest kriging <= best + 0.0132, skip rule (G1 rms >= 0.25 vs all submitted or handed-over files), full-predictive sd
  (x1.34) >= 0.002; order by P(beat) (x1.34; within 0.01 -> better dMSE). SAFE first, section 27 completion, verification, caps stand.
  P(beat) is quoted to the user at x1.0 with a band. A train-worse pick must be named as such at handover (told the user 12:35 KST).
- Section 27: AP6Bf (b6ap_locK_fsim, _s1, _s2) complete 11:14 KST -> d1018/g1mix11/q3a6_ap6b (leader then lineAq8a P6AB [B8f, APf,
  AP6Bf], pre-verified by verify-p6ab-1018: DELIVER conditional). AP4f (b4ap_locK_real5, _s1, _s2; 27-7..9 11:14:41-12:31:02 KST)
  complete -> rebuilt 12:39-12:40 KST into d1018/g1mix11/q3a6_ap4/{lineB, lineAq8ab3, lineAq8a} (README, judge_slots_pool.txt,
  judge_bold_fp_pool.txt). No SAFE. FP-BOLD leader: **line B P4A [B8f, AP4f]** (test eac61115, oof da32c9db): vs Q3A6 +0.055 2/5
  (train-WORSE), drop-top-3 +0.338; kriging28 9.9518, full 9.9511, P(beat Q3A6) 0.33 at x1.0 (band 0.26-0.44); leads only through
  its uncertainty (AP4f direction, no reading). Next: lineAq8a P4A 0.337 (x1.34), line B PALL 0.321, lineAq8a PALL 0.288, P6AB 0.274.
- Pre-verified 12:45-13:15 KST by workflow verify-p4a-1018 (record d1018/verify_p4a_1018/): all PASS_WITH_NOTES, critic DELIVER
  conditional on section 27 complete, no SAFE, P4A still the strict leader, md5s unchanged. Scratch preview (1 AP4Bf seed, never an
  option) showed no AP4Bf option overtaking it.
- AP4Bf (b4ap_locK_fsim, _s1, _s2; B1 recipe on tf_efficientnet_b4.ap_in1k): 27-10 12:31:02-12:57:54 KST ingested (parquet md5 train
  4dc94187 / test bc190f82, gate 0.990, tar 136 members); 27-11 started 12:57:54, then 27-12 (~13:52 KST). Then: rebuild the 3 g1mix11
  lines into d1018/g1mix11/q3a6_ap4b/, both judges over the 8 pool dirs, hand over the top FP-BOLD (new verification if not P4A), as
  submissions/blend_v39<name>.csv, new env with it in SUBMITTED_EXTRA (24 files). Section 28 still needs "섹션28 실행" from the user.

## 2026-10-18 11:05 KST: Q3A6 read 9.9443 = NEW BEST; kriging28, env run_env_1018a6.sh; pool re-judged, no file for 10-18 slot 5 - read this first
- blend_v39q3a6.csv (10-18 file 4, SAFE, Q3A6 = [B8f, APBf, NS3Bf, AP6f]) read 9.9443061756 (user 10:58 KST): -0.0947 public MSE vs
  Q8AB3 (9.9490659450), -0.1045 vs Q8A; kriging27 had predicted 9.9464 (z -0.15). It is the best file and the automatic final.
  JINWOO also wrote (10:59 KST) "이제 전체 제출횟수 6번 남음": 10-18 has 1 slot left, 10-19 has 5 (until 10:00 KST).
- Public model: hardness-cache scripts/d1016/pub/forensics28.py + kriging28.py (mk_pubN.py 28 v39q3a6=9.9443061756:dfa00a78;
  read28.py / read28_out.txt: LOO z v39q3a6 -0.25, v39q8ab3 -0.18, v38xn8 -1.83, v35xn +1.39).
- Env for every later build: hardness-cache scripts/d1018/run_env_1018a6.sh "v39q3a6 9.9443061756" (= run_env_1018q3q.sh, 23 files in
  the skip list, plus KRIG_LATEST -> kriging28 and the v39q3a6 best case: BEST_RUNS = its 12 runs; every builder reproduced Q3A6 bit
  for bit). The new best's line A (BASE = B8f,APBf,NS3Bf,AP6f) cannot be built (AP6f is a g1mix11 new family), so the default BASE
  fails safe; set BASE=B8f (line B), BASE=B8f,APBf,NS3Bf (lineAq8ab3, g1mix11 only) or BASE=B8f,APf (lineAq8a) after sourcing,
  per d1018/ADDENDUM_pool1019.txt.
- Whole pool re-judged 11:01-11:02 KST into q3a6_k28 dirs (record d1018/g1mix11/q3a6_k28/README.txt): no SAFE, no NEAR-TIE(strict).
  BOLD: g1mix10 line B P3B [B8f, NS3Bf] k28 9.9502 (P(beat) 0.31, train +0.064 2/5); g1mix11 Q8A-line P6A [B8f, APf, AP6f] 9.9516.
  SKIP: g1mix10 Q8A-line P3B 9.9421 (rms 0.241 vs Q3A6) and PALL 9.9443 (0.181).
- Card 11:03 KST (main thread): use the 10-18 last slot for a BOLD file or keep the 2-BOLD cap. **JINWOO picked "BOLD로 쓰기"
  11:04 KST.** Rule: hardness-cache scripts/d1018/DECISION_bold3_1018.txt (hash in d1018/NOTES.sha256): slot 5 = the lowest-kriging
  BOLD in the whole pool once section 27 is complete (or the families complete at 16:00 KST if a run fails), SAFE first if one
  exists, verified before handover. 10-19 keeps the 2-BOLD cap.
- Section 27: 27-5 b6ap_locK_fsim_s1 ingested (laptop 09:46:49-10:31:02 KST, 2653 s, rc 0; parquet md5 train 1215cb41 / test
  47f65269; tar 126 parquet members); 27-6 _s2 started 10:31:02 KST. Then AP4f (27-7..9) and AP4Bf (27-10..12).

## 2026-10-18 10:28 KST: Q3A6 (Q8AB3 + AP6f) verified and handed over as 10-18 file 4 (SAFE); env run_env_1018q3q.sh
- Workflow verify-q3a6-1018 (run wf_5b8a1dda-048, 10:00-10:24 KST): rules, file rebuild, readers and public model all
  PASS_WITH_NOTES, critic DELIVER (independent rebuild matched dfa00a78 / 6b3a3cdd; hand judge of all 23 option rows agreed with
  judge_slots1018.py). The SAFE bar (g1mix11 options.txt:32-33) has no drop-top-3 term; precedent blend_v37n2 (SAFE 10-16, dt3 +0.241).
  Told JINWOO plainly: the train edge rests on 3 rows (top 3 rows -86.7 SSE vs total -27.2), expected gain about -0.054 public MSE,
  P(beat Q8AB3) 0.58 under kriging27 (kriging26 gives 9.9560, P 0.37), a worse reading leaves Q8AB3 as the final.
- Handed over 10:26 KST: hardness-cache submissions/blend_v39q3a6.csv (10-18 file 4, SAFE). Env for every later build:
  hardness-cache scripts/d1018/run_env_1018q3q.sh "v39q8ab3 9.9490659450" (= run_env_1018q3.sh plus v39q3a6 in SUBMITTED_EXTRA,
  23 files in the skip list; hash in d1018/NOTES.sha256). Use run_env_1018q3.sh only for a rule-4 re-judge before v39q3a6 is submitted.
- After the v39q3a6 reading: mk_pubN.py 28 v39q3a6=<score>:dfa00a78, read28.py, a new env (with a v39q3a6 best case if it beats
  9.9490659450; then line A BASE = [B8f, APBf, NS3Bf, AP6f], and g1mix11 line A cannot carry AP6f-based options), re-judge the pool.
- Pool membership settled 10:31 KST, before the v39q3a6 reading: hardness-cache scripts/d1018/ADDENDUM_pool1019.txt (hash in
  d1018/NOTES.sha256). Line A options built on a superseded best file's families since Q8A stay in the pool (lineAq8a = [B8f, APf];
  lineAq8ab3 = [B8f, APBf, NS3Bf] if a later file replaces Q8AB3), rebuilt against the new BESTfixed with the latest kriging (rule
  2(d) reading). Under kriging27 the 10-19 BOLD order is lineAq8a P6A 9.9539, g1mix9 lineAq8a P4B 9.9546, line B P6A 9.9558.
- 10-18: 4 of 5 slots handed over; 1 left (SAFE, or NEAR-TIE(strict) 2 of 2). 27-5 b6ap_locK_fsim_s1 started 09:46:49 KST, then 27-6;
  AP6Bf complete about 11:15 KST -> rebuild g1mix11 under run_env_1018q3q.sh (or the k28 env if v39q3a6 has been read).

## 2026-10-18 10:01 KST: Q8AB3 read 9.9491 = NEW BEST; kriging27, env run_env_1018q3.sh; SAFE candidate Q3A6 in verification
- blend_v39q8ab3.csv (10-18 file 3, BOLD 2 of 2, Q8AB3 = [B8f, APBf, NS3Bf]) read 9.9490659450 (user 09:51 KST): -0.0098 public MSE vs
  Q8A (9.9495571078) and -0.1621 vs Q8AB; kriging26 had predicted 9.9623 (z -0.53). It is the best file and the automatic final.
- Public model: hardness-cache scripts/d1016/pub/forensics27.py + kriging27.py (mk_pubN.py 27 v39q8ab3=9.9490659450:c0573673;
  read27.py / read27_out.txt: LOO z v39q8ab3 -0.79, v38xn8 -1.83, v35xn +1.44).
- Env for every later build: hardness-cache scripts/d1018/run_env_1018q3.sh "v39q8ab3 9.9490659450" (= run_env_1018k26q.sh, 22 files
  in the skip list, plus KRIG_LATEST -> kriging27 and the v39q8ab3 best case: line A BASE = [B8f, APBf, NS3Bf], BEST_RUNS = its 9 runs;
  every builder reproduced Q8AB3 bit for bit). Line B: export BASE=B8f. The earlier Q8A line A stays in the pool: BASE=B8f,APf.
- Whole pool re-judged 09:56-09:57 KST (NOTE_slate1018 rule 5) into q8ab3_k27 dirs (d1018/g1mix11, d1017/g1mix10, d1017/g1mix9) and
  d1016/g1mix8/best_q8ab3_k27: SAFE = g1mix11 line A P6A = Q3A6 [B8f, APBf, NS3Bf, AP6f]: vs Q8AB3 -0.054 4/5, 40-split mean -0.071
  P 0.20, drop-top-3 +0.120, kriging27 9.9464 (P(beat best) 0.58), G1 rms vs Q8AB3 0.331. BOLD (for 10-19, the 10-18 cap is used):
  g1mix11 Q8A-line P6A [B8f, APf, AP6f] 9.9539; g1mix9 Q8A-line P4B [B8f, APf, NS4Bf] 9.9546; g1mix11 line B P6A [B8f, AP6f] 9.9558.
  'no': g1mix10 Q8A-line P3B [B8f, APf, NS3Bf] k27 9.9421 (train +0.000 2/5, rms 0.386), PALL 9.9443 (+0.001 2/5, rms 0.329).
- Staged, NOT handed over: hardness-cache submissions/blend_v39q3a6.csv (md5 dfa00a7887b2d62f2473be61fe6d3072; OOF
  blend_v39q3a6_oof.csv 6b3a3cdd4c29c51010469cefb652a025). Workflow verify-q3a6-1018 (run wf_5b8a1dda-048) started 10:00 KST.
  If it passes: hand over as 10-18 file 4 (SAFE; does not count against the BOLD cap), add v39q3a6 to the skip list with a new env.
- 10-18: 3 of 5 slots used (v39p7b NEAR-TIE 1 of 2, v39q8ab and v39q8ab3 BOLD 2 of 2). AP6Bf (27-5, 27-6) completes about 11:15 KST.

## 2026-10-18 09:13 KST: AP6f (B6 AdvProp, A3 recipe) complete; g1mix11 gives no file; Q8AB3 stays the file to submit
- 27-2 b6ap_locK_real5_s1 (laptop 07:37:40-08:20:06 KST, 2547 s) and 27-3 _s2 (08:20:06-09:03:58, 2632 s): batch 6, rc 0, no OOM;
  chunk and parquet sums match; parquet md5 (train / test) _s1 4b41d5ac / 3df45b41, _s2 41744213 / 4d219d56; appended to the tar
  (122 parquet members, re-extracted md5s equal); gate corr 0.993 / 0.994.
- Official g1mix11 builds 09:10 KST (no ALLOW_PARTIAL): hardness-cache scripts/d1018/g1mix11/q8a_ap6 (env run_env_1018k26.sh, the
  rule-4 re-judge) and q8a_ap6q (run_env_1018k26q.sh): same numbers. Line A P6A [B8f, APf, AP6f] -0.020 3/5, 40-split P 0.35,
  drop-top-3 +0.131, k26 9.9538, G1 rms vs Q8A 0.277; line B P6A [B8f, AP6f] -0.089 3/5, P 0.38, drop-top-3 +0.072, k26 9.9556,
  rms 0.362. Both lenient NEAR-TIE only (strict needs drop-top-3 < 0; BOLD needs rms >= 0.5): no file. Record q8a_ap6/README.txt.
- Q8AB3 (blend_v39q8ab3.csv, handed over 07:56 KST) stays the only BOLD on hand; told JINWOO 09:13 KST to submit it as is and
  withdrew the 07:56 card. Asked once more for "섹션28 실행" in the main thread (section 27 ends about 14:00 KST).
- 27-4 b6ap_locK_fsim (AP6Bf, B1 recipe) started 09:03:58 KST; then _s1, _s2, AP4f, AP4Bf. Rebuild g1mix11 as each family completes.

## 2026-10-18 07:57 KST: Q8AB3 verified and handed over as 10-18 file 3 (BOLD 2 of 2)
- Workflow verify-q8ab3-1018 (run wf_ab3b1165-51c, 07:29-07:53 KST): critic DELIVER; rules (own rebuild, kriging26, all 21 rms, judge
  agreement), file (bit-identical rebuild, independent numpy re-implementation, test-noise invariance), NS3Bf reader (chunk/parquet/tar
  md5 chain, seed sanity, Apache-2.0 tf_efficientnet_b3.ns_jft_in1k) and public-model lens all PASS_WITH_NOTES. Notes: k26 9.96228 is
  0.0005 under the bar; leave-one-out range 9.9604-9.9637; P(beat Q8A) 0.31, bias-corrected 0.23-0.28; kriging puts it about +0.10
  public MSE worse than Q8AB's reading; both earlier "add a Noisy Student family" readings were worse (v38q86, v39p7b).
- Handed over 07:56 KST: hardness-cache submissions/blend_v39q8ab3.csv (md5 c05736735286c14265489100de6f81a3; OOF
  blend_v39q8ab3_oof.csv 12ef9e0bbcb8d7828372c42c7011b136). A card asks JINWOO: submit now (recommended; NOTE_slate1018 rule 4) or wait
  until about 09:40 KST for the AP6f judgment (b6ap_locK_real5 seeds, d1018/g1mix11) and take the better of the two for 10-18's last
  BOLD slot. If AP6f gives a lower-kriging BOLD before the user submits Q8AB3, rule 4 applies either way: re-judge with
  run_env_1018k26.sh (Q8AB3 a candidate) and say which file to submit. Once Q8AB3 is submitted: run_env_1018k26q.sh (22 files).
- 27-1 b6ap_locK_real5 (laptop 06:52:52-07:37:39 KST, 2687 s, batch 6, no OOM) ingested 07:42 KST: sums match (train 85.7199 /
  2759.9410 / 2766.1715; test 171.0501 / 5531.5719 / 5520.7949), parquet md5 train fba8f081, test a407c8ab; appended to the tar after
  the workflow (118 parquet members, re-extracted md5s equal); gate corr 0.993 (scratch g1mix11 build). 27-2 (_s1) started 07:37 KST;
  B6 runs take about 45 min, so AP6f is complete about 09:10 KST and section 27 ends about 14:30 KST.
- 10-18: 2 files submitted (v39p7b, v39q8ab), Q8AB3 handed over; with it 3 of 5 slots, BOLD 2 of 2, NEAR-TIE 1 of 2. Verifier scratch
  copies (test predictions under /tmp/q8ab3verify, /tmp/q8abverify) deleted.

## 2026-10-18 07:31 KST: Q8AB read 9.9572 (worse than Q8A); kriging26; Q8AB3 is the only BOLD on hand, in verification
- blend_v39q8ab.csv (10-18 file 2, BOLD 1 of 2) read 9.9572104905 (user 07:25 KST): +0.1524 public MSE vs Q8A (9.9495571078), which stays
  the best file and the automatic final. kriging25 had predicted 9.9568 (z +0.02). hardness-cache scripts/d1016/pub/forensics26.py and
  kriging26.py (mk_pubN.py 26 v39q8ab=9.9572104905:aab27fb4; read26.py / read26_out.txt: LOO z v39q8ab +0.04, v38q8a -0.54, v38q8 +1.00).
- Env for every later build: hardness-cache scripts/d1018/run_env_1018k26.sh "v38q8a 9.9495571078" (= run_env_1018k25q.sh, 21 files in the
  skip list incl. v39p7b and v39q8ab, with KRIG_LATEST -> kriging26; sha in d1018/NOTES.sha256). Line B: export BASE=B8f after sourcing.
- Pool re-judged under kriging26 (NOTE_slate1018 rule 5): d1017/g1mix10/q8a_k26 and d1017/g1mix9/q8a_k26 official rebuilds (07:28 KST,
  rc 0, Q8A reproduced bit for bit) and the g1mix8 Q8A-rule lines (best_q8a_k24, best_q8a_ns4) re-kriged: every option SKIP, "no", or
  >= 9.9736, except g1mix10 line B PALL = Q8AB3 [B8f, APBf, NS3Bf]: +0.002 3/5, 40-split P 0.62, drop-top-3 +0.321, k26 9.9623 (bar
  9.9628), P(beat Q8A) 0.31, G1 rms vs Q8A 0.709, vs v39q8ab 0.395 -> BOLD, the lowest (only) BOLD on hand.
- Staged, NOT handed over: hardness-cache submissions/blend_v39q8ab3.csv (md5 c05736735286c14265489100de6f81a3) and
  blend_v39q8ab3_oof.csv (12ef9e0bbcb8d7828372c42c7011b136). Workflow verify-q8ab3-1018 (run wf_ab3b1165-51c; rules, file rebuild and
  re-implementation with test-noise invariance, NS3Bf reader health and provenance, public-model lens, critic) started 07:29 KST.
  If it passes: hand over as 10-18 BOLD 2 of 2 (10-18 file 3), add v39q8ab3 to the skip list with a new env, record it here and in
  LEADERBOARD. Do not append section 27 tables to het_cnn-tables.tar until the workflow ends.
- 10-18: 2 files submitted (v39p7b NEAR-TIE 1 of 2, v39q8ab BOLD 1 of 2), 3 slots left. Section 27 (d1018/g1mix11) runs until about
  14:00 KST; its BOLD files go to 10-19 if Q8AB3 takes the last 10-18 BOLD slot (SAFE needs no slot class; NEAR-TIE has 1 left on 10-18).

## 2026-10-18 07:01 KST: section 26 done; NS3Bf round judged; Q8AB stays the file to submit
- 26-5 b3ns_locK_fsim_s1 ingested 06:34 KST (parquet md5 train 835fc036, test 60bfd67c). 26-6 b3ns_locK_fsim_s2 (laptop
  06:29:48-06:52:13 KST, 1345 s, batch 8, no OOM) ingested 06:58 KST: sums match (train 86.6449 / 2759.6569 / 2771.8387; test
  173.0465 / 5532.5853 / 5530.0263), parquet md5 train 87723f5e, test b837e482; tar 116 parquet members, re-extracted md5s equal.
  Builder gate corr 0.991 / 0.991 / 0.990 for the three NS3Bf seeds. Section 26 is complete; section 27 (AdvProp B6/B4, d1018/g1mix11)
  started on its own 06:52:52 KST (27-1 b6ap_locK_real5, batch 6), expected to end about 14:00 KST 10-18.
- Official g1mix10 builds with complete NS3Bf: hardness-cache scripts/d1017/g1mix10/q8a_ns3b3/ (env d1018/run_env_1018k25.sh, Q8AB a
  candidate; the NOTE_slate1018 rule-4 re-judge, the user had not submitted v39q8ab) and its twin q8a_ns3b3q/ (run_env_1018k25q.sh,
  Q8AB in the skip list); same files in both. judge_slots1018.py: line A PAB SKIP, PALL [B8f, APf, APBf, NS3Bf] k25 9.9539 and P3B
  [B8f, APf, NS3Bf] 9.9550 both "no" (train +0.004 / +0.003 vs Q8A, G1 rms vs Q8A 0.43-0.44 < 0.5). Line B: Q8AB 9.9568 BOLD (lowest),
  PALL [B8f, APBf, NS3Bf] +0.002 3/5, 40-split P 0.62, drop-top-3 +0.321, k25 9.9619, P(beat Q8A) 0.33, G1 rms vs Q8A 0.709, vs
  v39q8ab 0.395 -> BOLD; P3B [B8f, NS3Bf] 9.9699 no.
- Rule-4 result: no change; the user should submit blend_v39q8ab.csv as handed over. Line B PALL ("Q8AB3"; test md5
  c05736735286c14265489100de6f81a3, OOF 12ef9e0bbcb8d7828372c42c7011b136) is the 10-18 BOLD 2 candidate, NOT staged: it shares B8f +
  APBf with Q8AB, so wait for the Q8AB reading, then mk_pubN.py 26 -> kriging26, a new env, re-judge this round and the pool
  (rule 5), and verify before any delivery.

## 2026-10-18 06:18 KST: Q8AB verified and handed over as 10-18 file 2 (BOLD 1 of 2)
- Workflow verify-q8ab-1018 (run wf_544b9be0-36f, 05:52-06:16 KST): critic DELIVER; rules, file and reader verifiers and the
  public-model lens all PASS_WITH_NOTES (two bit-identical rebuilds, independent numpy re-implementation matching Q8AB and the Q8A
  control, test-noise invariance, APBf chunk/parquet/tar integrity, Apache-2.0 weights). Public-model notes: kriging25 9.9568,
  P(beat Q8A) 0.35 is the most generous variant (others 0.25-0.30; B8f-line bias-adjusted about 0.2-0.3); (Q8AB - Q8A) correlates
  +0.787 with (v38q8 - Q8A) on the 433 G1 rows, so it moves about 62% of the way back toward B8f alone.
- Handed over 06:18 KST: hardness-cache submissions/blend_v39q8ab.csv (md5 aab27fb4fb477bcf60874a8dba5c4c79; OOF
  blend_v39q8ab_oof.csv 121c1309785f1f769f231b3871485fa4). 10-18: 2 files (v39p7b NEAR-TIE, v39q8ab BOLD), 3 slots left, BOLD 1 of 2,
  NEAR-TIE 1 of 2. Public reading pending; then mk_pubN.py 26 -> kriging26, a new env, re-judge.
- NOTE_slate1018 rule 4: if NS3Bf (26-5, 26-6) finishes before the user submits v39q8ab, re-judge g1mix10 with v39q8ab as a candidate
  (env run_env_1018k25.sh, v39q8ab not in the skip list) and say which file to submit. Once it is submitted, builds use
  hardness-cache scripts/d1018/run_env_1018k25q.sh (v39q8ab in the skip list, 21 files).
- 26-4 b3ns_locK_fsim (laptop 05:43:46-06:06:56 KST, 1390 s, batch 8, no OOM) ingested 06:11 KST: sums match (train 87.6585 /
  2752.6267 / 2769.6652; test 174.6839 / 5516.8253 / 5527.7741), parquet md5 train 6cf5bbf1, test fea32d79; appended to the tar after
  the workflow ended (112 parquet members, re-extracted md5s equal); gate corr 0.991. 26-5 (_s1) started 06:06 KST.

## 2026-10-18 05:52 KST: APBf complete; line B PAB = Q8AB = [B8f, APBf] is BOLD; staged, in verification
- 26-3 b5ap_locK_fsim_s2 (laptop 05:11:15-05:43:46 KST, 1951 s, batch 8, no OOM) ingested 05:48 KST: sums match (train 85.6749 /
  2753.9599 / 2782.1119; test 171.2053 / 5522.4649 / 5550.3109), parquet md5 train 5000cd15, test 99a7c3e7; tar 110 parquet members,
  re-extracted md5s equal; builder gate corr 0.990 / 0.991 / 0.990 for the three APBf seeds. 26-4 (b3ns_locK_fsim) started 05:43 KST.
- Official g1mix10 build (hardness-cache scripts/d1017/g1mix10/q8a_apb3/, README.txt; env d1018/run_env_1018k25.sh, kriging25):
  line A PAB [B8f, APf, APBf] +0.007 1/5 k25 9.9504, G1 rms vs Q8A 0.237 -> SKIP. Line B PAB = Q8AB [B8f, APBf] +0.002 3/5,
  40-split P 0.80, drop-top-3 +0.306, k25 9.9568, P(beat Q8A) 0.35, G1 rms vs Q8A 0.571 (min over the 20 files 0.445) -> BOLD.
  PALL = PAB in both lines (NS3Bf incomplete).
- Staged, NOT handed over: hardness-cache submissions/blend_v39q8ab.csv (md5 aab27fb4fb477bcf60874a8dba5c4c79) and
  blend_v39q8ab_oof.csv (121c1309785f1f769f231b3871485fa4). Workflow verify-q8ab-1018 (run wf_544b9be0-36f; rules, file rebuild
  and re-implementation, APBf reader health, public-model lens, critic) started 05:52 KST. If it passes: hand over as 10-18 BOLD #1
  (10-18 file 2), add v39q8ab to the skip list with a new env, and record it here and in LEADERBOARD.
- Then NS3Bf (26-4..26-6) -> P3B and PALL into a new g1mix10 OUT dir. Do not append 26-4 tables to het_cnn-tables.tar until the
  verify workflow ends (its reader check extracts from the tar).

## 2026-10-18 05:30 KST: P7B read 9.9576 (worse than Q8A); kriging25; g1mix9 re-judged, no file
- blend_v39p7b.csv (10-18 file 1) read 9.9575524935 (user 05:24 KST): +0.1592 public MSE vs Q8A (9.9495571078), which stays the best
  file and the automatic final. kriging24 had predicted 9.9572 (z +0.02). hardness-cache scripts/d1016/pub/forensics25.py and
  kriging25.py (mk_pubN.py 25 v39p7b=9.9575524935:e226678a; read25_out.txt: LOO z v39p7b +0.10, v38q8a -0.64).
- Env for every later build: hardness-cache scripts/d1018/run_env_1018k25.sh "v38q8a 9.9495571078" (= run_env_1018p.sh with
  KRIG_LATEST -> kriging25; sha in d1018/NOTES.sha256). Line B: export BASE=B8f after sourcing.
- g1mix9 rebuilt under kriging25 (hardness-cache scripts/d1017/g1mix9/q8a_k25/, README.txt; 05:28 KST, rc 0, Q8A reproduced bit for
  bit): line A P7B SKIP (= v39p7b; k25 9.9585), P4B +0.163 3/5 k25 9.9597 no, PALL SKIP (G1 rms 0.245 vs v39p7b); line B P7B k25
  9.9748, P4B 9.9757, PALL 9.9767, all above the 9.9628 BOLD bar. No SAFE, BOLD or NEAR-TIE(strict) file on hand.
- 10-18 slots: 1 used (v39p7b, NEAR-TIE 1 of 2), 4 left; BOLD 0 of 2 used. Next: 26-3 b5ap_locK_fsim_s2 (started 05:11 KST) completes
  APBf -> d1017/g1mix10 PAB (line A [B8f, APf, APBf], line B [B8f, APBf]) with run_env_1018k25.sh and judge_slots1018.py; then
  NS3Bf (26-4..6) -> P3B and PALL. Section 27 follows automatically; section 28 still needs "섹션28 실행".

## 2026-10-18 05:21 KST: JINWOO picked the NEAR-TIE class (strict, 2 per day); P7B handed over as 10-18 file 1
- 05:17:56 KST JINWOO tapped "엄격 조건, 하루 2개" on the 03:19 KST card; 05:18:00 and 05:18:03 KST also "하루 2개까지" on the two replaced
  cards (01:12, 02:13). Recorded in hardness-cache scripts/d1018/DECISION_neartie1018.txt (sha in d1018/NOTES.sha256): the STRICT
  variant (judge label NEAR-TIE(strict) only), at most 2 NEAR-TIE files per KST day on 10-18 and 10-19, outside the 2-BOLD cap,
  order SAFE > BOLD > NEAR-TIE, handed over only while no SAFE/BOLD file is on hand or in verification. The thread told JINWOO it
  follows the 03:19 card (strict) despite the two older taps.
- Re-judged 05:19 KST (q8a_ns4b3 builds): P7B line A NEAR-TIE(strict), no SAFE/BOLD on hand. Handed over 05:20 KST as 10-18 file 1:
  hardness-cache submissions/blend_v39p7b.csv (md5 e226678a487e1d2903eea3493d5144af; 433 G1 rows differ from Q8A, rms 0.342).
  10-18 NEAR-TIE count 1 of 2. Public reading pending; when it comes: mk_pubN.py 25 -> pub/kriging25.py, new env, re-judge.
- New env for every later build: hardness-cache scripts/d1018/run_env_1018p.sh "v38q8a 9.9495571078" (= run_env_1018.sh plus
  v39p7b in SUBMITTED_EXTRA, so the skip rule covers P7B; scratch g1mix9 build 05:20 KST: P7B and PALL now SKIP(rms<0.25)).
- JINWOO wrote "재개" 05:18 KST; the laptop never stopped (section 26 running, 26-3 started 05:11; section 27 follows; section 28
  still needs "섹션28 실행" in the main thread, asked again in the 05:20 reply).

## 2026-10-18 04:15 KST: section 25 complete; NS4Bf adds no file; P7B is still the only candidate
- 25-6 b4ns_locK_fsim_s2 (laptop 03:40:06-04:07:31 KST, 1645 s, batch 8, no OOM) ingested 04:12 KST: sums match (train 85.8068 /
  2769.5434 / 2771.1833; test 171.2466 / 5552.5924 / 5529.8110), parquet md5 train 1bf4354c, test ea9bbf8d; tar 104 parquet members,
  re-extracted md5s equal; builder gate corr 0.990 / 0.990 / 0.991 for the three B4-NS seeds. Section 26 started 04:07:49 KST (26-1
  b5ap_locK_fsim).
- Official g1mix9 build with NS4Bf complete (hardness-cache scripts/d1017/g1mix9/q8a_ns4b3/, README.txt; 04:12:59-04:13:23 KST):
  NS4Bf alone vs A3fixed -0.373 2/5 (weak). Line A P4B [B8f, APf, NS4Bf] +0.163 3/5 k24 9.9596 -> no; PALL [B8f, APf, NS7Bf, NS4Bf]
  -0.046 2/5 k24 9.9625 -> no. Line B P4B +0.280, PALL -0.026 2/5 k24 9.9753 -> no. P7B line A rebuilt bit for bit (test e226678a...,
  OOF 91594c22... = the staged blend_v39p7b files), still NEAR-TIE(strict). No SAFE or BOLD file from g1mix9.
- 26-1 b5ap_locK_fsim (laptop 04:07:49-04:40:00 KST, 1931 s, batch 8, no OOM) ingested 04:45 KST: sums match (train 85.5889 /
  2758.5403 / 2781.6122; test 170.8865 / 5529.3551 / 5549.9713), parquet md5 train 516942b9, test 9aa72a2c; tar 106 parquet members,
  re-extracted md5s equal; label-free gate (builder formula, no option built) corr 0.990. 26-2 (_s1) started 04:40 KST.
- 26-2 b5ap_locK_fsim_s1 (laptop 04:40:00-05:11:15 KST, 1875 s, batch 8, no OOM) ingested 05:17 KST: sums match (train 85.7465 /
  2768.6946 / 2767.6127; test 171.3864 / 5552.3452 / 5520.6430), parquet md5 train e65eda3e, test c60e0912; tar 108 parquet members,
  re-extracted md5s equal; gate corr 0.991. 26-3 (_s2) started 05:11 KST.
- Next: section 26 (APBf = b5ap_locK_fsim x3, NS3Bf = b3ns_locK_fsim x3) -> d1017/g1mix10 official builds and d1018/judge_slots1018.py;
  the card pick (cmsg_013RLctANMoWDtvGq8ZCRkU9T3Bzc36iGx12DzShoDwsck) still decides whether blend_v39p7b.csv goes out.

## 2026-10-18 03:20 KST: NS7Bf complete; P7B (line A) = NEAR-TIE(strict), verified, staged as blend_v39p7b.csv pending JINWOO's card pick
- 25-3 b7ns_locK_fsim_s2 (laptop 01:50:15-02:43:49 KST, 3214 s, batch 6, no OOM) ingested 02:48 KST: sums match (train 87.0799 /
  2751.5090 / 2780.0921; test 174.0026 / 5519.2321 / 5544.7508), parquet md5 train ce33e9a8, test fcaeaf61; gate corr 0.988.
- Official g1mix9 build (hardness-cache scripts/d1017/g1mix9/q8a_ns7b3/, README.txt; env d1017/run_env_1018.sh, kriging24):
  NS7Bf alone vs A3fixed -1.563 3/5 (now the strongest single family on train, above B7f's -1.530). Line A P7B [B8f, APf, NS7Bf] vs
  Q8A -0.218 3/5, 40-split mean -0.165 P 0.03, drop-top-3 -0.008 (drop-top-4 +0.051), k24 9.9572, P(beat Q8A) 0.35, G1 rms vs Q8A
  0.342 -> builder "no" (not SAFE, not BOLD); d1018/judge_slots1018.py -> NEAR-TIE(strict). Line B P7B [B8f, NS7Bf] -0.278 3/5,
  k24 9.9729 -> no. P4B waits for NS4Bf (25-4 in, 25-5/25-6 running); PALL = P7B until then.
- Workflow verify-p7b-1018 (02:53-03:18 KST; q8a_ns7b3/verify_p7b_output.json): all three verifiers PASS_WITH_NOTES, critic
  READY_ON_PICK. Staged (NOT handed over): hardness-cache submissions/blend_v39p7b.csv (md5 e226678a487e1d2903eea3493d5144af),
  blend_v39p7b_oof.csv (91594c22f5da6140ec261c3f5e1fd2de).
- Card: the 02:13 KST card (2-seed text, now wrong for the strict option) was withdrawn 03:19 KST and replaced by a card with the
  3-seed numbers: 엄격 조건, 하루 2개 (recommended) / 엄격 조건, 하루 1개 / 기본 조건, 하루 2개 / 지금 규칙 유지. Any of the first three
  releases blend_v39p7b.csv; record the pick in d1018/DECISION_neartie1018.txt (cite NOTE 7d58bf21..., DISCLOSURE 81ca1e86...).
  Before handing over, re-judge: a SAFE or BOLD file that is on hand first takes precedence.
- 25-4 b4ns_locK_fsim (laptop 02:43:49-03:12:39 KST, 1730 s, batch 8, no OOM) ingested 03:19 KST: sums match (train 85.6206 /
  2766.4451 / 2769.2897; test 171.5800 / 5546.7858 / 5527.3833), parquet md5 train 49694ad0, test 5310c05d; tar 100 parquet members.
  25-5 (_s1) started 03:12 KST.
- 25-5 b4ns_locK_fsim_s1 (laptop 03:12:39-03:40:06 KST, 1647 s, batch 8, no OOM) ingested 03:46 KST: sums match (train 84.9278 /
  2787.1792 / 2761.2927; test 169.8805 / 5589.8487 / 5508.1261), parquet md5 train 7c808dd0, test 1307d6f2; tar 102 parquet members,
  re-extracted md5s equal. Label-free gate (builder formula, no option built) corr 0.990 for seed 0 and seed 1. 25-6 (_s2) started
  03:40 KST; when it is in, run the official g1mix9 build (new OUT dirs) for P4B and PALL, then judge_slots1018.py.

## 2026-10-18 02:14 KST: 25-2 in (2-seed read only); NEAR-TIE card replaced with a disclosed one; section 28 stop times tightened
- 25-2 b7ns_locK_fsim_s1 (laptop 00:55:07-01:50:15 KST, 3308 s, batch 6, no OOM) ingested 01:57 KST: sums match (train 87.1978 /
  2776.2415 / 2763.0529; test 174.6217 / 5567.8024 / 5511.4991), parquet md5 train 7a35fbb8, test d26e138d; tar 96 parquet members;
  gate corr 0.987. 25-3 (_s2) started 01:50 KST.
- Two-seed diagnostic (d1017/g1mix9/early_s0/README.txt, build_line{A,B}_s01.log; never submitted): NS7Bf alone vs A3fixed -1.477
  3/5; line A P7B [B8f, APf, NS7Bf] vs Q8A -0.193 3/5, 40-split mean -0.151 P 0.03, drop-top-3 +0.023, k24 9.9583, P(beat Q8A) 0.33,
  G1 rms 0.343 -> no under the g1mix9 bars; line B P7B [B8f, NS7Bf] -0.237 3/5, k24 9.9746 -> no.
- Independent check workflow verify-section28-g1mix12 (critic 01:58 KST): section 28 and g1mix12 fine to run; the NEAR-TIE note
  needed a disclosure. Written: d1018/DISCLOSURE_neartie1018.txt (the class came 10 min after a diagnostic that meets it; errata
  (a)-(e); clarifications; the stricter variant = also 40-split mean < 0 and drop-top-3 < 0, the 10-11 SAFE addendum A, which both
  P7B diagnostics fail). NOTE_neartie1018.txt unchanged; both sha256 in d1018/NOTES.sha256 (note 7d58bf21..., disclosure 81ca1e86...).
- The 01:12 KST card was withdrawn at 02:13 KST and replaced (02:13 KST) by a card with the disclosure and 4 options (하루 2개까지 recommended,
  하루 1개까지, 엄격 조건만, 지금 규칙 유지). The pick goes to d1018/DECISION_neartie1018.txt citing both shas.
- Wording erratum d1018/g1mix12/ERRATUM.txt: NS7Bf is the strongest Noisy Student or AdvProp reader, not the strongest single reader
  (effnetv2-l B7f -1.530 alone). Section 28 stop rule in notes/handoff/laptop-gpu.md changed before any section 28 run: no B7 run
  starts after 03:50 KST 10-19, no B8 run after 03:30 KST, so every run ends before g1mix12's 05:00 KST fallback cut-off.

## 2026-10-18 01:11 KST: 25-1 in (1-seed read only); section 28 (AdvProp B7/B8) prepared and pre-registered; NEAR-TIE class proposed
- 25-1 b7ns_locK_fsim (laptop 00:00:51-00:55:07 KST, 3256 s, batch 6, no OOM) ingested 01:00 KST: sums match (train 87.2829 /
  2746.9357 / 2782.2409; test 174.3948 / 5510.2022 / 5549.5627), parquet md5 train 299d1bc9, test 7563a75d; tar 94 parquet members;
  gate corr 0.987. 25-2 (_s1) started 00:55 KST.
- Seed-0 diagnostic only (hardness-cache scripts/d1017/g1mix9/early_s0/, ALLOW_PARTIAL, never submitted): NS7Bf alone vs A3fixed
  -1.292 (strongest single family so far); line A P7B [B8f, APf, NS7Bf] vs Q8A -0.128 3/5, P 0.17, k24 9.9627, G1 rms 0.344 -> no
  (too close for BOLD, kriging above the best reading so not SAFE); line B P7B [B8f, NS7Bf] -0.145 3/5, k24 9.9807 -> no.
- Section 28 written to notes/handoff/laptop-gpu.md (01:10 KST): the B1 and the A3 recipe on tf_efficientnet_b7.ap_in1k (batch 6)
  and tf_efficientnet_b8.ap_in1k (batch 4), 3 seeds each (28-1..28-12, about 13.5 h after section 27; no run starts after 04:00
  KST 10-19). Pre-registered before any run: hardness-cache scripts/d1018/g1mix12/options.txt (sha256 5f4f6131...), mk_g1mix12.py,
  build_g1mix12.py (hashes in g1mix12/options.sha256; dry run 01:09 KST reproduces blend_v38q8a bit for bit). Weights smoke-built
  in the cloud 01:06 KST (B7-AP 62.4M, B8-AP 82.9M with the fpn head; saved activations B8/B7 1.25). Needs "섹션28 실행".
- Proposed NEAR-TIE slot class (hardness-cache scripts/d1018/NOTE_neartie1018.txt, sha256 in d1018/NOTES.sha256; not in force
  until JINWOO picks it on the 10-18 card): line A files that are better than the best file on train (SAFE train conditions)
  with kriging <= best + 0.0132 but too close to be BOLD may fill spare slots, 1 or 2 per day, after SAFE and BOLD.

## 2026-10-18 00:15 KST: NS4f complete (no file); section 27 (AdvProp B6/B4) prepared and pre-registered - read this first
- 24-12 b4ns_locK_real5_s2 (laptop 23:34:37-00:00:25 KST, 1548 s, batch 8) ingested 00:07 KST: sums match (train 84.3426 / 2779.6230 /
  2759.8092; test 167.8805 / 5576.4292 / 5502.0067); tar 92 parquet members. Section 24 complete. NS4f gate corr 0.994/0.995/0.994.
- g1mix8 line vs the Q8A rule (g1mix8/best_q8a_ns4, build_0008.log, judge8_0008.log; env d1017/run_env_1018.sh, BASE=B8f): Q84 = [B8f,
  NS4f] +0.356 3/5, k24 9.9748 -> no; Q8ALL = [B8f, NS6f, APf, NS6Bf, NS4f] +0.261 2/5, k24 9.9763 -> no. NS4f alone vs A3fixed only
  -0.108 (the A3 recipe on B4-NS is weak, like on B6-NS). No 10-18 file from section 24.
- Section 25 started on the laptop 00:00:51 KST (25-1 b7ns_locK_fsim, batch 6, first epoch 59 s vs 24-7's 55 s, no OOM). At the
  observed speeds sections 25 and 26 should end about 06:00-06:30 KST 10-18 (the plan's 09:15 was conservative). Laptop estimate
  at 00:20 KST: section 25 ends about 04:30, section 26 about 07:10, section 27 about 14:00 KST 10-18.
- Section 27 written to notes/handoff/laptop-gpu.md (00:14 KST): A3 and B1 recipes on tf_efficientnet_b6.ap_in1k and
  tf_efficientnet_b4.ap_in1k, 3 seeds each (27-1..27-12, about 7 h; no run starts after 04:00 KST 10-19). Pre-registered before any
  run: hardness-cache scripts/d1018/g1mix11/options.txt (sha256 bb568132...), mk_g1mix11.py, build_g1mix11.py (hashes in
  g1mix11/options.sha256; dry run 00:14 KST reproduces blend_v38q8a bit for bit). Both weights smoke-built in the cloud 00:12 KST
  (B6-AP 39.7M, B4-AP 17.0M with the fpn head). JINWOO wrote "섹션 27 실행" (main thread 00:18 KST 10-18); the laptop queued it
  00:20 KST after sections 25 and 26 (C: 257 GB free, above the 10 GB bar).

## 2026-10-17 23:45 KST: Q8A read 9.9495571078 = NEW BEST; kriging24; everything re-based on Q8A - read this first
- blend_v38q8a (Q8A = [B8f, APf], 10-17 extra #2) public 9.9495571078 (user, main thread 23:42 KST): -0.1378 public MSE vs XN8; NEW BEST
  and automatic final while best. kriging23 had predicted 9.9690 (z -0.68), kriging22 9.9644 (z -0.50). 10-17 is closed (4 files).
- pub/forensics24.py + kriging24.py (mk_pubN.py 24; BUILT_ON v30b); pub/read24.py -> read24_out.txt (LOO z v38xn8 -1.85, v38q8a -0.92,
  v35xn +1.37, v38q8 +1.34, v38q86 +1.13, rest within +-0.80).
- New env hardness-cache scripts/d1017/run_env_1018.sh (sha256 fd2248e8...): KRIG_LATEST kriging24, SUBMITTED_EXTRA + v38q86, v38q8all,
  v38q8a, best case v38q8a (BASE = B8f,APf; BEST_RUNS = B8 3 seeds + AP 3 seeds; reproduces blend_v38q8a bit for bit). Usage:
  `source .../d1017/run_env_1018.sh v38q8a 9.9495571078` (then BASE=B8f for the g1mix8 line). Bars vs the Q8A rule: BOLD kriging bar
  9.9628, SAFE kriging < 9.9495571078.
- Re-based g1mix8 line (g1mix8/best_q8a_k24, build_2344.log, judge8_2344.log): Q86B +0.073 3/5, k24 9.9795 -> no; Q8ALL+AP +0.147 2/5,
  k24 9.9734 -> no; Q8, Q86, Q8A SKIP (submitted). g1mix7 line A is not built while the best holds APf (NOTE_slate1018 item 2a); its
  XN8-base options all sit < 0.25 from XN8. So no 10-18 file on hand yet.
- B4-NS: 24-10 (seed 0) and 24-11 (b4ns_locK_real5_s1, laptop 23:07:50-23:34:37 KST, 1607 s, batch 8) ingested, sums match (s1 train
  84.8252 / 2790.6527 / 2748.7404; test 168.9568 / 5594.2969 / 5484.3804); tar 90 parquet members. 24-12 (_s2) started 23:34 KST; section
  25 (25-1 b7ns_locK_fsim) starts after it on the laptop.
- Next: NS4f complete -> rebuild the g1mix8 line (Q84, Q8ALL) with run_env_1018.sh; sections 25/26 -> g1mix9/g1mix10 lines A (BASE =
  [B8f, APf]) and B ([B8f]); pick per d1017/NOTE_slate1018.txt (10-18 and 10-19 morning, 2 BOLD per day).

## 2026-10-17 23:24 KST: extra #2 = Q8A (blend_v38q8a.csv) handed over 23:20 KST; 10-17 done (4 of 5 files) - read this first
- Workflow verify-q8a-1017 (rules re-derived under kriging23, file rebuilt and checked, critic with its own rebuild): rules_k23 and
  file_q8a PASS_WITH_NOTES, critic DELIVER (hardness-cache scripts/d1016/g1mix8/verify_q8a_output.json; APf reader health and
  compliance from verify_q8allap_partial.json). Record g1mix8/DELIVERY_extra2_q8a.txt.
- Q8A = [B8f, APf] under the XN8 rule: OOF -0.312 MSE 2/5 folds (40-split P 0.125, drop-top-3 +0.110), kriging23 9.9690 (bar 9.9697,
  P(beat XN8) 0.33), G1 rms vs XN8 0.638, min over the 18 submitted/handed-over files 0.466 (v37n2) -> BOLD. md5 78a31699...,
  OOF aca4900f.... The reply said the 21:42 KST APf-seed-0 scratch preview existed (before the extra rules; not used for the pick).
- 10-17 slots: v38q8, v38xn8, v38q86, v38q8a; the 5th stays empty (ADDENDUM_extra1017: at most 2 extras). 10-18: 5 slots, 2-bold cap.
- 24-10 b4ns_locK_real5 (NS4f seed 0) ingested 23:23 KST: sums match the dump (train 84.5885 / 2774.2729 / 2763.8907; test 168.1159 /
  5563.7034 / 5513.8197; no NaN); tables train 9883efe4..., test 40bfc798...; het_cnn-tables.tar 88 parquet members (re-extracted,
  md5 equal). 24-11 (_s1) running since 23:07 KST, then 24-12 (_s2).
- Next: when the user reports Q8A, mk_pubN.py 24 + read24, new env with kriging24 and v38q8a in SUBMITTED_EXTRA; when NS4f has 3
  seeds, rebuild g1mix7/g1mix8 with NS4f (BEST = the best file then) and judge for 10-18; sections 25 and 26 overnight (g1mix9,
  g1mix10). Skip rule now covers 19 files (16 submitted + v38q86, v38q8all, v38q8a).
- Slot plan pinned 23:29 KST before the Q8A reading and before NS4f seeds 1-2 / sections 25-26: hardness-cache scripts/d1017/
  NOTE_slate1018.txt (sha256 f220d50e..., in d1017/NOTES.sha256): 10-18 has 5 slots and 10-19 has 5 more until the 10:00 KST deadline
  (earlier records missed 10-19); at most 2 BOLD per day; pool = NS4f rebuilds + g1mix9 + g1mix10 + undelivered 10-17 options; SAFE
  first, then strictly lowest latest kriging; a reading or round arriving before the user submits is applied first.

## 2026-10-17 23:10 KST: Q86 read 9.9809869429 (worse than XN8); kriging23; extra #2 re-picked = Q8A (B8f + APf) - read this first
- blend_v38q86 (extra #1) public 9.9809869429 (user, main thread 23:05 KST): +0.489 public MSE vs XN8; kriging22 had predicted
  9.9618 (z +0.66). XN8 stays the best file. Added as pub/forensics23.py + kriging23.py (mk_pubN.py 23; BUILT_ON v30b); pub/read23.py
  -> read23_out.txt (LOO z v38xn8 -1.78, v35xn +1.27, v38q8 +1.21, v38q86 +0.94, rest within +-0.73).
- 24-6 b5ap_locK_real5_s2 ingested 22:46 KST (sums match the dump; tar 86 parquet members). APf complete -> rebuilds g1mix8/best_xn8_ap
  and g1mix7/best_xn8_ap (env g1mix8/run_env_extra2.sh); pick under kriging22 (g1mix8/PICK_extra2.txt) was Q8ALL+AP = [B8f, NS6f, APf,
  NS6Bf] 9.9640 (blend_v38q8allap.csv); its verification (verify-q8allap-1017) was stopped at the critic when the Q86 reading came;
  its three verifier reports (all PASS_WITH_NOTES) are in g1mix8/verify_q8allap_partial.json.
- Re-judge under kriging23 (env g1mix8/run_env_extra2_k23.sh; builds best_xn8_ap_k23, files identical; judge8_2306.log; record
  g1mix8/PICK_extra2_k23.txt): Q8A = [B8f, APf] vs XN8 -0.312 2/5 (40-split P 0.125), k23 9.9690 (P(beat XN8) 0.33), rms vs XN8
  0.638, min 0.466 -> BOLD (the only one); Q86B k23 9.9802 and Q8ALL+AP 9.9832 fail the kriging bar 9.9697; g1mix7 all SKIP or no.
  Extra #2 = submissions/blend_v38q8a.csv (md5 78a31699cf40a8dcfdb5a29b0483afb2, OOF aca4900f...); verify-q8a-1017 DELIVER (above).
- Disclosure (PICK_extra2_k23.txt): a 21:42 KST scratch preview with APf seed 0 only (before the extra rules existed; not an option).
- Laptop: 24-10 b4ns_locK_real5 (NS4f seed 0) started 22:41 KST; chunk 0 in (sums 84.5885 / 2774.2729 / 2763.8907).

## 2026-10-17 22:38 KST: extra #1 = Q86 (blend_v38q86.csv) handed over; extra #2 waits on APf seed 2 - read this first
- Workflow verify-q86-1017 (rebuild, compliance perturbation, rules re-derivation, critic with its own numpy/pandas rebuild):
  critic DELIVER_Q86 (output hardness-cache scripts/d1016/g1mix8/verify_q86_output.json). Q86 = [B8f, NS6f]: vs the XN8 rule -0.090
  MSE 3/5 (40-split P 0.60, drop-top-3 +0.248), kriging22 9.9618 (P(beat XN8) 0.427), G1 rms vs XN8 0.618, min over the 16
  submitted files 0.488 (v37n2) -> BOLD, not SAFE. Switching from the card's Q8ALL follows the pinned rules (reading before
  delivery; ADDENDUM_q8 rule 1, g1mix6/ADDENDUM_delivery rule 2). verify-q8all-1017: file sound, critic DELIVER_IF_USER_AGREES
  (verify_q8all_output.json).
- 22:38 KST reply: submit blend_v38q86.csv (10-17 file 3); said plainly the card named Q8ALL and the XN8 reading moved Q86 first
  (9.9618 vs 9.9635, 43% vs 42%); offered blend_v38q8all.csv as the alternative, only one of the two (G1 rms between them 0.255).
  Record g1mix8/DELIVERY_extra1_q86.txt.
- Extra #2 set fixed before APf seed 2 exists: g1mix8/NOTE_extra2_set.txt (22:18 KST, sha256 21d4b378..., pinned in options.sha256):
  rebuild into g1mix8/best_xn8_ap and g1mix7/best_xn8_ap; candidates = their BOLD options minus extra #1 (the 21:10 Q8ALL is
  superseded by the rebuilt Q8ALL with APf); XN8-rule bars; strictly lowest kriging22. Env g1mix8/run_env_extra2.sh = run_env_1017.sh
  + v38q86 and v38q8all in SUBMITTED_EXTRA (both count as handed over).
- 24-5 b5ap_locK_real5_s1 (laptop 21:36:38-22:08:38 KST, 1920 s, batch 8) ingested 22:15 KST: sums match the dump (train 86.0886 /
  2772.0106 / 2757.9995; test 171.8311 / 5559.0782 / 5502.5814); het_cnn-tables.tar 84 parquet members (re-extracted, md5 equal).
  24-6 (_s2) started 22:08 KST.

## 2026-10-17 21:53 KST: XN8 read 9.9564798521 = NEW BEST; kriging22; re-based on XN8; extra bold files tonight - read this first
- blend_v38xn8 (XN8 = B1f + B4f + NSf + B8f, 10-17 bold #2) public 9.9564798521 (user, main thread 21:49 KST): -0.0594 public MSE vs
  N2; NEW BEST and automatic final while best. kriging21 had predicted 9.9632 (z -0.98), kriging18 9.9670 (z -0.67).
- 21:44 KST card "오늘 남은 3칸에 과감 파일을 더 낼까요?"; the user picked "더 낸다" at 21:47 KST: hardness-cache
  scripts/d1016/g1mix8/ADDENDUM_extra1017.txt (sha256 be1502fd..., pinned in g1mix8/options.sha256) is in effect (record
  g1mix8/DECISION_extra1017.txt): up to 2 extra bold files on 10-17 only (the 2-bold/day cap is a thread rule; SAFE is out of reach
  near the best because the best's own kriging is above its reading). 10-18 keeps the cap unless the user says otherwise.
- pub/forensics22.py + kriging22.py (mk_pubN.py 22; BUILT_ON v30b) add the reading; pub/read22.py -> read22_out.txt (LOO z v38xn8
  -1.80, v38q8 +1.44, v35xn +1.34, rest within +-0.71). New env g1mix7/run_env_1017.sh (KRIG_LATEST kriging22, v38xn8 in
  SUBMITTED_EXTRA, best case v38xn8 with the 24-entry XN8 reader list; reproduces blend_v38xn8 bit for bit); run_env_1016.sh unchanged.
- Re-based builds 21:51 KST (outputs in g1mix7/best_xn8/ and g1mix8/best_xn8/, logs build_2151.log), bars vs the XN8 rule: BOLD =
  OOF dMSE <= +0.42, kriging22 <= 9.9697, rms vs XN8 >= 0.5. g1mix8: Q86 = B8f + NS6f -0.090 3/5 P 0.60, k22 9.9618 (P(beat XN8)
  0.43), rms vs XN8 0.618, min-sub 0.488 (v37n2) -> BOLD; Q8ALL -0.117 3/5, k22 9.9635 (0.42) -> BOLD; Q86B -0.239 3/5, k22 9.9651
  (0.39) -> BOLD. g1mix7 (BASE = XN8 families): P6 SKIP (rms 0.223 vs XN8), P6B SKIP (0.195; k22 9.9562), PALL no (rms 0.279; k22
  9.9578). No SAFE file.
- Extra #1 = the strictly lowest kriging22 BOLD file = Q86 (submissions/blend_v38q86.csv, md5 8c5bba8b..., OOF 5a85f329...): the
  reading arrived before extra #1 was delivered, so ADDENDUM_q8 rules 1 and 4 apply first (latest kriging, new best); Q8ALL (named in
  ADDENDUM_extra1017 under kriging21) is now second. Workflows verify-q86-1017 (file + pick) and verify-q8all-1017 (file) running.
- Extra #2 after APf (24-5, 24-6) completes about 22:45 KST: rebuild with run_env_1017.sh, then the lowest kriging22 BOLD file on
  hand (skip rule vs all handed-over files incl. Q86), verified, delivered before 00:00 KST.

## 2026-10-17 21:39 KST: NS6Bf complete; XN8 (blend_v38xn8.csv) delivered as 10-17 bold #2 - read this first
- 24-9 b6ns_locK_fsim_s2 (B1 recipe on B6-NS, seed 2; laptop wall 20:22:16-21:05:16 KST, 2580 s, batch 6, no OOM) ingested at
  21:10 KST: train md5 e865e0de..., test f09c7850...; sums match the dump (train 87.9386 / 2768.4439 / 2775.9521, test 175.2104 /
  5551.9398 / 5539.0278); het_cnn-tables.tar now 111 members (re-extracted, md5s equal).
- Official builds at 21:10 KST under kriging21 (g1mix7/build_2110.log; g1mix8/build_2110.log + judge8_2110.log):
  g1mix7 (BASE N2): P6B = N2 + NS6Bf vs N2 -0.023 2/5, k21 9.9600, rms vs N2 0.341 -> no; PALL = N2 + NS6f + NS6Bf +0.080 3/5,
  k21 9.9614, rms 0.435 -> no. g1mix8 (BASE B8f): Q86B = B8f + NS6Bf -0.050 2/5, k21 9.9674, rms vs N2 0.718, min-sub 0.453
  (v38q8) -> BOLD; Q8ALL = B8f + NS6f + NS6Bf +0.072 2/5, k21 9.9645, rms 0.611 -> BOLD. NS6Bf alone (diagnostic) +0.122, k21 9.9802.
  No SAFE file (all fail on OOF, folds or k21 >= 9.95946).
- Bold #2 (ADDENDUM_q8 rule 4 + NOTE_bold2_order, strictly lowest kriging21 among BOLD): XN8 9.9632 < Q8ALL 9.9645 < Q86B 9.9674
  < X8 9.9698 -> XN8 = hardness-cache submissions/blend_v38xn8.csv (md5 8d7bbd7a63f4604f7bfd3b1f6dd311c3; [B1f, B4f, NSf, B8f]; OOF 10.454;
  k18 9.9670, P(beat N2) 0.29; vs N2 +0.189 2/5; rms vs N2 0.591, min-sub 0.269 vs v37x78, vs v38q8 0.824). Delivered 21:39 KST
  after workflow verify-bold2-1017 (rules verifier recomputed every number from raw files and swept 91 + 40 cached files; critic
  DELIVER_XN8; report g1mix8/verify_bold2_output.json, md5 93f786b7..., sweeps in g1mix8/verify_bold2/). The file itself was
  rebuilt bit for bit and compliance-tested by verify-v38 and is unchanged.
- Noted by the verifiers (not failures): XN8 leads Q8ALL by 0.0013 only, and under k21 P(beat N2) ranks Q8ALL 0.44 > Q86B 0.39 >
  XN8 0.29 (XN8's sd is small); the pre-registered order is by point kriging. X8 is now closed (at most one of XN8/X8).
- 10-17 after XN8: both bold slots used; the 3 remaining 10-17 slots take SAFE files only. 24-4 b5ap_locK_real5 (A3 recipe,
  B5-AP) started 21:05 KST; chunk 0 arrived 21:38 KST; APf (24-4..6) complete around 22:45 KST, then PA/PALL and Q8A/Q8ALL are
  built (SAFE needs k21 < 9.95946, unlikely). Then 24-10..12 (NS4f), sections 25 and 26 overnight.

## 2026-10-17 19:21 KST: Q8 read 9.9649369209 (worse than N2); kriging21; N2 stays best; XN8 leads bold #2 - read this first
- blend_v38q8 (Q8 = B8f alone, 10-17 bold #1) public 9.9649369209 (user, main thread 19:18 KST): +0.1091 public MSE vs N2.
  kriging20 had predicted 9.9553 (z +0.44 with sd x1.34); kriging18 9.9702 (z -0.10). ADDENDUM_q8 rule 3: the best stays v37n2
  (the N2 rule) and every bar stays as it was; nothing is re-based (mk_rebase_Q8.py is not used).
- pub/forensics21.py + kriging21.py (mk_pubN.py 21; BUILT_ON v30b) add the reading; g1mix7/run_env_1016.sh KRIG_LATEST -> kriging21.
  pub/read21.py -> read21_out.txt: kriging21 now splits N2's public gain between NSf and B8f (B8f alone 9.9651, NSf alone 9.9743;
  N2 9.9602, XN78 9.9604). LOO z all within +-0.83.
- Re-judged under kriging21 (g1mix6/rebased_N2/judge_k21_1920.log; g1mix7/build_1920.log; g1mix8/build_1920.log + judge8_1920.log):
  XN8 BOLD under both k18 (9.9670) and k21 (9.9632), P(beat N2) 0.29, G1 rms vs v38q8 0.824; X8 BOLD (k18 9.9704, k21 9.9698);
  XN7 SKIP (rms 0.209 vs v37x78); P6 no (k21 9.9616, rms vs N2 0.369); Q86 no (k21 9.9618, rms 0.488). judge_both_N2.py checks the
  14 files before v38q8 (sha-pinned); read21 adds the rms vs v38q8 (XN8 0.824, X8 0.877).
- Asked JINWOO again for "섹션25 실행" and now also "섹션26 실행" (19:21 KST reply).
- Bold #2 (ADDENDUM_q8 rule 4): lowest latest-kriging BOLD candidate once the NS6Bf options (24-7..9) are judged, or at 22:30 KST.
- 20:29 KST (24-7, 24-8 ingested; 24-9 running): hardness-cache scripts/d1016/g1mix8/NOTE_bold2_order.txt (sha256 12ee59c8aeb983c7
  c4f6bc1c9a6c050f07b1752dca8991fb6e5a410cdae6a267, pinned in g1mix8/options.sha256) fixes the cross-round order for bold #2: the
  strictly lowest latest kriging (no "within 0.002" tie-break across rounds; exact ties go to the better vs-N2 OOF). Written after a
  2-seed scratch preview (disclosed in the note: Q8ALL k21 9.9637, Q86B 9.9670, XN8 9.9632) and before NS6Bf seed 2 exists.
  24-7 b6ns_locK_fsim (train md5 7bedb4b3..., test 55a09448...) and 24-8 b6ns_locK_fsim_s1 (train b3b6b2a3..., test e3ed0fa2...)
  ingested, sums match, tar 109 members.

## 2026-10-17 19:06 KST: 24-3 in, NS6f complete; P6 (N2 + NS6f) and Q86 (B8f + NS6f) judged: no file - read this first
- 24-3 b6ns_locK_real5_s2 (B6-NS, A3 recipe, seed 2; laptop wall 18:10:05-18:54:54 KST, batch 6, no OOM) ingested: train md5
  21c89f5ea48b5f23e685fb4bd5dd1139, test md5 c5e4485f43d0be60aec991c58123a66a; chunk sums match (train 87.4877/2788.1477/2751.4636,
  test = chunks 1+2); appended to hardness-cache het_cnn-tables.tar (105 members, re-extract md5 checked). Health corr 0.994.
- g1mix7/run_env_1016.sh now lists v38q8 (handed over 18:46 KST) as submitted, for the rms < 0.25 skip rule. Do not also pass it on
  the command line (build_g1mix7.py asserts each name once).
- Official g1mix7 build on N2 (hardness-cache scripts/d1016/g1mix7/build_1902.log; `source run_env_1016.sh v37n2 9.9594601661`):
  P6 = NSf + B8f + NS6f: vs the N2 rule +0.125 MSE, 1/5 folds, 40-split P(>=0) 0.72; kriging20 9.9629; G1 rms vs N2 0.369 -> no
  (SAFE fails on train; BOLD needs rms >= 0.5). PALL = P6 (NS6f is the only complete section 24 family). md5 50b47d9e...
- Official g1mix8 build (BASE=B8f OUT=g1mix8; build_1903.log, judge8_1903.log): Q86 = B8f + NS6f: vs N2 +0.099, 2/5, P 0.75,
  drop-top-3 +0.469; kriging20 9.9587; rms vs N2 0.488 -> no (BOLD rms bar 0.5 missed). Q8 shows SKIP only because it is the
  handed-over file itself. md5 8c5bba8b...
- NS6f alone (diagnostic): +0.555 vs N2, 1/5. The A3 recipe on B6-NS is weak on train, like NSf (A3 recipe on B5-NS); the B1-recipe
  readers (B8f on B5-NS) carry the gains, which is why 24-7..9 (NS6Bf = B1 recipe on B6-NS) were moved up.
- Next: 24-7 started 18:55 KST; 24-9 ends about 21:10 KST. Then P6B/Q86B and PALL/Q8ALL (NS6f + NS6Bf) are built and judged; bold #2
  of 10-17 = lowest latest-kriging BOLD candidate on hand then (now XN8, kriging20 9.9624) or at 22:30 KST (ADDENDUM_q8 rule 4).

## 2026-10-17 18:46 KST: verify-v38 passed; Q8 delivered as 10-17 bold #1; ADDENDUM_q8 pre-registered - read this first
- verify-v38 (4 agents; report copied to hardness-cache scripts/d1016/g1mix8/verify_v38_output.json): rebuild PASS x3 (independent
  numpy/pandas rule reproduces Q8, XN8, X8, N2, XN78 bit for bit); rules PASS_WITH_NOTES; format/compliance PASS (Q8 with notes;
  perturbing every test reader value leaves all fits and the train OOF bit-identical; only the 433 G1 rows change); critic: DELIVER Q8,
  XN8 then X8 fallback OK. Notes: Q8 is the most public-driven file so far (promoted diagnostic, BOLD only); P(beat 9.9595) 0.575
  under kriging20, 0.415 under kriging18; on train it ties N2 (edge on 3 rows), so its expected private score is about N2's.
- g1mix8/ADDENDUM_q8.txt (sha256 1449bb5503915d2702b726968adc56127019e7f98a6b425176147c7a4f0d388b, written 18:45 KST before delivery
  or any reading) with g1mix8/mk_rebase_Q8.py (sha256 515dcb6d7b602283e839876e8a2e30d5fee5c2f663707a6162fe2cf72848a5ac): new readings
  -> kriging21 (g1mix6 leftovers under kriging18 AND latest); if Q8 reads below 9.9594601661 it becomes BEST (BASE [B8f], g1mix7 =
  g1mix8 build, XN8/X8 re-based on the Q8 rule by mk_rebase_Q8.py); bold #2 on 10-17 = lowest latest-kriging BOLD candidate when the
  NS6Bf options are judged or at 22:30 KST, whichever first; at most one of XN8/X8 ever; no more diagnostic promotions and no
  seed/weight variants of Q8.
- Delivered to JINWOO: blend_v38q8.csv (bold #1 of 10-17), with "섹션25 실행" asked for (queues behind section 24).
- Section 26 (B1 recipe on B5-AP and B3-NS, 3 seeds each, about 2.6 h; after section 25) written to notes/handoff/laptop-gpu.md
  18:49 KST with its judging pre-registered in hardness-cache scripts/d1017/g1mix10 (options.txt 00d1e0a5d84003aca9d0bd6883de52fa
  0e0d16a20256caea4e8464487369f282, mk_g1mix10.py 2f6c2d93..., build_g1mix10.py 1844a3a1...; families APBf, NS3Bf; options PAB,
  P3B, PALL on line A (best file's families) and line B ([B8f]); bars as g1mix9). JINWOO starts it with "섹션26 실행".
- Public-model helper: hardness-cache scripts/d1016/pub/mk_pubN.py (`python3 mk_pubN.py 21 v38q8=<reading>:e68c6808` writes
  forensics21.py + kriging21.py from 20; dry-run checked in scratch).

## 2026-10-17 18:18 KST: resumed 17:47 KST; Q8 (B8f alone) is 10-17 bold #1 once verify-v38 passes - read this first
- JINWOO "재개" 17:47 KST 10-17. Laptop reconnected; 24-2 (b6ns_locK_real5_s1) resumed at fold 3 17:49:59, done 18:10:05 (rc 0,
  1206 s; first part 10-16 about 1700 s for folds 0-2), batch 6, no OOM. 24-3 (b6ns_locK_real5_s2) started 18:10 KST. Original
  section 24 order unless JINWOO writes "섹션24 순서 변경 실행" (24-R in notes/handoff/laptop-gpu.md: NS6B 24-7..9 right after the run
  in progress).
- 24-2 ingested (s19_extract.py, src.het_cnn --ingest): chunk sums match the laptop's (train het 87.5600 logN 2789.5811 bmean
  2749.3245; test 1-500 87.3902 / 2800.4309 / 2738.7501; test 501-1000 86.8961 / 2792.2599 / 2746.0003; no NaN). het_cnn-tables.tar
  now 103 members.
- g1mix6 leftovers re-based on N2 (g1mix6/ADDENDUM_rebaseN2.txt sha256 b2dc60ce..., mk_rebase_N2.py 2e811f6c..., generating
  build_g1mix6_N2.py da10f03c... and judge_both_N2.py 564c7652...; full hashes in g1mix6/options.sha256): XN8 BOLD (vs N2 +0.189,
  2/5; k18 9.9670, k20 9.9624; min rms 0.269 vs v37x78), X8 BOLD (+0.222, 3/5; 9.9704 / 9.9653; min 0.260), XN7 SKIP (min rms 0.209).
- New round d1016/g1mix8 (the B8f line), options.txt sha256 577a31ca8bc1db5fd2d9a1f9b4a947249dff65d1faf332a990cad2cf75db42a8,
  judge8.py 7f1ae8f2ed785b4c1eac063e4b443d49c1e9e6e31b250e5b800aa84d4a44605b, written 17:55 KST before NS6 seeds 1-2 existed.
  Q8 = [B8f] alone was a g1mix6 diagnostic; promoting it is a public-driven piece choice after the N2 reading (its kriging20 was
  known), so it is BOLD only. Q8 -> submissions/blend_v38q8.csv (md5 e68c6808...): vs N2 -0.114 MSE, 2/5 folds, 40-split P 0.450,
  drop-top-3 +0.215; k20 9.9553 (sd x1.34 0.440; P(beat 9.9595) 0.575 under k20, 0.415 under k18); rms vs v37n2 0.614. XN8 ->
  blend_v38xn8.csv (8d7bbd7a...), X8 -> blend_v38x8.csv (909f243b...). Pair rms: Q8-XN8 0.824, Q8-X8 0.877, XN8-X8 0.288.
- d1017/g1mix9 (section 25 readers) pre-registered 18:00 KST: options.txt 5a99362d..., mk_g1mix9.py 14294dd0..., build_g1mix9.py
  2bed2d07... (full hashes in g1mix9/options.sha256); section 25 in notes/handoff/laptop-gpu.md (commit d8bb80f).
- verify-v38 workflow (rebuild, rules, format/compliance, critic) running; rebuild PASS for all three (independent numpy
  implementation reproduces Q8/XN8/X8 bit for bit), rules PASS_WITH_NOTES (Q8 not truly pre-registered; train only ties N2,
  edge on 3 rows; no further diagnostic promotions or seed/weight variants of Q8).
- Plan for 10-17 (5 slots, at most 2 bold): bold #1 = Q8; bold #2 = the lowest latest-kriging BOLD candidate on hand by 22:30
  KST (fallback XN8, then X8); SAFE files from g1mix7/g1mix8 as section 24 families complete (NS6f after 24-3, then APf ...).
  g1mix7 official: `cd .../scripts/d1016/g1mix7 && source run_env_1016.sh v37n2 9.9594601661 && python3 build_g1mix7.py`; g1mix8:
  the same env plus `BASE=B8f OUT=.../d1016/g1mix8`, then judge8.py.

## 2026-10-16 21:37 KST: PAUSED (JINWOO moving location, main thread 21:35 KST) - read this first
- Laptop: all jobs stopped 21:36 KST, GPU idle (0 MiB). Done and sent: sections 21-23 and 24-1 (b6ns_locK_real5). 24-2
  (b6ns_locK_real5_s1) stopped at fold 3 epoch 2; folds 0-2 are kept on the laptop (data/het_cnn/b6ns_locK_real5_s1) and are
  skipped on the rerun. The laptop wrote run24r.sh (24-2 to 24-12 in order; copy notes/run24r.local.sh, git-excluded) and
  LAPTOP_RESUME.local.md (its resume order). It committed and pushed nothing. On "재개": 24-2 restarts at fold 3, then 24-3 ..
  24-12 (about 7-8 h), with the 3 chunk messages and the wall time after each run.
- Cloud: nothing running. Records committed and pushed; reader tables in hardness-cache het_cnn-tables.tar (101 members, incl.
  b6ns_locK_real5); laptop dumps in scripts/d1016/s19/; verify-v37 report in scripts/d1016/g1mix6/verify_v37_output.txt; g1mix7
  dry runs and seed-0 looks in scripts/d1016/g1mix7/scratch_1016/.
- Section 25 design workflow stopped in its synthesis stage. Its 9 proposals (3 designers) and both critiques are saved in
  scripts/d1016/s25design/journal_results_partial.json (workflow script beside it); its brief predates the N2 reading. Rules
  critic: fix NS7Bf (B1 recipe on EfficientNet-B7 NS) and NS4Bf (B1 recipe on B4-NS), drop NS7f, APBf, NSM, B1M, NSLF, NSW4, FDx;
  feasibility critic: keep 6, fix 3 (NS7Bf, B1M, FDx).
- Slots: 10-16's last slot lapses (P6 needs NS6 seeds 1-2). 10 left: 5 on 10-17 (at most 2 bold), 5 on 10-18.
- After "재개": (1) ingest each section 24 run as its chunks arrive (s19_extract.py RUN 0 1 2, src.het_cnn --ingest, refresh
  het_cnn-tables.tar); (2) when the 3 NS6f seeds are in, run g1mix7 officially:
  `cd /mnt/project-files/work/hardness-cache/scripts/d1016/g1mix7 && source run_env_1016.sh v37n2 9.9594601661 && python3
  build_g1mix7.py` (BEST = the best public file at build time; bump KRIG_LATEST after any new reading), apply options.txt
  SAFE/BOLD, verify, deliver; later families (APf, NS6Bf, NS4f) the same way as they complete; (3) re-run the section 25
  synthesis with the N2 reading folded in, write section 25 into notes/handoff/laptop-gpu.md plus a pre-registered
  g1mix8/options.txt, push, and ask JINWOO for "섹션 25 실행" so the GPU does not idle after section 24.

## 2026-10-16 21:33 KST: blend_v37n2 9.9595 = NEW BEST (automatic final while best); blend_v37x78 9.9599 - read this first
- JINWOO 21:30 KST: blend_v37x78 = 9.9599001009, blend_v37n2 = 9.9594601661. Both beat v35xn (9.9722); N2 (NSf + B8f, the two
  Noisy Student readers only) is the best public file, XN78 0.0004 behind (public MSE +0.009). Observed dMSE vs v30b -1.215 /
  -1.223; kriging19 predicted -1.125 / -1.017 -> z -0.32 / -0.31. 10-16 used v35xn, v36l7, v37x78, v37n2: 1 SAFE slot left today
  (0 if v34l2 went in on 10-16), 11 in all (1 + 5 + 5).
- pub/forensics20.py + kriging20.py (= 19 + both readings, BUILT_ON v30b); read20.py -> read20_out.txt. LOO z under kriging20 all
  within +-0.7 (v34l2 +0.68, v31s +0.62). Re-krige vs the new best (k20 rmse; OOF vs the N2 rule): XN8 9.9624 (+0.189, 2/5), XN7
  9.9679 (+0.071), X8 9.9653 (+0.222), diagnostic B8f alone 9.9553 (-0.114, 2/5; diagnostics are never submitted). No g1mix6 file
  beats N2.
- g1mix7 re-based per options.txt: BASE = [NSf, B8f], BEST = v37n2 (BEST_RUNS = the N2 rule, reproduces blend_v37n2 bit for bit),
  KRIG_LATEST = kriging20; env helper g1mix7/run_env_1016.sh (`source run_env_1016.sh v37n2 9.9594601661`). Scratch look with
  NS6 seed 0 only (decide nothing): P6 = [NSf, B8f, NS6f] vs the N2 rule -0.004, 2/5 folds, kriging20 9.9627 -> fails SAFE, so the
  last 10-16 slot most likely stays empty; the official build runs after 24-3 anyway.

## 2026-10-16 21:30 KST: blend_v37x78 (XN78) and blend_v37n2 (N2) delivered as 10-16 SAFE files; NS6f pending - read this first
- B8 seeds 1-2 (23-4 b5ns_locK_fsim_s1, 23-5 b5ns_locK_fsim_s2) and 24-1 (b6ns_locK_real5, NS6 seed 0) ingested from the laptop
  dumps (scripts/d1016/s19/, het_cnn-tables.tar 101 members). Official g1mix6 build 20:30 KST (all options; log copied to
  g1mix6/build_official_2030KST.log), then g1mix6/judge_both.py (sha 858b7005..., written 19:40 KST before B8 seeds 1-2 existed) under kriging18 AND
  kriging19, skip rule vs all 12 submitted v30b-base files.
- Two options pass SAFE under both models:
  XN78 = [B1f, B4f, NSf, B7f, B8f] -> submissions/blend_v37x78.csv (md5 93cf38f0...): honest nested CV 10.504 -> 10.444; vs XNfixed
  -0.416 MSE, 4/5 folds, 40-split P 0.00, drop-top-3 -0.128; kriging18 9.9677, kriging19 9.9644 (sd x1.34 0.279, P(beat 9.9722)
  about 0.7); min G1 rms vs submitted 0.435 (v35xn).
  N2 = [NSf, B8f] -> submissions/blend_v37n2.csv (md5 d30852ba...): 10.504 -> 10.445; vs XNfixed -0.398, 3/5, 40-split P 0.175,
  drop-top-3 +0.241 (narrow: two fold-4 rows carry most of the gain); kriging18 9.9697, kriging19 9.9698 (sd x1.34 0.660, P about
  0.5); min rms 0.801 (v35xn).
  Others: XN8, X8 fail on folds (2/5); XN7 still fails kriging18 (9.9727).
- Verification workflow (verify-v37: dumps, rebuild, rules, format, compliance, laptop commands) passed every check; critic:
  deliver both, XN78 first, both on 10-16. Delivered to JINWOO 21:24 KST in one reply, before any reading.
- g1mix6/ADDENDUM_delivery.txt (21:23 KST, sha in options.sha256, before delivery): a delivered N2 stands whatever XN78 reads;
  ADDENDUM_newreading rule 2 re-bases only files not yet delivered (re-based on XN78, N2 would fail: +0.018, 1/5).
- Next: on each reading, pub/forensics20 + kriging20 (then 21). When 24-2 (b6ns_locK_real5_s1) and 24-3 (_s2) arrive: ingest,
  refresh the tar, then g1mix7 for the third 10-16 slot (only if v34l2 counted on 10-15; SAFE only) with KRIG_LATEST = the
  latest kriging and SUBMITTED_EXTRA += v36l7, v37x78, v37n2; if XN78 has read below 9.9722, BASE = B1f,B4f,NSf,B7f,B8f and
  BEST = v37x78. Section 25 (GPU after section 24 ends, about 05:00 KST 10-17) is being designed; JINWOO starts it with
  "섹션 25 실행".

## 2026-10-16 19:40 KST: blend_v36l7 read 9.9779 (v35xn 9.9722 stays best); kriging19 written; XN7 still fails - read this first
- JINWOO 19:34 KST: blend_v36l7 = 9.9778545791 (worse than v35xn by 0.0057; better than its kriging18 9.9843). It went in right
  after the 19:29 KST ask "submit today only if the list shows v34l2 on 10-15", so v34l2 is taken as a 10-15 file (inference).
  10-16 used: v35xn, v36l7 (both bold, cap reached); 3 slots left today, SAFE files only; 13 left in all (3 + 5 + 5).
- hardness-cache scripts/d1016/pub/forensics19.py + kriging19.py (= 18 + v36l7, BUILT_ON v30b); read19.py -> read19_out.txt.
  v36l7 observed dMSE vs v30b -0.857, kriging18 predicted -0.728 sd 0.287 (x1.34: 0.384) -> z -0.34. LOO z under kriging19:
  v35xn -1.10, v36l7 -0.39, v34l2 +0.69, v31s +0.64; others within +-0.24.
- Re-krige (k18 -> k19 rmse): XN7 9.9727 -> 9.9693 (passes k19, still fails the k18 SAFE bar 9.9722, so not delivered per
  ADDENDUM_newreading rule 1; BOLD also fails, rms vs v35xn 0.379); XNf 9.9740 -> 9.9737; B8 seed-0 scratch looks: XN8 9.9668 ->
  9.9664, XN78 9.9673 -> 9.9644, X8 9.9716 -> 9.9697, N2 9.9693 -> 9.9705 (decide nothing; official build after 23-5).
- Next: after 23-5 (B8 seed 2) ingest, rerun g1mix6/build_g1mix6.py (all options), then apply SAFE under BOTH kriging18 (in the
  build) and kriging19 (read19-style), skip rule vs all 12 submitted v30b-base files incl. v36l7, verify, deliver.
  Section 24 files: g1mix7 with KRIG_LATEST=kriging19 and v36l7 in SUBMITTED_EXTRA.

## 2026-10-16 19:30 KST: blend_v36l7 (XL7) delivered as 10-16's second bold file; XN7 failed; B8 seeds 1-2 pending - read this first
- Laptop resumed after JINWOO's "재개" (18:16 KST). Its background runner had died with the 17:37 KST park: 23-1 (B8 seed 0,
  b5ns_locK_fsim) had finished; 23-2 restarted from scratch at 18:18 KST. Now ingested and in het_cnn-tables.tar (95 members):
  ev2l_locK_fsim_s2 (22-2), b5ns_locK_fsim (23-1), b5ns_locK_real5_s1 (23-2), b5ns_locK_real5_s2 (23-3). 23-4 (B8 seed 1) started
  19:21 KST, then 23-5, then section 24. If the laptop session parks again the runner dies too: ask JINWOO for "재개".
- Official builds (18:24-18:26 KST): g1mix6/judge_g5.py (OUT g1mix6/g5judge) and g1mix5/build_g1mix5.py give the same md5 for
  every option. Only XL7 = [B1f, B4f, B7f] qualifies: vs XNfixed -0.301 MSE 4/5, 40-split P 0.10, drop-top-3 +0.117 (two rows,
  TRAIN_000183 and TRAIN_000410, carry 106% of the gain), kriging18 9.9843 (BOLD bar 9.9854), rms vs v35xn 0.661, min rms vs all
  11 submitted v30b-base files 0.467 (v33xb). LL, XLL, ALLL fail (kriging18 10.0113 / 9.9910; ALLL rms vs v35xn 0.450).
- g1mix6/ADDENDUM_newreading.txt (18:27 KST, before XL7 was delivered or read): after each new public reading, files not yet
  delivered must pass under both kriging18 and the latest kriging; a new best re-bases the g1mix6 bars as g1mix7 does.
- Verification workflow (6 agents, all PASS_WITH_NOTES, critic DELIVER; reports in /tmp/v36verify, gone with the container):
  dumps == transcript == parquet for all 8 XL7 runs; rebuild byte-identical three ways (incl. an implementation without
  g1core/n5lib); rules and skip rule re-derived; test-reader perturbation leaves every fit and the OOF bit-identical; laptop
  commands match the plan as announced. Open notes: (a) v34l2's submission day is our inference, not the user's word - if it
  went in on 10-16, XL7 is 10-16's third bold file, so JINWOO was asked to check the list (10-15: submit today; 10-16: hold for
  10-17). (b) judge_g5.py's skip list covers only 6 of the 11 submitted files: check all 11 by hand for later files.
  (c) The 21-3 resume (folds 2-4) command and the laptop's score.json args were never shown to the cloud; ask the laptop for
  them in the next section (resume guard does not check arch/batch/seed).
- Delivered: hardness-cache submissions/blend_v36l7.csv (md5 4ed3ca53...) and blend_v36l7_oof.csv (38e730cd..., OOF RMSE 10.4499).
- XN7 [B1f, B4f, NSf, B7f] with all seeds (g1mix6/build_g1mix6.py, 19:27 KST): vs XNfixed -0.327 4/5, P 0.03, kriging18 9.9727
  -> fails SAFE (needs < 9.9722) and BOLD (rms vs v35xn 0.379 < 0.5). NS seeds shrink as B6/B7 did: XNf (v35xn families, all NS
  seeds) kriging18 9.9740. B8 seed-0 scratch look (decides nothing): XN8 9.9668, XN78 9.9673, N2 9.9693 SAFE flags; X8 fails
  (2/5 folds). Next: after 23-5 rerun build_g1mix6.py (all options), check rms vs all 11 submitted files (+ v36l7 if submitted).

## 2026-10-16 18:15 KST: laptop session parked at 17:37 KST; code-check inventory done; cloud-only files backed up - read this first
- The laptop's Remote Control session went offline at 17:37 KST (parked after sitting idle; the device stays online). The
  user's next message in the main thread restarts it with its transcript. Asked JINWOO at 18:14 KST to write one line
  ("재개"). 22-2 (B7 seed 2) was due ~17:35 KST; its dump has not arrived. Sections 23 and 24 follow once it resumes.
- Code-check prep (top-10 private teams submit code): hardness-cache repro/INVENTORY.txt (1269 lines, plan A: ship the 69
  cache scripts byte-identical at their absolute paths, Docker or root symlinks; owner decisions O1-O7 in section 4.9) and
  repro/CRITIC.txt (gaps: build_u1 needs L1_formula/CANDIDATE_v28n_gridhet_V4_lv.csv; build_clean needs N4/clean/options.sha256;
  feats_lines.py / seg_alt.py missing from the copy list; disclosure items D1-D8 plus E1 (render-gap study scored test images,
  label-free). From cached intermediates every blend v18..v35xn rebuilds byte-identically; GPU steps are not bit-exact.
- Cloud-only artifacts backed up: hardness-cache cloud-only-2026-10-16.tar (data_restored/*.parquet x15, data/het_cnn/prep.npz,
  submissions/blend_v22b(+oof), blend_v23_oof, experiments/cnn_*kf; list and md5 beside it).

## 2026-10-16 16:40 KST: B7 seed 1 in; g1mix7 builder ready - read this first
- 22-1 (ev2l_locK_fsim_s1, 3880 s, batch 6) dumped, sums verified, ingested; het_cnn-tables.tar refreshed. 22-2 (B7 seed 2)
  started 16:29 KST; then section 23 (23-1..23-5), then section 24 (JINWOO's "섹션 24 실행" 15:42 KST; queued behind 23-5).
- Two-seed scratch look (decides nothing): XN7 vs XNfixed -0.291 4/5, kriging18 9.9719 (SAFE flag, at the bar); XL7 -0.279 4/5,
  kriging18 9.9848 (BOLD flag, near the 9.9854 bar); B7 alone vs XNfixed -1.160 (seed 0) -> -0.879 (2 seeds): it shrinks with
  seeds as B6 did.
- Evening steps: after 22-2 ingest run g1mix5/build_g1mix5.py and g1mix6/judge_g5.py (official), then g1mix6/slate.py; after
  23-3 (NS seed 2) run g1mix6/build_g1mix6.py (XN7 is then complete); after 23-5 rerun it (all options). Deliver a file when it
  qualifies (g1mix6/ADDENDUM_g5judge.txt rule 3), each after an independent verification.
- g1mix7/build_g1mix7.py (sha 2e22250a..., generator mk_g1mix7.py) self-tested (SELFTEST.txt): BESTfixed reproduces v35xn and,
  with BEST_RUNS for v33xb, v33xb bit for bit. Env BASE / BEST_NAME / BEST_PUB / BEST_RUNS / SUBMITTED_EXTRA / KRIG_LATEST /
  RENAME set the best file, the latest kriging and OOM renames at build time.

## 2026-10-16 15:41 KST: B7 seed 0 in; g1mix5 judged by the g1mix6 bars; g1mix7 + laptop section 24 written - read this first
- 21-3 done on the laptop (ev2l_locK_fsim, B1 recipe on effnetv2-l, batch 6, ~4041 s over two sittings, no OOM); dump verified
  (chunk sums) and ingested; het_cnn-tables.tar refreshed. 22-1 (B7 seed 1) started 15:25 KST; 22-2, then section 23 follow.
- Seed-0 scratch look (decides nothing; g1mix5/ADDENDUM_B7seeds.txt rule 1): g1mix6 XN7 [B1f+B4f+NS+B7] vs XNfixed -0.358 MSE
  4/5, 40-split P 0.00, kriging18 9.9701 (SAFE flag); g1mix5 XL7 [B1f+B4f+B7] vs XNfixed -0.369 4/5, kriging18 9.9828 (BOLD
  flag); B7 alone vs XNfixed -1.160. Official builds wait for all seeds: g1mix5 options need B7 seeds 1-2 (22-2), XN7 also
  needs NS seeds 1-2 (23-3), the B8 options need 23-5.
- g1mix6/judge_g5.py (sha e35706bc...) = build_g1mix6.py with the g1mix5 option lists: applies the g1mix6 bars (vs XNfixed,
  kriging18, rms vs v35xn) to LL/XLL/XL7/ALLL, as g1mix6/options.txt requires for unsubmitted g1mix5 files. Its seed-0 md5s
  equal build_g1mix5.py's (LL 1f924fac, XLL 8042b4e1, XL7 163586ea, ALLL 02ced768).
- d1016/g1mix7 pre-registered 15:42 KST (options.txt sha 2a7f15a0...) before any section 24 run: new families NS6f (B6-NS),
  APf (B5-AdvProp), NS6Bf (B1 recipe on B6-NS), NS4f (B4-NS), 3 seeds each; options BASE + one family, BASE + all, where BASE =
  the best public file's families; bars = the g1mix6 rule re-based on the best file.
- Laptop section 24 (notes/handoff/laptop-gpu.md) written for those 12 runs; needs JINWOO's "섹션24 실행"; runs after 23-5.

## 2026-10-16 14:57 KST: blend_v35xn 9.9722 = NEW BEST; g1mix6 (NS family) pre-registered; laptop sections 22 + 23 - read this first
- User 14:51 KST 10-16: blend_v35xn (g1mix4 XN = B1f + B4f + NS reader b5ns_locK_real5) public 9.9721726754 = new best and
  automatic final while best (-0.29 public MSE vs v33xb; kriging17 had predicted 9.9992, z -0.93). 14 submissions left
  (10-16: 4 more, v34l2 counted on 10-15; 1 of them bold by the 2-per-day cap; 10-17 and 10-18: 5 each).
- Public model now kriging18 (scripts/d1016/pub/forensics18.py, kriging18.py; read18_out.txt re-kriges every built option).
  No built option qualifies under the new bars (XLN: train -0.160 vs XN 4/5, kriging18 9.9794, rms vs v35xn 0.277).
- d1016/g1mix6 (options.txt sha a585cfbd..., written 14:54 KST before any new reader): families B1f, B4f, NSf (NS + seeds 1-2),
  B7f (ev2l_locK_fsim + seeds), B8f (B1 recipe on B5-NS + seeds); options XN8, XN7, XN78, X8, N2; reference XNfixed (= v35xn,
  md5 asserted); SAFE: better than XN on train (>= 3/5, P <= 0.5) and kriging18 < 9.9722; BOLD: vs XN <= +0.42, kriging18 <=
  9.9854, rms vs v35xn >= 0.5; these bars replace the v33xb-relative ones for anything not yet submitted. Options are built only
  when all seeds of their families are in. Builder build_g1mix6.py, self-test SELFTEST.txt.
- Laptop: 21-3 (B7 seed 0) running; section 22 (B7 seeds 1-2) queued by the user's "섹션 22 실행" (14:52 KST); section 23
  (B8 seeds 0-2, NS seeds 1-2; runs before 22 if 22 has not started) needs the user's "섹션23 실행".
- Code-check prep: inventory workflow writing hardness-cache repro/INVENTORY.txt (read-only trace of the build chain).

## 2026-10-16 14:49 KST: resumed; v34l2 read 10.0055; 10-16 = XN now, B7 file after section 22 - read this first
- User (14:43 KST 10-16): "재개", blend_v34l2 10.0054603343 (public dMSE vs v30b -0.305; kriging16 predicted -0.436, z +0.21).
  B4f + effnetv2-l (A3 recipe) reads worse than B4 alone (v33b4 9.9961) and than v33xb (9.9868): the effnetv2-l A3-recipe reader
  does not carry to public. Best and automatic final is still blend_v33xb 9.9868. 15 submissions left (5 a day 10-16..10-18).
- Public model: scripts/d1016/pub/forensics17.py / kriging17.py (+ v34l2); read17_out.txt has kriging16 vs kriging17 for every
  g1mix3/g1mix4 option (XN 9.9978 / 9.9992; XL 9.9886 / 9.9942; XLN 9.9939 / 9.9986).
- g1mix5 addendum (ADDENDUM_B7seeds.txt, 14:48 KST, before any B7 data): B7 files only from 3 seeds (laptop section 22); every
  file must pass SAFE/BOLD under both kriging16 and kriging17; bold order by kriging17. Builder updated (B7_SEEDS env, kriging17,
  rms vs v34l2); self-test reproduces the g1mix4 3-seed md5s.
- 10-16 slot 1: submissions/blend_v35xn.csv = g1mix4 XN [B1f + B4f + B5-NS] (md5 d1eab131..., BOLD under both krigings).
  Next: 21-3 chunks -> ingest; section 22 (B7 seeds 1-2, needs the user's "섹션22 실행") -> ingest -> build_g1mix5.py -> the second
  10-16 bold file if one qualifies.

## 2026-10-15 21:07 KST: PAUSED (JINWOO moved location, main thread 21:04 KST) - read this first
- Laptop: all jobs stopped 21:05 KST, GPU idle. 21-3 (ev2l_locK_fsim) stopped at fold 2 epoch 6; folds 0-1 kept on the laptop;
  on "재개" the laptop reruns the identical 21-3 command (finished folds are skipped; laptop-gpu.md section 21, pause note).
- Cloud: nothing running. All records committed and pushed; reader tables (data/het_cnn/*/*.parquet, *.json) backed up to
  hardness-cache het_cnn-tables.tar (restore: tar -xf /mnt/project-files/work/hardness-cache/het_cnn-tables.tar from the repo
  root; the laptop dumps are also in scripts/d1016/s19/ for re-ingest).
- Open for JINWOO: slot 5 of 10-15 (until 00:00 KST 10-16) = submissions/blend_v34l2.csv (BOLD, expected about 9.999).
- After "재개": ingest ev2l_locK_fsim (s19_extract.py RUN 0 1 2, src.het_cnn --ingest), run g1mix5/build_g1mix5.py, apply the
  g1mix4 SAFE/BOLD rules and ADDENDUM_order.txt, then plan the 10-16 slate (XN is the next BOLD file). Add any score the user
  reports to forensics16 PUB/BUILT_ON and re-krige first.

## 2026-10-15 21:00 KST: sections 20-21 (effnetv2-l, B5-NS); slot 5 of 10-15 = blend_v34l2 (BOLD) - read this first
- Section 20: B6 = ev2l_locK_real5 (A3 recipe on effnetv2-l) and NS = b5ns_locK_real5 (EfficientNet-B5 Noisy Student).
  Section 21: B6 seeds 1-2 (ev2l_locK_real5_s1, _s2; laptop 3870 s / 3878 s each, batch 6), then 21-3 = ev2l_locK_fsim (B1 recipe
  on effnetv2-l), training on the laptop since 20:37 KST. Health corr with A3f 0.993 for every new reader.
- g1mix4 official 3-seed build (hardness-cache scripts/d1016/g1mix4/B6x3, log build_log_B6x3.txt, notes in RESULT_B6.txt):
  vs the fixed XB4 option XL -0.236 (3/5), XN -0.120 (3/5), XLN -0.280 (3/5), L2 [B4f+B6] -0.269 (3/5), B6 alone -0.601 MSE;
  drop-top-3 positive for all, gain mostly fold 3 (0-based). kriging16: XL 9.9886, XLN 9.9939, XN 9.9978, L2 9.9989, B6 10.0089.
  No SAFE file; BOLD = XN and L2 (XL and XLN miss rms vs v33xb >= 0.5). ADDENDUM_order.txt (18:33 KST, before the seeds):
  XN and L2 within 0.002 kriging, so the better OOF first -> L2.
- Delivered 21:00 KST for slot 5 of 10-15: submissions/blend_v34l2.csv (md5 698f5eb67d2f6af5e646cbd878ee767f, OOF
  blend_v34l2_oof.csv). Independent verification (3 checkers, scripts/d1016/g1mix4_verify/): byte-identical rebuild, dumps =
  ingested parquets, 1000 rows, IDs in order, only the 433 G1 rows change vs v30b, rules and compliance PASS.
- Next: XN is the next BOLD file (10-16). 21-3: extract/ingest ev2l_locK_fsim, then run scripts/d1016/g1mix5/build_g1mix5.py
  (options LL, XLL, XL7, ALLL; same SAFE/BOLD rules and order, pre-registered 18:34 KST). When the user reports the v34l2
  score, add it to forensics16 PUB/BUILT_ON and re-krige before choosing the 10-16 files.

## 2026-10-15 16:54 KST: section 19 done, no 10-16 file from it; section 20 (effnetv2-l, B5-NS) training - read this first
- All section 18/19 readers are in (B1 s1/s2, B4 s1, B5, RG = rgy4_locK_real5, RN = r50d_locK_real5; health corr 0.986-0.996).
  Final g1mix3 build (hardness-cache scripts/d1016/g1mix3/RESULT_S19.txt): no option beats v33xb. vs the fixed XB4 option:
  X3 +0.052, M2 +0.121, ALL6 +0.466, DIV4 +0.661 MSE; kriging16 best X3 9.9898 vs v33xb 9.9868. RegNetY and ResNet50-D are weak
  G1 het readers (alone +1.77 / +1.61 MSE), so non-EfficientNet backbones are closed. Seed averaging also gave nothing:
  XB4f with 3+2 seeds +0.024 vs the seed-0 XB4 file.
- g1mix4 (section 20 options XL, XN, XLN, L2, B6; SAFE and BOLD slate rules) pre-registered 16:40 KST before 20-1 started:
  scripts/d1016/g1mix4/options.txt (sha a46f9a87...), builder build_g1mix4.py, self-test SELFTEST.txt (reproduces g1mix3 and
  v33xb). Ingest each section 20 reader with scripts/d1016/s19/s19_extract.py RUN 0 1 2 and python -m src.het_cnn --ingest,
  then run build_g1mix4.py. Slot 5 of 10-15 is still open until 00:00 KST if a SAFE file appears before then.

## 2026-10-15 14:19 KST: round N7 found no lever for the 8s; reader pipeline continues; E1 test running - read this first
- PLAN_N7 (hardness-cache scripts/d1016/N7/PLAN_N7.txt, written 14:15 KST before a container restart; lens reports in
  N7/L1..L4, L3 verified): all four lenses null. Per-phase / per-grain functionals (L1) and kernel mean-embedding distribution
  regression (L2) on clean rows are worse than the base; a power test rules out a simple additive per-grain functional of >= 10
  HV^2. Denoise-then-segment on G1 rows (L3) recovers geometry the base already has; its nested -1.70 was refuted (2 rows carry it,
  100 re-splits -0.12/-0.50). Feasibility (L4): public < 9.0 needs a single lever of >= 15 public MSE; P(< 9.0 by 10-19) about
  0.2%; realistic landing 9.83-9.95 with the reader pipeline.
- Next, in order: (1) section 19 readers -> g1mix3 (pre-registered) -> files that beat v33xb on the fixed-XB4 comparison and
  kriging16. 17-3 closed 14:39 KST: its gate failed rms(full6 test - 5-fold-mean test) 1.630 > 1.5 (corr 0.9960 and KS p 0.683
  passed); per the section rule no kf member and no P2 (C0/C1) look; y-CNN line closed. 19-1 (B5) died at fold 2 epoch 9
  (Windows service install, 14:27 KST), resumed and finished 15:00 KST. B5 ingested; g1mix3 rebuilt with kriging16
  (scripts/d1016/g1mix3/RESULT_B5.txt): every B5 option is worse than XB4 on train (M2 +0.009, X3 +0.039, ALL6 +0.080, B5 +0.200
  MSE) and on kriging (X3 9.9889, M2 9.9918 vs v33xb 9.9868), so slot 5 of 10-15 stays unused. Laptop now on 18-1 (B1 s1). (2) E1 = full-menu nested test of the post hoc G1 elongation offset (PLAN_N7 2b): FAIL, verified
  (scripts/d1016/N7E1/RESULT.txt, VERIFY.txt: 113 options frozen 14:21 KST, +2.03 all-500, 2/5, re-splits +1.62, P 0.62); no file,
  G1 elongation family closed. (3) Port the final
  file's recipe into the repo for the top-10 code check (Evaluation.md).

## 2026-10-15 13:03 KST: v33xb 9.9868 (first below 10, new best); target now the 8s; round N7 running - read this first
- User (12:56 KST): blend_v33xb 9.9868332159, v33t3 9.9921641853, v33b4 9.9960660597, v33s5 10.0031466453; then "앞으로의
  목표지향점수를 8점대로 변경": the target is now public below 9.0 (about -19 public MSE from 99.74). 4 of the 5 slots of 10-15 used;
  16 submissions left. All four read 0.02-0.29 MSE better than their kriging predictions (z -0.1 to -0.2). Public order matches
  train: B4/B1 readers best, the A3-family file worst. Re-kriged (scripts/d1016/pub/forensics16.py + kriging16.py + read16.py):
  A3B4 9.9994 and S4X2B4 9.9908 cannot beat v33xb, so blend_v33ab is dropped.
- The 10-18 public-tuned fallback (s110/test_a3x110.csv) was only for "still >= 10.0 on 10-18"; that condition is gone.
- Laptop (user's "색션 19 실행", 12:56 KST): 17-3 started 12:58 KST, then 19-1 (B5 = ev2m_locK_fsim, batch 8 like B4), 18-1 (B1 seeds
  1-2), 19-2 (B4 seed 1), 19-3 (RegNetY-4GF), 19-4 (ResNet50-D); dumps c0-c2 after each run. The laptop's git pull had been blocked
  by the cloud-committed experiments/cnn_ev2s_A3init_anc05_s3 files (it moved its local copies aside): do not commit experiment
  files the laptop also writes locally without telling it. When B5 arrives: ingest, run d1016/g1mix3/build_g1mix3.py (options
  pre-registered 12:33 KST; self-test reproduces the 10-15 files), krige with kriging16, slot 5 of 10-15 if an option beats v33xb.
- Error budget of the current best (train OOF, MSE 109.6): clean 69.9 (34% of SSE; ~110 for N 100-200, ~47 for N >= 270), G1hi 180
  (53%; 233 in the lowest-N quintile), midLF 106, LV 90. Round H (d1011/H/PLAN_H.txt) put the leaders' true edge at ~14 public MSE,
  most likely the 1/N share on coarse rows.
- Design round N7 (workflow, started 13:02 KST; scripts/d1016/N7/<lens>/REPORT.txt, VERIFY.txt, PLAN_N7.txt): L1 per-phase /
  per-grain label functionals on clean rows, L2 distribution regression on per-grain sets, L3 denoise-then-segment on G1 rows, L4
  honest feasibility table for 8.x.

## 2026-10-15 12:27 KST: resumed after an 88 h pause; 4-file structural slate for 10-15; section 17-3 running - read this first
- User (11:03 KST): blend_v32m 10.0298 (hold; v30b 10.0207 stays best). Next target: public below 10.0, i.e. public dMSE
  below -0.414 vs v30b. Exactly 20 submissions left (5 a day, 10-15 to 10-18); deadline 10-19 10:00 KST.
- User's card (11:11 KST), "last-day insurance": the honest route until 10-18 (structural pieces at their honest-CV strengths;
  public readings only pick pieces, never weights, strengths or seeds). If public is still >= 10.0 on 10-18, submit
  hardness-cache scripts/d1015/s110/test_a3x110.csv (A3 offset x1.10, public-fitted, expected 9.9945, md5 ceae9744...). That is
  the one sanctioned exception to "never set a strength from public".
- 10-15 slate (hardness-cache scripts/d1015/g1mix2/README.txt), delivered 12:22 KST: submissions/blend_v33xb (B1 + B4 readers),
  blend_v33t3 (A3 5 seeds + B1 + B4), blend_v33s5 (A3 5 seeds + B1), blend_v33b4 (B4 alone); each changes only the 433 G1 test
  rows of v30b. Held: blend_v33ab (A3 + B4) as a slot-5 fallback. All BOLD: vs the fixed A3 rule -0.21 to -0.51 MSE, 4/5 folds,
  re-splits P>=0 <= 0.03, but no gain left without the top 3 rows. Joint public model: P(at least one < 10.0) 0.695.
- Withdrawn before delivery: the d1015/g1mix seed slate (S4X2, A3X, P1-P3): single-seed draws of one reader, so a public pick
  among them is a seed lottery. Rule: never submit single-seed variants side by side or together with their own average.
- Fair train comparison for a fixed option is the FIXED replace_A3/0.75 option (v30b's real test rule), not the v30b honest OOF
  (p30), which carries a -0.271 selection penalty that flatters every fixed option.
- Builder bug fixed: `from g1core import *` rebinds HERE to N4/g1, so build_g1mix*.py re-set HERE after the import (stray
  outputs deleted 12:21 KST; README in d1015/g1mix).
- Section 18 (user, 11:47 KST "섹션 18 실행"): readers B4 = ev2m_locK_real5 (A3 recipe, effnetv2-m) and A3 seed 4 =
  ev2s_locK_real5_s4 sent as text dumps (d1015/s18/, helper s18_extract.py) and ingested into data/het_cnn/ bit-identical.
  18-1 (B1 seeds 1-2) trains after 17-3. Open question for 10-16: a B1 seed average would be a seed variant of the B1 files
  already submitted, so it can only replace them, not run beside them.
- Section 17 (laptop back online 11:10 KST): 17-0 self-test and 17-1 gate passed; 17-2 stage-1 PASS on the diversity route
  (cnn_ev2s_A3init_anc05_s3 OOF 12.260 vs s3 13.25; fine-noisy cell 11.60 vs 11.26 missed); verdict pushed 12:08 KST and 17-3
  (6 seeds + full-data kf) started. Cloud steps when gate17_3 arrives: extract chunks (d1015/s17/s17_extract.py checks n and
  sum), write experiments/cnn_ev2s_A3init_anc05_gpu_s6 and _gpu_s6kf (kf oof = s6 oof, test = kf), run hardness-cache
  scripts/d1011/N6/run/cnn_a3init/cnn_swap.py (options C0/C1 frozen 12:10 KST, sha 8cfa7cff...), verify independently; a
  slot-5 file only if C1 is picked, else blend_v33ab tonight.
- Public-history model: d1015/g1mix/forensics15.py (PUB, BUILT_ON) + kriging.py; joint slate model d1015/g1mix2/joint/. When
  the 10-15 scores come in, add them to PUB with BUILT_ON 'v30b', re-run both, then plan 10-16 (pieces only). Keep at least one
  10-18 slot for the fallback while public is >= 10.0.

## 2026-10-11 23:15 KST: N6 round -> blend_v32m (BOLD) for 10-12; laptop section 17 running - read this first
- N6 design (hardness-cache scripts/d1011/N6/PLAN_N6.txt; 4 lenses, 2 judges): picks P1 g1_sizesub (CPU), P2 = N6-G1 y-CNN
  started from the A3 reader (GPU section 17 + a cloud N4 look), P3 a label-free gate for strength-only G1 files (adopt when
  a strength option appears; it removes nothing this round). Everything else dropped (list in PLAN_N6 section 4).
- P1 result (N6/run/g1_sizesub): honest -0.154, 2/5, drop-top-3 +0.095, re-splits -0.32: FAIL under SAFE. Handed over as
  BOLD anyway (user wants bold files; change rms 0.36 on 433 G1 rows): submissions/blend_v32m.csv md5 0a6a4190...,
  independent rebuild exact (N6/run/g1_sizesub_verify). Expected public 10.0154; <= 9.986 carried, >= 10.056 dropped.
- Section 17 (commit fa733c1): train_cnn --init-het/--head hetfpn/--anchor/--nlm-zero-stem, het_cnn --dump-blocks,
  tools/n6/gate17.py. User's run line 21:33 KST ("섹션17 올라오면 바로 실행") covers 17-0..17-3. The laptop cannot read
  thread replies, so the cloud writes verdict lines to notes/handoff/gate17_verdicts.txt and pushes; the laptop polls git.
  Cloud jobs per step: 17-1 message 1 -> `python3 hardness-cache/scripts/d1011/N6/run/gate17/gate17_bc.py <msg>` for gate
  (c) (the 200 y-hat values are never scored against labels); messages 2-3 -> `python -m src.het_cnn --ingest
  ev2s_locK_real5 c0 c1 --dump-blocks` (for N6-G2's pre-check). 17-2 -> stage-1 bars with
  hardness-cache scripts/d1011/bold/L6_leaders/l6_cnn_gate.py <oof> --ref s3. 17-3 -> PLAN_N6 P2 (options C0/C1 frozen
  before the 17-2 bars are read: N6/run/cnn_a3init/options.txt).

## 2026-10-11 20:05 KST: v31s 10.1953, v31f 10.1705 (both worse than v30b 10.0207) - read this first
- User (20:00 KST): blend_v31f 10.1704996257, blend_v31s 10.1953136309. All 5 slots of 10-11 used. v30b stays the best
  and the automatic final.
- Reading: v31s >= 10.06, so the G beta shrink is dropped (+3.53 public MSE vs v30b; expectation +0.29, sd about 1.5).
  Decomposition (scratch, rebuild exact): 99% of the v31s test change is the offset shrink, the larger stretch k
  (1.372 -> 1.463) is a small part. As a public slope this puts lambda near 1.35 +- 0.2 vs train inner-CV 0.7, i.e. public
  values the A3 offset more than train does; not acted on (public never sets strengths). v31f vs v31s -0.025: hold,
  A3 stays. Lesson: train-side shrinkage found by inner CV on 208 G1 rows did not transfer; treat G1-row CV gains under
  about 1 MSE as noise.

## 2026-10-11 20:00 KST: N5 round (new directions) -> blend_v31s and blend_v31f for the last 2 slots of 10-11 - read this first
- User (18:26 KST): plenty of time left; new directions, many plans including bold ones, new CSVs. User (project chat
  18:58 KST): this project owns the laptop GPU unconditionally until 10-19 10:00 KST (no per-run grant, no hand-back;
  never touch the robot project's files or jobs). See notes/handoff/laptop-gpu.md.
- Shared harness hardness-cache scripts/d1011/N5/n5lib.py (load30, honest nested vs v30b, 40 re-splits, N4 verdict).
  Every candidate has a frozen options.txt (sha in result.json) and an exact rebuild check of v30b.
- FAIL: g1cell_a3 (G1 cell boundary by the A3 reader) +0.89; g1cpu2nd (CPU localized reader as a second term) +1.26;
  np_route (clean-preset noisy route) never picked; g1_fgls (FGLS beta) +0.43 but re-splits -1.07 (P 0.00).
- New finding: v30b's OLS G beta over-fits. Shrinking it (lambda about 0.7, picked by inner CV) helps on honest nesting and
  re-splits; mechanism = winner's curse of the best-of-18 pick plus stacking leakage, which also fits public having
  carried only about a third of G's CV gain. Found post hoc in g1_fgls diagnostics, so labelled BOLD.
- Files (hardness-cache submissions/, independently rebuilt in N5/*_verify, exact):
  - blend_v31s = v30b with G beta x lambda (g1_shrink): honest -0.56, 4/5, re-splits 39/40 (mean -1.25), md5 c50a5e8b...;
    expected public 10.035. Reading: <= 9.99 shrink carried, >= 10.06 dropped, between = hold.
  - blend_v31f = v31s with the B1 reader (ev2s_locK_fsim) in the A3 slot (g1_shrink_b1): honest -0.11, 3/5, re-splits
    38/40 (mean -2.07), md5 d4c1babf...; expected 10.049. Reading vs v31s: -0.03 or better = switch to B1, +0.03 or worse
    = keep A3.
- Laptop readers ingested in data/het_cnn/ (cloud): ev2s_locK_real5_s2, ev2s_locK_fsim (B1), ev2s_locK_real5_s3.
  Seed 4 exists only on the laptop. B4 (ev2m_locK_real5, effnetv2-m backbone) pending: score it with
  `python3 hardness-cache/scripts/d1011/N5/rswap/rswap.py ev2m_locK_real5 b4` after ingesting the chunks.
- Later GPU ideas from the N5 judge (N5/PLAN_N5.txt): re-dump A3 blocks b00..b15, other-architecture reader on real5,
  multi-task reader (bd|logN, dark fraction), beta-NLL y-CNN, Noise2Self denoiser then the clean pipeline, preset-aware
  reader, D4-TTA plus full-data A3 refit.

## 2026-10-11 18:30 KST: v30b public 10.0207 (best), v30a 10.0312; no file for slots 4-5 - read this first
- User (17:20 KST): blend_v30b 10.0206885658 (new best, auto-final while best), blend_v30a 10.0312450482. Reading:
  G (A3 reader on the 433 G1 test rows) carried about a third of its CV gain (public dMSE about -0.72 vs v29g; full-CV
  hypothesis -2.34, calibrated -0.34); C (clean mix 0.75) read worse (+0.21 MSE) and is dropped. v30b is the new base.
- User asked for 2 improved files for slots 4-5. Every candidate failed its pre-declared N4 rule vs v30b (hardness-cache
  scripts/d1011/N4/<dir>/README.txt; honest nested dMSE vs v30b):
  gscope (G only on G1hi / separate midLF fit) +0.60, 0/5; cleancnn (CNN het mixed into clean V4) picks V4 5/5 (CNN het
  on clean images is worse than one segmentation map); r2seed (A3 seeds 0+1, laptop 17:22-17:46) +0.14, 2/5;
  gnoise (noise-dependent offset slope) +2.64, 0/5; r3 (A3 on 6-map V4 block targets, laptop section 16-1) +0.93, 1/5.
  The extra readers are near copies of A3 on G1 rows (het corr 0.998 seed 1, 0.993 V4 targets), so reader variants of
  the A3 recipe are used up. Slots 4-5 of 10-11 stay empty; v30b stays the base.
- Code: src/het_cnn.py gained --target-file (A4 patch, selftest ok 14) and tools/a4/v4_block_targets.py (58d880e).

## 2026-10-11 16:40 KST: new criterion N4 and the v30b / v30a files - read this first
- User (13:02 KST): make a new criterion and a 4-submission plan for the goal. Criterion N4 (hardness-cache
  scripts/d1011/N4/slate/PLAN_N4.txt + its 14:40 addendum): judge by expected value vs v29g instead of pre-registered
  bars. SAFE = selection-honest nested dMSE < 0, >= 3/5 folds, 40-re-split mean < 0, drop-top-3 < 0. BOLD = CV cost
  <= +0.5 MSE, at most 2 per day. Public readings only decide which pieces go in, never weights. Public expectation is
  calibrated as null + 0.47 (cv - null) (v29g / v28n / v26r readings).
- Screen (workflow n4): G = A3 het reader (ev2s_locK_real5) + stretch step 0.75 on G1 rows: honest -2.40 (strict -2.20),
  4/5, re-split -0.52 -> SAFE. C = clean mix weight 0.75 (post hoc) -0.51, re-split +0.21 -> BOLD. CNN pooling FAIL
  (+1.07). S (stretch only) and C alone have ~0 expected value: not submitted. User chose "2개만" on the card (14:07 KST).
- A3 test het came from laptop section 15 (CPU dump, same-run sums check passed). Label-free gate D4 missed by one row
  (9/433 test offsets beyond the train max vs < 2%); the 9 test offsets were clipped to the train range (N4/g1clip).
- Files (hardness-cache submissions/, independently rebuilt, md5 match): blend_v30b = v29g + G_clip (md5 26291df0...,
  433 rows, expected public 9.94 full / 10.04 calibrated, P(<=10) ~0.36); blend_v30a = v30b + C (md5 ad21ece3..., 952 rows,
  9.91 / 10.06, ~0.33). Order: v30b then v30a; slots 4-5 of 10-11 empty. G family closes if v30b >= 10.1505.

## 2026-10-11 10:20 KST: blend_v29g public 10.0564 (best; inconclusive); section 14 all FAIL - read this first
- blend_v29g scored 10.0564 public (user, 08:38 KST): -0.015 vs v28n 10.0712, new best (auto-final while best).
  Pre-registered reading: inconclusive (9.90-10.24), v29g stays the base for every later file.
- Size check (hardness-cache laptop-gpu-2026-10-11/RESULTS.txt): 580 test rows changed (about 174 public). Full-CV
  hypothesis predicts public dMSE -7.9 (sd 3.3); observed -0.30 (z +2.3); no-signal predicts about +6.4 (z -2.0).
  Likeliest: about half the CV gain is real (CV optimism from picking V4 among ~25 variants plus public noise).
  No train/test shift in V4 on clean rows (quantiles equal, KS p 0.27, 0.6% of test outside the train range).
- Section 14 (laptop GPU from 08:17 KST, hand-over tool; robot runner blocked until hardness_gpu_done): CTRL passed;
  every pre-registered candidate so far FAILED (ledger: hardness-cache laptop-gpu-2026-10-11/RESULTS.txt):
  - C1 AdaBN (replace): dMSE -0.34, folds 2/5, sign-flip p 0.39.
  - C2 A2 fsim-50% retrain (twobeta): -0.90, 2/5, p 0.33. C3 A3 route-K real5 (twobeta): -1.39, 3/5, p 0.28.
  - Post hoc A2+A3 replace -2.20 (3/5, p 0.05), but selection-honest nesting over the 13 options gives -0.58 (3/5,
    p 0.39; 40 re-splits -0.19). GPU-A family closed, no file.
  - GPU-B 1/N-weighted y-CNN (--wN 1): stage 1 all 13.174 vs bar 13.12, 3/5 folds, cell 11.41 vs 11.20; swap +0.50.
  - GPU-C (full-image grid4 y-CNN, 09:33-10:09 KST): cv 13.426, fails all three bars (clean het4 +0.52 vs <= +0.35,
    G1hi het_cnn +0.28 vs <= +0.20, all 13.426 vs <= 13.274); swap into v28n +0.39.
  - Section 14 is finished with no new file; the remaining 4 slots of 10-11 stay empty. The laptop wrote
    hardness_gpu_done at 10:09 KST (GPU back with the robot project).

## 2026-10-11 05:30 KST: bold round -> blend_v29g for 10/11 slot 1 - read this first
- User (10-10 24:00 KST): public top-10 8.8662 / 8.9139 / 9.1817 / 9.5134 / 9.7090 / 9.8001 / 9.8040 / 9.8362 / 10.0415 /
  10.0470; we are 12th. No methods posted on the boards. "Any idea, be bold." Read as relaxing our own bars (labels may
  now score candidates, with proper nesting), never the competition rules. Evaluation.md: private 100%, top-10 code check.
- Bold round (hardness-cache scripts/d1011/bold: 8 lenses, TRIAGE.txt, impl_C1-C5 with verify, PLAN_BOLD.txt):
  - **blend_v29g** (= U1/C4; md5 496f5cc9...; OOF 10.618 vs v28n 10.991): het re-measured as the mean over 6 segmentation
    maps of the exact 4x4-grid statistic (V4); clean 9.10 -> 8.36. Details in experiments/LEADERBOARD.md and
    hardness-cache submissions/blend_v29g.json. Expected public about 9.80 (9.60-9.95). Lines: <= 9.90 carried,
    9.90-10.24 inconclusive (still the base), >= 10.24 refuted (close the grid-het re-measurement family).
  - The other 4 slots of 10/11 stay empty (C1/C2 are near-copies of U1; C3 T-CP rides inside U1; C5 found nothing).
  - GPU plan (PLAN_BOLD section 4): GPU-A het session on G1 rows (AdaBN on train G1 images, degrade_sim 50% retrain,
    route-K real preset), GPU-B y-CNN with 1/N row weights (--wN 1), GPU-C full-image grid4 y-CNN (optional). User OK'd
    GPU use tonight on the card (00:05 KST); take it via the handover tool, never pause the robot job unless the user
    says so. Code check before the laptop run: workflow gpu14-review (bold/gpu14_review_A.txt, _B.txt); fixes applied
    06:00 KST (het_adabn never falls back to CPU, train_cnn --wN guards, q7 --gate-hi). Laptop commands:
    notes/handoff/laptop-gpu.md section 14 (A1-A3, then GPU-B stage 1, GPU-C optional); pre-registered bars in the
    PLAN_BOLD addendum 2026-10-11 05:55 KST (C1 gate 0.995, OOM rule, GPU-B binding swap bar). Waits for the
    laptop session to reconnect and the user's line in the main thread.
  - For the top-10 code check: port V4 / FD6t / f8_C2 / the T-CP gate into src/ by 10/16 so the repo rebuilds v29g.
- Laptop CPU checks 10-11 00:05 KST: per-grain mixture law null (9.53-9.90 vs 9.52); no train/test duplicates.


## 2026-10-10 22:30 KST: design round H toward 9 - no candidate; 10/11 default is no submission - read this first
- User chose on the card (20:38 KST): focus next on the 1/N low-grain-count error.
- Round H (hardness-cache scripts/d1011/H/PLAN_H.txt; 4 lenses, 8 proposals, 3 independent judges each):
  7 proposals dropped 0/3 (per-grain set learner, clean size-block re-measurement, booking 1/N as irreducible,
  pixel-fd-matched degrade_sim_v3, a term reader in it, realistic-noise y-CNN swap, ev2m y-CNN). Only A-P1 kept:
  ask the user (optional) for the public top-10 and any method posted on the competition's boards; ideas only,
  nothing copied, nothing that uses test images.
- Facts recorded:
  - The leaders' edge is real (z 2.9-3.4), about 9-17 public MSE (central 14). G1hi at exact clean level alone gives
    8.91, so the most consistent story is less 1/N error on coarse rows. Our 1/N slope b already fell 23.6k -> 15.1k
    (v18 -> v28n); all eight of our model families share v28n's coarse-row error (a common blind spot, not a proven floor).
  - Our public reading is lucky by about 2 sd (101.43 vs an expected 120-123 MSE). Expected private for v28n about
    10.98-11.10 (sd about 0.3; inferred from nested OOF).
  - Labels follow the realized grain map (nominal-only labels refuted for het4 and bd_frac); residuals are Gaussian at
    every N; error variance follows the grain count, not het4^2. No stop rule on the 1/N share.
  - Stage 0's fd ceiling (r 0.69) is a degrade_sim property; real G1hi keeps more fd signal, but v28n already uses it
    (scope correction appended to G/PLAN_G.txt). Never train a G1hi reader in the degrade_sim world.
- Outlook: P(public <= 9 by 10/19) below 0.01; P(<= 10) about 0.01-0.02. Only a new lever changes that.


## 2026-10-10 20:00 KST: no file for 10/11 slot 1; G1hi term-reader route closed - read this first
- Best public is still blend_v28n 10.0712 (automatic final while it stays best).
- Plan G (hardness-cache scripts/d1011/G/PLAN_G.txt): a CNN that reads the formula7 terms on high-noise (G1hi) images,
  trained on clean train images degraded by the new simulator degrade_sim (G/forensics). Default NO-GO; a label-free
  CPU stage 0 decided whether the route re-opens.
- Stage 0 (G/s0, label-free, no test image, each item checked by an independent verifier):
  - S0a realism PASS: degrade_sim copies vs real G1hi train, fast-stat AUC 0.679 (bar 0.83), quality 0.546 (0.80).
  - S0b fd ceiling FAIL: a per-grain reader given the TRUE grain map reads fd_area at r 0.69 (bar 0.90; S0b' 0.79).
  - S0d het PASS (marginal): the cpu het CNN reads het better on degrade_sim copies than on its own renders
    (pc 0.423 vs 0.347, bar +0.05; bootstrap interval touches 0).
  - Mechanical outcome: re-opened on the het branch. But gate G1 needs f7-level corr >= 0.86 on sim copies, and with fd
    capped at 0.69 the best possible is 0.79 even with every other term exact (G/s0/s0_g1_bound.py). So the route is
    closed at G1 without training; no GPU word or G1hi label exception is needed for it.
- 10/11 slot 1: no file. Decision card to the user (20:05 KST): next focus on the 1/N low-grain-count error
  (recommended), a het-only GPU retrain in the degrade_sim world (needs GPU word + one G1hi label exception; about
  -0.03 if it passes, inferred), or both.


## 2026-10-10 15:00 KST: blend_v27u 10.1041 (best); two files for the last 10/10 slots - read this first
- Public (user, 14:20 KST): **blend_v28n 10.0712** (best; -0.033 vs v27u, inconclusive band, carried needed <= 10.0641).
  Finals v27u + v28n (user's rule). blend_v28p next; switch to v28p only if it reads <= 10.0112. To reach 10: -0.071.
- Public (user, 14:21 KST): blend_v28p 10.0796 (-0.025, inconclusive by 0.005). Joint reading: no conclusion on
  low-half vs pooled; no further G1hi stretch files. Finals stay v27u + v28n. All 5 slots of 10/10 used.
- 10/11 design (15:50 KST, hardness-cache scripts/d1011/PLAN_1011.txt): 4 lenses, 6 proposals, all 18 judge votes
  'drop'. Default for 10/11: submit nothing; finals stay v27u + v28n. Anatomy: 66% of train SSE follows a 1/N law
  (grain-level sampling, slope 1.00, calibration flat; looks label-side), 23% is the G1hi noise constant. Only
  unopened channel: a GPU edge-cut head on G1 rows (P(pass) about 0.07; needs the user's GPU word and an exception to
  the G1hi pre-set), not recommended.
- Public (user, 12:08 KST): blend_v27u **10.1041**, exactly the value predicted from the v26L and v26r readings (build
  check passed). Finals v26L + v27u (user's choice 12:06 KST). User: the last two 10/10 submissions only after careful
  review.
- Design (hardness-cache scripts/d1010/D): the idea that fold averaging flattens test predictions on G1hi was refuted
  (train-only inner-CV ratio 0.98; the test flatness is composition). What survives: inside G1hi the OOF slope depends on
  noise (low-noise half 1.49, high half 0.94), found post hoc.
- **blend_v28n** (submit first; md5 396a52f2...): half-step stretch of the low-noise half of G1hi (167 test rows),
  k 1.262. Registered bar PASS (all-500 -3.07, 4/5); independent rebuild. Expected public about -0.03 (null +0.07),
  reading sd 0.08. Lines: <= 10.0641 carried, > 10.1441 refuted (close the G1hi stretch family), > 10.3541 build check.
- **blend_v28p** (submit second, whatever A reads; md5 d35dfd6b...): pooled stretch of all 320 G1hi rows, k 1.100.
  Bar PASS (-0.80, 3/5 at the line). Lines: <= 10.0741 carried, > 10.1241 refuted. Joint reading table in D/B/PREREG.txt.
- Finals rule (user's choice 14:17 KST, before any reading; appended to D/A and D/B PREREG): always v27u + v28n;
  v27u + v28p only if R_B <= R_A - 0.06; v28n > 10.3541 means build check and finals v26L + v27u.

## 2026-10-10 09:30 KST: blend_v26L 10.1302 (best, carried); v26r inconclusive - read this first
- Public (user, 09:23 KST): **blend_v26L 10.1302** (-0.109 vs v25; pre-set carried line 10.2094; expected -0.03, as
  built -0.074). The unchanged clean formula7 transfers strongly to noisy rows where the clean segmentation works.
- blend_v26r 10.2136 (-0.026): inside the pre-set inconclusive band 10.212-10.256 by 0.0016. ringall is closed; v26r is
  never a final. The union blend_v27u is not submitted (its public score follows without a slot: 10.1041) and is not a
  final.
- Finals: v26L first, v25 second. New base for further files: blend_v26L. To reach 10: public MSE -2.62 more.
- 10/10 slots 3-5: only files that pass a new single-shot pre-registered label test plus an independent review.
  Next-round design (gate extension, G1 rows, formula form, fresh angles): hardness-cache scripts/d1010/N/.
- Next-round design (11:20 KST, scripts/d1010/N/PLAN_NEXT.md): 6 proposals, 3 judges, no pick, no label look. E-narrow G1
  dark fraction (corr 0.97 with the d1009/A null estimator), reliability-gated formula7 on noisy rows, unstable-clean
  fallback, FGLS formula7 and a G1 plug-in formula were dropped; the midLF RC-f7 near-miss has expected effect about 0.
  10/10 slots 3-5 stay empty unless the user swaps the second final.
- Open decision (card to the user 11:30 KST): second final v25 -> blend_v27u (v26L + ringall on clean rows; must be
  submitted to be selectable; its public must read 10.1041 or it is a build error). Rationale: if the better of the two
  finals counts, v25 is almost surely worse than v26L, while ringall's honest train gain (clean MSE -1.1 per row) matches
  its public reading (-1.0), so v27u is at least as likely to beat v26L on private (expected about -0.02).
- 12:06 KST: the user chose the swap. Submit blend_v27u (md5 97a7cf2b...) on 10/10; if it reads 10.1041 (4-decimal
  rounding) the finals are v26L + v27u, otherwise it is a build error and the finals stay v26L + v25.

## 2026-10-10 01:30 KST: two files for 10/10 (blend_v26L, blend_v26r) - read this first
- d1010 C clean-row search (128 candidates, one nested harness, fixed bar): no survivor; new clean terms are exhausted at
  n = 267 (scripts/d1010/C/SYNTHESIS.md). Leads: ringall_avg5 (post hoc, refuted by re-splits).
- **blend_v26L** (hardness-cache/submissions, md5 5e47b2fd...): the unchanged clean formula7 at fixed w 0.5 on the 42
  line-visible mid-noise test rows. Pre-registered single test PASS (scripts/d1010/L). Expected about -0.03 vs v25.
  Reading: <= 10.2094 carried (first final), 10.2094-10.2694 inconclusive (finals unchanged), > 10.2694 refuted.
- **blend_v26r** (md5 1b6037e7...): bold probe of ringall_avg5 on the 519 clean rows. Expected about -0.03, SD 0.028.
  Reading: <= 10.184 carried (first final); otherwise finals unchanged; > 10.256 closes the edge-count family.
- The two change disjoint rows, so their public MSE changes add exactly; a union file (blend_v27u) is built only if both
  carry. Slots 4-5 stay empty unless a new pre-registered test passes. Plan: scripts/d1010/PLAN_1010.md.
- Phase-contrast diagnostic (scripts/d1010/E): the real G1 phase contrast c_d is near-constant, but the matrix spread e
  is narrower than the render prior; a narrowed-prior dark-fraction estimator gains in renders (R2 0.684 -> 0.751) but
  the plan's strict rule closes the route; at most a weak 10/11 CPU test.

## 2026-10-09 23:20 KST: blend_v25 10.2394 (best); route K ended with no file - read this first
- Public (user, 22:51 KST): blend_v25 **10.2394** (-0.274 vs v24; past J/PREREG's <= 10.40 line). Public gained about the
  full train nested gain. Public-optimal mix weight about 0.66 vs train 0.56: no re-weighting (worth < 0.02). Finals:
  v25 first, v24 second. Slot 5 of 10/09 left empty. To reach 10: about -0.24 more (about one more v25-sized gain).
- Route K (laptop GPU 22:55-23:16 KST, returned with hardness_gpu_done): base re-scoring matched (0.5213); D1 'real'
  prior failed P_cur 0.492 < 0.501 and P_real 0.666 < 0.671 (real clean R rose 0.786 -> 0.854; G1 agreement with v24's
  estimator 0.94); D2 ev2m P 0.519 < 0.571. PICK none: no label test, no file (scripts/d1010/K/PREREG.txt). The het-CNN
  route is saturated.
- Running on cloud CPU: workflow clean-next-1010 (hardness-cache scripts/d1010/C): edge-corrected grain-size
  estimators, edge-term refinements, a cross-fitted residual probe and a mid-noise extension of formula7, each judged by
  one nested harness with a fixed bar (clean nested gain >= 0.08, 4/5 folds, perm p < 0.05) and two skeptics.

## 2026-10-09 19:40 KST: blend_v25 handed over (slot 4); route K prepared - read this first
- Section 12 (GPU 15:25-16:38 KST, returned with the done file): 4 stronger localized designs all beat v24's model on
  renders (0.521 -> 0.532-0.546) but none cleared the pre-set +0.05 bar: PICK none, no 5-fold run (scripts/d1009/H).
- **blend_v25.csv** (md5 f1e6dbc4..., hardness-cache/submissions; builder scripts/d1009/J/j_build.py): on the 519 clean
  test rows (raw ic_noise < 9.5) 0.5*v24 + 0.5*formula7, other rows bit-identical to v24. formula7 = OLS on fd_area,
  pore_frac, logasp_aw, S_align, log N_eff, het4, bd_frac (share of grains cut by the image edge; J/cleanhead feats).
  Nested CV 11.465 -> 11.263, clean 9.705 -> 9.251 (5/5 folds). Found by exploring the labels, so expect ~2/3 of it.
  The earlier "bd_frac explains the anti-Hall-Petch slope" claim is withdrawn (collinear with logN, corr -0.97).
  Pre-set reading vs v24 10.5130 (J/PREREG.txt): <= 10.40 final candidate; 10.40-10.46 keep; 10.46-10.53 finals
  v25 + v24; 10.53-10.60 v24 stays first; > 10.60 drop v25 and every edge-count extension. The 0.5 weight is never
  tuned on public. Slot 5 of 10/09 stays empty.
- Headroom (scripts/d1009/I, J): public/nested ratio ~0.84; <= 10 needs ~10.5 more public MSE. G1-only routes cap near
  -0.1 to -0.2 public. Honest odds of <= 10 by 10/10: under 1%.
- Route K (next GPU lever, decision card to the user 19:30 KST): retrain the localized het CNN on a more realistic
  render preset (`src/het_cnn.py --render-preset real`, line/blur/noise closer to the real noisy images), stage 1 ->
  1b -> one label test (scripts/d1010/K/k_test.py, PREREG.txt) -> blend_v26 only on PASS; effnetv2-m fallback. Needs
  the user's own GPU line in the thread and the robot hand-over tool re-armed.
- Route K prep done 21:20 KST (9ea4ec1): `src/het_cnn.py --render-preset real / --eval-presets / --eval-only`, the
  label-free pick `src/het_cnn_pick_k.py`, laptop commands in notes/handoff/laptop-gpu.md section 13. Prior gates
  passed with the pre-registered seeds (AUC 0.792 vs 0.902 now, ridge 98%, coverage 94%; K/gates/). PREREG addenda 1-2
  fix run names, the logN bar (vs the base runs, not the prep) and pin the blob ids. Two review rounds; defaults
  bit-identical.

## 2026-10-09 15:45 KST: blend_v24 scored 10.5130 (best); section 12 running - read this first
- Public 10/09 (user): blend_v24 **10.5130** (-0.332 vs v23, train-expected -0.22 +- 0.24). Public-optimal scale along
  v23 -> v24 about 1.24 (worth about 0.01): no rescale (hardness-cache scripts/d1009/I/pub_analysis). Estimated private
  gain vs v23 about -0.27 +- 0.16, P(better) about 0.95. Final picks: v24 first; v23 second unless v25 passes.
- GPU: the robot side's hand-over tool (C:\Dacon\WM_Runtime\handover_after_jobs.py <ledger job>) handed the GPU over at
  15:25 KST when robot submit4 ended; the laptop started section 12 at 15:27 (screens -> pick -> 5-fold), return with
  hardness_gpu_done expected 16:30-16:50 KST. User line 14:43: "인계도구로 두 세션이 서로 점유를 주고받는 활동 시작할것".
- Section 12 next steps: 12b pick output; stage 1b (pooled P >= 0.54); then once `python H/h_test.py --het
  data/het_cnn/ev2s_loc12 --name loc12` (vs v24); if PASS `G/g_build.py blend_v25 data/het_cnn/ev2s_loc12`, verify,
  hand over as slot 4. Pre-set reading vs v24 in H/PREREG.txt addendum (<= 10.463 in line, > 10.613 demote).
- Slot 5 of 10/09 stays unused: the design review (scripts/d1009/I) found no CPU candidate worth a label test.
- Reaching <= 10 needs about 10.5 more public MSE, as much as the whole het route gave from v21 to v24; it needs new
  information for the G1 rows with ic_noise >= 12 (nested RMSE 14.1 vs 9.7 clean).

## 2026-10-09 afternoon: blend_v24 handed over (3rd slot)
- Public 10/09 (user): blend_v23 **10.8445** (-0.130 vs v22), blend_v23m15 **10.8393** (v23 with the 9.5-12 line-free part
  of the offset x1.5; public-only hedge, -0.005). Public-scale analysis (hardness-cache scripts/d1009/Q): public alone
  wanted that part x2.5-3, train says x0.9, combined private optimum x1.2-1.6; after v23m15 the public optimum is ~x1.3.
  No more rescaling.
- Laptop section 11 (GPU exclusive 12:57-13:47 KST, returned with the done file, liaison confirmed): localized het CNN
  ev2s_loc_e32r4 + _s1. Stage 1: render partial corr 0.49 / 0.50 (old 0.25), within 0.71 (old 0.43), real clean
  held-out 0.72 mean (old 0.29). Stage 2 (G/PREREG.txt, run once on the seed average): G1 13.770 -> 13.385, 4/5 folds,
  perm p 0.0005, drop-top 0.298: PASS. Dumps + parquets: hardness-cache laptop-gpu-2026-10-09/.
- **blend_v24.csv** (md5 f4953098..., hardness-cache/submissions; builder scripts/d1009/G/g_build.py): nested CV 11.465
  (v23 11.653). Independent verification (scripts/d1009/V24: reproduce, audit, skeptic, judge): exact, no leakage;
  all the gain is in raw ic_noise >= 12 (5/5 folds), the 9.5-12 band is neutral; concentrated in a few coarse-grain
  rows; bootstrap P(no gain vs v23) ~7%. Expected public 10.64-10.69 +- 0.2. Reading: <= 10.744 carried over;
  10.744-11.108 not shown (keep v24); >= 11.108 harmful (back to v23). Final picks: v24 + v23 (safe).
- Next lever (asked the user for GPU, decision card 14:15 KST): laptop-gpu.md section 12 screens 4 stronger localized
  designs (lambda-within 2 / 4, 64 epochs, 8 renders) on folds 0-1, picks by src/het_cnn_pick.py, trains the pick on 5
  folds; stage 2 vs v24 pre-registered in scripts/d1009/H (PREREG.txt, h_test.py sha 9cc5fc76...). If PASS:
  blend_v25 via G/g_build.py on the new dir. The laptop needs the user's own line in the thread to start.

## 2026-10-08 20:30 KST: paused for the user's move (laptop off) - read this first
- **On "재개":** the laptop runs `git pull origin claude/lb-under-10-7080jr`. If the user has written a GPU grant in the
  thread ("GPU 사용 허가", exclusive or shared), the laptop runs `notes/handoff/laptop-gpu.md` section 11 (a -> b -> c ->
  d -> e). Section 10 is superseded (dark-phase fraction is not the noisy-image problem). No laptop job was running
  at the pause; nothing on the laptop needed backing up (only CPU smoke outputs).
- **10/09 files so far:** `blend_v23.csv` is the safe file (md5 15fbe5e8...; in hardness-cache/submissions). Verified
  2026-10-08 evening (hardness-cache scripts/d1009/V1, V2, P23): exact reproduction, no leakage, but its gain over
  v22b comes from one train image (TRAIN_000239); expected public about -0.02 vs v22, a tie. d1009 X1 (whole-image
  noise-robust size-spread statistics) failed its pre-registered test.
- **Main lever in progress:** localized het CNN for line-free noisy images. `src/het_cnn.py` (92d6ebc) gained `--head
  fpn`, `--mosaic/--mosaic-dlogn`, `--zoom`, `--lambda-within`, `--screen` (defaults bit-identical). Pre-registration:
  hardness-cache `scripts/d1009/G/PREREG.txt` (stage-1 design pick from a CPU screen, GPU stage-1 gate, stage-2 bar on
  G1 vs blend_v23: >= 0.10 better, 4/5 folds, perm p < 0.05, survives dropping the top row).
- CPU screen (cloud, resnet18, fold 0, 16 epochs; hardness-cache `scripts/d1009/G/screen/`): the within-image
  deviation loss lifts render partial corr 0.20 -> 0.47-0.49; the combined design was picked. Its 5-fold CPU version
  (`cpu_loc_r18_e16`, render pc 0.35) FAILED stage 2 at 00:15 KST 10/09: G1 13.770 (v23) -> 13.848, 2/5 folds, perm p
  0.021 (signal, but weaker than the old GPU estimator). Render gains did not carry to the real labels. The GPU
  effnetv2-s run of the same design (section 11) is the last pre-registered test of this route (p bar 0.025).
- Told the user at 00:20 KST 10/09: submit blend_v23 first on 10/09 (safe); GPU grant needed on return.

## 2026-10-08 17:33 KST: v21 11.0620, v22 10.9742 (best) - read this first
- Public (user, 17:33 KST): blend_v21 11.0620 (-0.013 vs v19), **blend_v22 10.9742** (-0.088 vs v21). By the pre-set
  reading (<= -0.04) the GPU het-CNN estimate carries real signal on the noisiest images (raw ic_noise >= 12); v22's
  public gain beat its CV gain (-0.068). CV-to-public offset for v22: 0.714. All 5 submissions of 10-08 used.
- LB (user, 17:33 KST): #1 8.8662, #2 8.9630, #3 9.6509. Task now: improved files for 2026-10-09, "아주 신중한 판단으로".

## 2026-10-08 evening: files for the last 2 slots of the day
- **Submit in this order** (files in `/mnt/project-files/work/hardness-cache/submissions/`, also attached in the thread):
  4. `blend_v21.csv` (safe, nested CV 11.756, md5 a57dc832...): v19 + v20's in-cell CNN share, cell CNN test averaged
     half-and-half with the laptop full-data runs. Expected ~11.03-11.04.
  5. `blend_v22.csv` (exploratory probe, nested CV 11.688, md5 0ab58a1a...): v21 + beta * (het_cnn - g(logN_cnn)) on
     the 326 test images with raw ic_noise >= 12 (165 train), g linear and beta = 103.5 fitted on those train rows
     against the v21 nested OOF residual. FAILS its bar: gated 14.658 -> 14.492 (-0.165), perm p 0.02, but 3/5 folds
     (fold deltas -0.41 / +0.002 / -0.45 / +0.54 / -0.55; need 4/5), and the design (gate >= 12, residualised form)
     was chosen after seeing the primary test. Read v22 - v21 on public: <= -0.04 -> the CNN het estimate carries
     signal on noisy images, worth more GPU seeds / a better renderer; >= +0.04 -> drop the route; between -> undecided.
- GPU het CNN (`src/het_cnn.py`, laptop RTX 5060, exclusive hand-over 16:30-17:22 KST, returned with hardness_gpu_done):
  effnetv2-s, 32 epochs, 4 renders per clean train image in the noisy-preset style, targets = the 16 block b values and
  log N_eff measured on the clean originals (y never read). Seed 0: render corr 0.80, partial corr | logN 0.25; seed 1
  0.74 / 0.14; test het_cnn of the two seeds correlate 0.988. Dumps: hardness-cache
  `laptop-gpu-2026-10-08/het_ev2s_e32r4/{train,test}.csv` (seed 1 stays on the laptop in data\het_cnn\ev2s_e32r4_s1).
- Pre-registered primary stage 2 (`src/het_cnn_offset.py`, gate 9.5, OLS on [1, het_cnn, exp(0.25 logN_cnn)] on
  blend_v19): noisy 13.721 -> 13.799, 2/5 folds, FAIL (9.5-12 band 11.25 -> 11.77, 12-15 16.08 -> 15.95, 15-21 13.98 ->
  13.94). Secondary stage 2b and the v22 builder: hardness-cache `scripts/het-1008/stage2b/{stage2b.py,build_v22.py}`
  (`python build_v22.py --train <seed csv...> --test <seed csv...>`). Averaging seed 1 would flip the fold count to 4/5
  in ~40% of simulated draws (fold 1 is a tie), but moves the file itself by far less than its public noise.
- Also null today under pre-registered bars: clean-het refinement (`src/het_refine.py`, 9.717 -> 9.694, 3/5 folds) and
  the member route (het columns inside the five members, blend_v22m 11.794 vs v21 11.756).
- Next (after the user's v21 / v22 scores): noisy images still hold ~63% of the squared error. If v22 helps, train more
  het-CNN seeds and widen the renderer's noise range; if not, the noisy-image het estimate needs a different input
  (e.g. denoised or restored views) rather than more seeds.

## 2026-10-08 afternoon: blend_v19 scored 11.0751 (best)
- Public scores for 2026-10-08 (user, 14:45 KST): v18 11.7304 (tie with v17), **v19 11.0751** (-0.655 vs v18, so the
  het term is real by the pre-set rule), v20 11.6924 (-0.038 vs v18, below the 0.07 bar for a dedicated cell CNN).
  3 of 5 used; 2 left for the day.
- `src.het_blocks` het4 breaks right above its gate. Partial corr with the v18 residual given log N_eff: 0.52 / 0.48 /
  0.46 for raw ic_noise <6 / 6-8 / 8-9.5, but -0.13 (9.5-10.5), -0.20 (10.5-12), +0.15 (12-15), -0.02 (15-21). Applying
  the clean coefficients or a band-specific cross-fitted OLS makes every band above 9.5 worse, so the gate stays at 9.5.
- Laptop: full-data CNNs finished 00:35 KST (effnetv2-s 6 seeds, test sum 193318.54; ConvNeXt-tiny 3 seeds, test sum
  194022.17); test files in hardness-cache `laptop-gpu-2026-10-08/full6`, `full3` (no OOF). The GPU is held by
  RobotWorldModel again; the user will say in the thread when it is free (mode not chosen yet).
- Plan for the last 2 slots (user: "남은 두 파일은 GPU가속과 그 외 가능한 최대 심사숙고를 통해서 v19 v20만들것"):
  v21 = v19 + v20's in-cell CNN share; v22 = v21 + het for the noisy images from a CNN trained on clean train images
  re-rendered in the noisy-preset style (targets measured on the clean originals, never y), under a pre-registered bar.

## Paused 2026-10-07 20:10 KST for a PC restart - read this first
- The user paused all activity ("이 세션의 모든 활동 임시중단. pc 재시작 후 재개할수 있게 모든 스레드에 준비할것").
  At the pause: no cloud job running, git clean (branch = main = this commit's parent chain), every file below backed up.
  Laptop: no GPU job running (RobotWorldModel held the GPU, 99% / 6 GB with gpu.lock); section 8 g (3 seeds) and h
  (full data) are not started, all inputs cached on the laptop (restorer retrain PSNR 32.01, 1500 restored images,
  restored deg bank, fold-0 screen 13.724 vs 13.635). They need the user's GPU go-ahead (~1-1.5 h) and are worth ~0.015.
- **Submissions for 2026-10-08 (all 5 used on 10-07; count resets 00:00 KST), in this order, each scored before the next:**
  1. `blend_v18.csv` (safe, md5 3e095ae7...): v17 with effnetv2-s alone as the cell CNN; expected ~11.72 (tie with v17).
  2. `blend_v19.csv` (bold, md5 4ce52e76...): v18 + grain-size heterogeneity offset on the 519 low-noise test images.
     Read vs v18: <= v18 - 0.2 -> het is real, extend it to noisier images; >= v18 + 0.2 -> drop it; between -> undecided.
  3. `blend_v20.csv` (optional probe, md5 073ca7d8...): v18 with the cell CNN at a fixed share 0.8; expected ~-0.01,
     public SE ~0.034; <= v18 - 0.07 -> the in-cell CNN premium is large, a dedicated cell CNN is worth GPU time.
  Slots 4-5: no candidate; do not fill them with near-duplicates. Files: `/mnt/project-files/work/hardness-cache/submissions/`
  (also attached in the thread). They are git-ignored; rebuild in the repo with the commands in the v18-v20 bullets below.
- **How to resume:** the user powers on the laptop, opens Claude Code in C:\Daker\microstructure-hardness-prediction,
  re-enables Remote Control, and writes "재개" in the main thread; the laptop runs `git pull` first (branch
  claude/lb-under-10-7080jr). The cloud thread then: restore `data/` from hardness-cache if the container is new
  (open.zip + hardness-cache/data/*.parquet, cnn_cache, experiments-test.tar), check the user's scores against the
  reading rules above, and continue the gap work (het in noisy images is the open problem; null so far: restored-image
  het, learned per-block estimator, grid4 CNN head, extra clean-formula terms).
- **Resumed 21:04 KST; noisy-image het, three more nulls (workflow, pre-registered, train-only choices; hardness-cache
  `scripts/gap2-1007/`):** on the 233 train images with raw ic_noise >= 9.5, on top of blend_v19 nested OOF:
  block ACF length L50 +0.153 [+0.03, +0.29] (perm p 0.98, 2/5 folds), block power-spectrum mean frequency +0.125
  [+0.07, +0.18] (perm p 0.90), smoothed region segmentation for coarse images stopped at fidelity (corr with clean het4
  -0.08 on degraded copies). Lesson: raw fidelity with het4 comes from grain size (het4 vs log N_eff -0.83); judge any
  future estimator by partial corr with het4 given log N (all three ~0), and block ACF length follows block dark-phase
  fraction (0.50) more than block grain size (0.21). Test-image features (acf, spec) are in that folder, extraction only.

## Resumed 2026-10-07 afternoon - read this first
- The user resumed after the move; the laptop is back on the branch with Remote Control in this thread.
- **blend_v16 scored public 11.7830** (user), best so far but only -0.013 vs v14 (11.7960). The nested-OOF simulation of
  300-image subsets expected -0.28 (90% [-0.47, -0.09]); landing at -0.013 or worse has probability ~1%, so one of v16's
  two changes probably does not carry over to public: (a) the cell NNLS (CNN 0.2 -> 0.61 + LightGBM in the fine-noisy
  cell) or (b) the out-of-cell NNLS that drops the CNN (0.2 -> 0) and reweights the feature members. (b) runs against
  the public CNN trend (public has favoured the CNN more than CV since v7).
- **5th submission of 2026-10-07: blend_v17** (`--mode nnls --other share --share-other 0.2 --members <v16 members>`):
  v16 on the 284 cell images, v14 on the other 716; nested CV 12.291 (v14 12.425, v16 12.143). It replaces the
  pre-registered v15 because it splits v16 exactly: in-cell effect = v17 - 11.7960, out-of-cell effect = 11.7830 - v17
  (exact in public MSE). Conditional on v16's score the simulation expects v17 ~11.74 (sd 0.05), P(v17 < v16) 0.78;
  v15 ~11.74 too. One public score moves +-0.07 on differences this size, so read it as a direction:
  v17 <= 11.75 -> the out-of-cell change hurt, keep the CNN outside the cell in later blends;
  v17 >= 11.80 -> the in-cell change did not carry over, v16 stays the reference; in between -> both small, v16 stays.
- **Value of the section-8 GPU run, estimated 2026-10-07 ~14:30 KST** (3 analyses + an adversarial check; scripts and
  reports in hardness-cache `scripts/gpuq-1007/`, REPORTS.md): the best blend with all section-8 outputs is expected to
  beat v17 by only ~0.015 public (P > 0.02 ~30%, P > 0.05 ~10%; public SE of such a change ~0.02). Reasons: 3->6 seeds
  and ConvNeXt left the v17-style blend unchanged (+0.004 / +0.03); inside the fine-noisy cell restored-image features
  are worse than raw ones (ridge +0.33) and do not correct the CNN's errors, so g has ~15% chance of a 0.3 cell gain;
  i (2x, 112 crops) is weakest: cell grains are already ~14 px, the residual is noise, small crops lose context;
  h1/h2 full-data ~-0.005 to -0.01 (fold averaging already holds most of it) and cannot be validated by CV. Simulated
  new members were ~0.024 nested too optimistic (they leak y), calibrate any such simulation on real CNN additions.
  Told the user it is not worth 2-3.5 h of the shared GPU; if run, g only (~1 h), and decide after v17's score
  (v17 >= 11.80 means cell gains do not carry over to public, so a better in-cell CNN likely won't either).
- Still waiting on the user: the GPU line (exclusive or shared mode) for `notes/handoff/laptop-gpu.md` section 8
  (run order g, i, h1, h2; 2-3.5 h). The laptop saw a robot-project process on the GPU after resume. No GPU run is approved.

## Evening of 2026-10-07: gap to LB #1 (8.963) - read this first
- Full reports and scripts: hardness-cache `scripts/gap-1007/` (REPORTS.md; workflow of 9 agents).
- #1 cannot get 8.96 from noisy images alone: clean COARSE images (RMSE 13.7-14.8 even in our cleanest) must improve.
  At least 30-50% of the clean "b/N label noise" is predictable model error (our b ~25k; #1 needs <= 18.7k).
- Found: a within-image grain-size heterogeneity term. het4 = sd over the 16 blocks of a 4x4 grid (64 px) of
  -2 log(mean a^-1/2) of grain areas; q2_fcv (CV of quadrant grain counts) is the simple version. With it, Hall-Petch
  gets the normal sign (finer = harder): the anti-Hall-Petch fit came from omitting it. Gated to clean images (raw
  ic_noise < 7.49): v18 + cross-fitted offset 11.920 (p=0.25), member route (feat6 + unpenalised gate*het4,
  gate*N^0.25, p=0.5, pre-registered) nested 12.045 (-0.230, 5/5). Verification: the 64-px alignment is NOT special
  (non-aligned grids do about as well), full nested grid selection gives -0.232 [-0.51, +0.02], the gain is carried by
  ~10 coarse clean images (median gate error gets worse). Honest effect -0.2 to -0.25 CV. Next: a shift-invariant
  version (averaged over grid offsets/sizes, interior grains) before using it in a bold slot.
- **Follow-up (18:30 KST): the het gain is mostly selection.** A pre-registered shift-invariant version (mean block-sd
  over 249 grids, mirror-corrected border grains) gives only -0.017 nested (hardness-cache `scripts/het-1007/`, inv/).
  With VISIBLE (uncorrected) border-grain areas it returns to -0.18 (post hoc: the label may be computed on the canvas,
  border-cut grains counting as small). 80%+ of every variant's SSE gain sits in ~10 coarse clean images (like v12).
  No blend_v19 was made. Restored images do not measure het in mid images (fidelity corr 0.20; blur merges grains),
  but the clean pipeline (filter sigma 1.0) on RAW images measures het/N well up to ic_noise 9.5 ("lower mid", ~100
  train / ~207 test): those images were never used to choose het4, so they are the out-of-sample test (running).
- **Out-of-sample check (19:00 KST)** on the 53 lower-mid train images (raw ic_noise 8.0-9.5) never used to choose het4,
  coefficients fitted on clean images only: RMSE 11.385 -> 10.119 (-1.27, 90% CI [-2.97, +0.60], 4/5 folds, median
  |e| 6.76 -> 5.24, 33/53 better), but the top 5 images carry the gain: fails the pre-registered bar (scripts in
  hardness-cache `scripts/conf-1007/`). Still handed over as a BOLD slot: **blend_v19** = v18 + offset on images with
  raw ic_noise < 9.5 (`python -m src.het_blocks; python -m src.het_offset --base blend_v18 --out blend_v19`), nested
  11.754 (selection-inflated), test offset sd 5.0 on 519 images, public SE vs v18 ~0.21, so one score can only show
  a big effect: v19 <= v18 - 0.2 -> het is real, extend it (noisy images, member route); >= v18 + 0.2 -> drop it.
- **Grid-aware CNN screen (19:10 KST), null:** `src.train_cnn --pool grid4` (crop 256, mean+sd over the fixed 4x4 grid of
  64-px cells), resnet18 CPU folds 0-1: 13.665 vs 13.743 for avg at crop 256 (-0.08, CI [-0.68, +0.55]); residuals of
  both still correlate +0.45 with het4 on low-noise images, so the head does not learn the grid term (details in
  notes/handoff/cnn-trainer.md). A GPU effnetv2-s version is not expected to change that.
- **blend_v20** (optional third slot for 2026-10-08): v18 with the cell's effnetv2-s at a fixed share 0.8
  (`--cell-share 0.8`), nested 12.277, public SE vs v18 ~0.034, calibration expects ~-0.01. v20 <= v18 - 0.07 would mean
  the in-cell CNN premium is large (then a dedicated cell CNN is worth GPU time).
- Noisy images: noise/blur destroy little information; the noisy preset compresses dark-phase contrast by ~half
  (offset -8.7 vs -17.5 grey), which is what loses the signal. Noise-robust measurement families (ACF length, histogram
  deconvolution, dark-tail pores, structure tensor) gave no significant gain. A learned grid het estimator: null in
  noisy, small real signal in mid (-0.05 to -0.10, not grid-specific).

## Afternoon of 2026-10-07: v17 scored, file for 2026-10-08 00:00 KST
- **blend_v17 scored public 11.7285** (user ~15:04 KST), best so far. Exact decomposition: in-cell change (v16 cell)
  -0.0675 public (CV -0.134), out-of-cell change of v16 (no CNN, restored ridge + embedding) +0.0545 (CV -0.148).
  Leaderboard #1 at 15:05 KST: **8.9630**, so the old "perfect model ~10.5" estimate is wrong; a gap investigation is next.
- Public calibration model (scratchpad `v18/calib/predict_public.py`, backed up in hardness-cache `scripts/v18-1007/`):
  public MSE(p) = c + mean_test[(p - t)^2], t = B + kappa (C - B), B = OOF-NNLS cell/outside blend (= v16 on test),
  C = CNN mix; kappa 0.32 [0.10, 0.54]; leave-one-submission-out error 0.053 vs 0.136 for "CV - 0.6". A "feature
  selection optimism" model fits worse. Implied: the CNN mix alone would score ~12.08 public (OOF 13.16).
  Candidates that are linear mixes of v7/v14/v16/v17 (all v14-outside share variants, v16-base + CNN outside) have
  their public score already pinned by the existing scores (+-0.004-0.015): submitting them teaches little.
  Selected feature-side gains carry over to public at about 0 (-0.43 +- 0.35); CNN share earns a public premium.
- **File for 2026-10-08 00:00 KST: blend_v18** (`--cnn-nnls cnn_ev2s_rawnlm_degcons_gpu_s6:1`, nested CV 12.2752, -0.016
  vs v17 in 4/5 folds; only the 284 cell images change). Expected about -0.01 vs v17 (a tie). Plan for the other 4 slots:
  the user wants only 1-2 safe submissions per day (said 16:12 KST), so v18 is the safe one and the other 4 go to bold
  attempts whose public score is not already pinned: gap-investigation features/models first, in-cell CNN share 0.8-1.0,
  the restored-channel CNN if the user grants the GPU tonight. Fallbacks in
  scratchpad `v18/cands/` (`cmb_Iev2s_Oc` = cell as v18, outside 50/50 v14-outside and v16-base + CNN 0.2, nested
  12.230 but outside reweighting is the kind public punished; `in_Iev2s_sh80` = more CNN in the cell, CV flat).
- GPU section 8: judged not needed now (restored-channel CNN g ~+0.013 expected, P(>0.03) ~0.2; h1 full-data ~+0.009,
  not CV-checkable). The laptop finished the CPU prep (8b, banks match the cloud: 1953 pairs, rms 2.23); GPU steps wait
  for the user's mode (exclusive = user stops the robot job; shared = user allows gpu_turn.py in the PC's /permissions).

## Morning of 2026-10-07 (read this first)
- **blend_v11 scored public 11.7997** (user, ~04:30 KST), the new best. CNN share 0 / 0.07 / 0.131 gave public
  11.8795 / 11.8267 / 11.7997 (v7 / v10 / v11), each time about CV - 0.55 to 0.61. The CNN helps public more than CV
  says: the CV-optimal share is 0.131, a parabola through the three public points has its minimum near 0.19.
- **blend_v12 scored public 11.9094** (user, 04:25 KST), worse than v10 (11.8267) and about v7 (11.8795), although its
  nested CV was 0.26 better (public only CV - 0.25). Not a bug: own columns are applied to the 333 high-noise test
  images exactly as to train (feat7 - feat6 differences have sd 5.2 on both). On OOF the v12 - v10 direction correlates
  -0.21 with v10's errors; on public about -0.03. **Cause unknown.** The OOF gain sat in about 10 images (73% of it),
  but that does not explain the miss: v4 -> v6 (top-10 share 0.79) and v6 -> v7 were as concentrated and carried over.
  Candidates: overfitting that 5-fold CV cannot see (the member was picked among several own-slope variants on the
  same folds), or luck on the ~300 public images. Keep own-slope members out of submissions until one of these is
  settled. blend_v13 (also uses that member) stays shelved.
- **blend_v14 scored public 11.7960** (user, ~04:39 KST), a tie with v11: the global CNN share is at its public optimum
  (a parabola through v7/v10/v11/v14 bottoms at share 0.18, 11.794). Do not push the global share further.
- **Next file: blend_v15** (`python -m src.blend_cells ... --share-other 0.2 --share-cell 0.7 --out blend_v15`, nested CV
  12.327 vs v14 12.425, 4/5 folds): v14 with CNN share 0.7 only for fine-grained AND noisy images (log calibrated grain
  count density and raw ic_noise above their train medians; 132 train / 284 test images). On OOF the CNN mix beats the
  base there (11.33 vs 12.04) and loses everywhere else (13.76 vs 12.56; coarse or clean images get LS share <= 0). All
  25 cut combinations (train quantiles 0.33-0.67) improve on v14; the change correlates -0.23 with v12's failed change.
  Public SE of the v15 - v14 difference is about 0.06. Smooth gates (share = clip(a + b.grain + c.noise)) reach nested
  12.28 but switch steeply between 0 and 1 and move test predictions twice as much (sd 2.0): kept as a later option.
- **blend_v16** (`python -m src.blend_cells --cnn $C --mode nnls --members feat6_rest_ridge_all,feat5_v3cal_ridge_het_spat_v4_milspl,
  feat3_v23cal_lgbs_het_spat,feat2_v23cal_lgbs_hetN,emb_effv2s_256_gridge3_noise_cs24 --out blend_v16`, nested CV **12.143**):
  separate NNLS weights in the fine-noisy cell (CNN mix 0.61 + LightGBM 0.39) and elsewhere (restored ridge 0.59 +
  embedding head 0.38 + LightGBM 0.03, no CNN). Half of the gain is outside the cell (the global weights were a
  compromise: LightGBM is the best feature model in the cell, 11.57, but poor elsewhere). Weights are stable across
  folds; re-discovering the split inside each outer fold (median cuts, CNN-share rule) picks the same cell in all 5
  folds (12.145); random 132-image cells do not help (+0.03). Offered to the user as the 4th submission instead of
  v15 (or as the 5th if v15 was already in). Not worth it: 2x2 or tercile regimes (overfit), bigger member pools per
  regime (-0.02 to -0.07, noisy; regime 1 then picks old members such as feat2_v3_lgb).
  The ensembler's covariate stacker (null on 2026-10-06, before the GPU CNNs) gives only -0.06 to -0.09 on these
  members: its linear/tercile forms cannot express the fine AND noisy cell.
- Where blend_v16 still loses (nested OOF RMSE by calibrated grain-count tercile x raw ic_noise tercile): coarse-noisy
  18.2 (every member 18.2-20), mid-noisy 13.9, coarse-midnoise 14.2; clean cells 7.9-13.0. In the noisy third the
  predictions are too flat (slope of y on prediction 1.21) and dark-fraction measures correlate about 0.55 with
  hardness vs 0.71 on clean images: the loss is measurement under heavy blur + white Gaussian noise (sd 16-21).
- Null on top of v16 (2026-10-07 morning): degraded copies of clean train images as extra labelled rows (ridge/LGBM
  worse), multi-scale GMM/threshold dark fractions (cal_ features stay best in noisy images), a noise-dependent stretch
  (-0.035, 3/5 folds) or global stretch (-0.019), absolute elongation direction (|rho| < 0.04 with the residual).
- The train-vs-test adversarial AUC of about 0.6 comes only from the 174 calibration-source train images:
  `src.features.cal_apply` fits them in-sample (identity rows), so their cal_ values differ from comparable test images
  (AUC 0.72; other train images vs test 0.54). It does not matter for predictions: validating with cross-fitted cal_
  values changes member CV by -0.01 to +0.03, and retraining on a cross-fitted table is not better (scratchpad
  `calxf/`). v4/v5 calibrations were already cross-fitted.
- CNN prep committed (5e51ca1, cnn-trainer): `--input raw+nlm+rest` (restored channel), `--full` (all-train models,
  test predictions only), `--scale S`; `src.restore --device cuda`. Laptop plan: `notes/handoff/laptop-gpu.md`
  section 8, run order g (restored-channel effnetv2-s, 3 seeds), i (`--scale 2 --crop 112`, 3 seeds), then h1/h2
  (full-data effnetv2-s x6, ConvNeXt x3). Asked the user ~05:55 KST for the GPU mode; nothing runs until they write
  the line in the thread. Judge new CNNs by the fine_noisy cell RMSE (ev2s_s6 11.26) and by v16-style blends.
- Full refit on all 500 train images (one model instead of the mean of 5 fold models) is worth little for the ridge
  members: one model on 400 rows vs the mean of 5 inner models on 320 rows gives -0.035 (feat6_rest_ridge_all) and
  -0.001 (feat5_v3cal_ridge_het_spat_v4_milspl). LightGBM members and the embedding head are being checked
  (scratchpad `refit/`, copies in hardness-cache `scripts/refit/`).
- Averaging gain of the CNN test predictions (mean of 5 fold models vs one fold model on OOF): from the CPU resnet18
  caches only 2-5% of the OOF MSE, too small to explain why public prefers a CNN share of 0.18 vs the CV fit 0.10.
- Next GPU candidate: a CNN with a restored-image input channel (cnn-trainer prepares the code and laptop steps,
  `notes/handoff/laptop-gpu.md` section 8). A GPU run needs the user's say-so and mode in the thread first.

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
tar -xf /mnt/project-files/work/hardness-cache/het_cnn-tables.tar   # G1 het reader tables data/het_cnn/<run>/*.parquet (laptop readers)
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
