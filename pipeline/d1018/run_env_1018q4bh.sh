# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v40b8all
# (g1mix11 q8a4b_k29 lineB PALL = [B8f, AP6f, AP6Bf, AP4f, AP4Bf]; 10-19 file 1, BOLD by full-predictive P(beat best), user card
# 12:34 KST 10-18) was staged (2026-10-18 15:50 KST) and handed over in the reply after this file, to be submitted after 00:00 KST 10-19
# (d1018/g1mix11/q8a4b_k29/README.txt; verification d1018/verify_1019_file1/, critic DELIVER). Written 2026-10-18 15:50 KST by the main
# thread: = d1018/run_env_1018q4b.sh (best v39q8a4b 9.9434377740, kriging29, 24 files in the skip list) plus v40b8all in
# SUBMITTED_EXTRA, so the skip rule (G1-row rms >= 0.25 vs every submitted or handed-over file) covers it (25 files; it makes g1mix11
# lineAq8a PALL, G1 rms 0.154, SKIP). After the v40b8all reading: mk_pubN.py 30 -> kriging30 and a new env (with a v40b8all best case
# if it beats 9.9434377740).
# usage: source run_env_1018q4bh.sh v39q8a4b 9.9434377740   (then set BASE per line, as in run_env_1018q4b.sh)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018q4b.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v40b8all:adb8dcd40ba3ef506f74b888b94cdb1f
echo "env (1018q4bh): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v40b8all handed over; 25 files in the skip list)"
