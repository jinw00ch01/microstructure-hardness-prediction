# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q8a4b
# (Q8A4B = [B8f, APf, AP4Bf], g1mix11 lineAq8a P4AB; 10-18 file 5, the 3rd BOLD of 10-18 by full-predictive P(beat best), user cards
# 11:04 and 12:34 KST) was handed over (2026-10-18 14:40 KST; d1018/g1mix11/q3a6_ap4b/README.txt, verification d1018/verify_p4ab_1018/).
# Written 2026-10-18 14:38 KST by the main thread: = d1018/run_env_1018a6.sh (best v39q3a6 9.9443061756, kriging28, 23 files in the
# skip list) plus v39q8a4b in SUBMITTED_EXTRA, so the skip rule (G1-row rms >= 0.25 vs every submitted or handed-over file) covers it
# (24 files). After the v39q8a4b reading: mk_pubN.py 29 -> kriging29 and a new env (with a v39q8a4b best case if it beats 9.9443061756).
# usage: source run_env_1018a6b.sh v39q3a6 9.9443061756   (then set BASE=B8f | B8f,APBf,NS3Bf | B8f,APf per line, as in run_env_1018a6.sh)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018a6.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v39q8a4b:cfe4f97c3877613696e68d02c4131a6e
echo "env (1018a6b): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v39q8a4b handed over)"
