# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q3a6
# (Q3A6 = [B8f, APBf, NS3Bf, AP6f], g1mix11 line A P6A = PALL; 10-18 file 4, SAFE) was handed over (2026-10-18 10:26 KST;
# d1018/g1mix11/q8ab3_k27/README.txt). Written 2026-10-18 10:27 KST by the main thread: = d1018/run_env_1018q3.sh (best v39q8ab3
# 9.9490659450, kriging27, line A BASE = [B8f, APBf, NS3Bf], 22 files in the skip list) plus v39q3a6 in SUBMITTED_EXTRA, so the skip
# rule (G1-row rms >= 0.25 vs every submitted or handed-over file) covers Q3A6 (23 files). Use run_env_1018q3.sh instead only for a
# NOTE_slate1018 rule-4 re-judge before the user has submitted v39q3a6. After the v39q3a6 reading: mk_pubN.py 28 -> kriging28 and a
# new env (with a v39q3a6 best case if it beats 9.9490659450).
# usage: source run_env_1018q3q.sh v39q8ab3 9.9490659450   (set BASE=B8f after sourcing for line B, or BASE=B8f,APf for the earlier
#        Q8A line A kept in the pool)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018q3.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v39q3a6:dfa00a7887b2d62f2473be61fe6d3072
echo "env (1018q3q): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v39q3a6 handed over)"
