# Env for the d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q8ab3 (Q8AB3 = [B8f, APBf, NS3Bf],
# g1mix10 line B PALL; 10-18 BOLD 2 of 2) was handed over (2026-10-18 07:56 KST; d1017/g1mix10/q8a_k26/README.txt). Written 2026-10-18
# 07:55 KST by the main thread: = d1018/run_env_1018k26.sh (best v38q8a 9.9495571078, kriging26, line A BASE = [B8f, APf], 21 files in
# the skip list) plus v39q8ab3 in SUBMITTED_EXTRA, so the skip rule (G1-row rms >= 0.25 vs every submitted or handed-over file) covers
# Q8AB3 (22 files). Use run_env_1018k26.sh instead only for a NOTE_slate1018 rule-4 re-judge before the user has submitted v39q8ab3.
# After the v39q8ab3 reading: mk_pubN.py 27 -> kriging27 and a new env.
# usage: source run_env_1018k26q.sh v38q8a 9.9495571078   (set BASE=B8f after sourcing for line B)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018k26.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v39q8ab3:c05736735286c14265489100de6f81a3
echo "env (1018k26q): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v39q8ab3 handed over)"
