# Env for the d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q8ab (Q8AB = [B8f, APBf], 10-18
# BOLD 1 of 2) was handed over (2026-10-18 06:18 KST; d1017/g1mix10/q8a_apb3/README.txt). Written 2026-10-18 06:18 KST by the main
# thread: = d1018/run_env_1018k25.sh (best v38q8a 9.9495571078, kriging25, line A BASE = [B8f, APf], v39p7b in SUBMITTED_EXTRA) plus
# v39q8ab in SUBMITTED_EXTRA, so the skip rule (G1-row rms >= 0.25 vs every submitted or handed-over file) covers Q8AB (21 files).
# Use run_env_1018k25.sh instead only for a NOTE_slate1018 rule-4 re-judge before the user has submitted v39q8ab.
# After the v39q8ab reading: mk_pubN.py 26 -> kriging26 and a new env.
# usage: source run_env_1018k25q.sh v38q8a 9.9495571078   (set BASE=B8f after sourcing for line B)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018k25.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v39q8ab:aab27fb4fb477bcf60874a8dba5c4c79
echo "env (1018k25q): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v39q8ab handed over)"
