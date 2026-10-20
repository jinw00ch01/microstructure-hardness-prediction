# Env for the d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39p7b (P7B = [B8f, APf, NS7Bf],
# NEAR-TIE(strict)) was handed over as 10-18 file 1 (2026-10-18 05:20 KST; d1018/DECISION_neartie1018.txt). Written 2026-10-18 05:20 KST
# by the main thread: = d1017/run_env_1018.sh (best v38q8a 9.9495571078, kriging24, line A BASE = [B8f, APf]) plus v39p7b in
# SUBMITTED_EXTRA, so the skip rule (G1-row rms >= 0.25 vs every submitted or handed-over file) covers P7B. Nothing else changes.
# Until P7B's public reading arrives the best file stays v38q8a and the public model stays kriging24; after the reading, a new env.
# usage: source run_env_1018p.sh v38q8a 9.9495571078   (set BASE=B8f after sourcing for line B)
source /mnt/project-files/work/hardness-cache/scripts/d1017/run_env_1018.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v39p7b:e226678a487e1d2903eea3493d5144af
echo "env (1018p): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v39p7b handed over)"
