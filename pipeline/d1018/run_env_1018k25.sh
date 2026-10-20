# Env for the d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39p7b (P7B, 10-18 file 1) read
# 9.9575524935 (user, main thread 2026-10-18 05:24 KST; worse than Q8A by +0.1592 public MSE, kriging24 had predicted 9.9572, z +0.02).
# Written 2026-10-18 05:27 KST by the main thread: = d1018/run_env_1018p.sh (best v38q8a 9.9495571078, line A BASE = [B8f, APf],
# v39p7b in SUBMITTED_EXTRA) with KRIG_LATEST -> kriging25 (= kriging24 + the v39p7b reading; pub/read25_out.txt). Best file unchanged.
# usage: source run_env_1018k25.sh v38q8a 9.9495571078   (set BASE=B8f after sourcing for line B)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018p.sh "$@" >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging25.py
echo "env (1018k25): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v39p7b read 9.9576)"
