# Env for the d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q8ab (Q8AB = [B8f, APBf], 10-18
# file 2, BOLD 1 of 2) read 9.9572104905 (user, main thread 2026-10-18 07:25 KST; worse than Q8A by +0.1524 public MSE, kriging25 had
# predicted 9.9568, z +0.02; pub/read26_out.txt). Written 2026-10-18 07:27 KST by the main thread: = d1018/run_env_1018k25q.sh (best
# v38q8a 9.9495571078, line A BASE = [B8f, APf], v39p7b and v39q8ab in SUBMITTED_EXTRA, 21 files) with KRIG_LATEST -> kriging26
# (= kriging25 + the v39q8ab reading). Best file unchanged.
# usage: source run_env_1018k26.sh v38q8a 9.9495571078   (set BASE=B8f after sourcing for line B)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018k25q.sh "$@" >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging26.py
echo "env (1018k26): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v39q8ab read 9.9572)"
