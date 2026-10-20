# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v40b8all
# (10-19 file 1, g1mix11 q8a4b_k29 lineB PALL) read 9.9465055964 (user, main thread 2026-10-19 00:35 KST; worse than the best
# blend_v39q8a4b 9.9434377740 by +0.0610 public MSE; kriging29 had predicted 9.9460 (z +0.04), the full predictive 9.9453 (z +0.24);
# pub/read30_out.txt). Written 2026-10-19 00:39 KST by the main thread: = d1018/run_env_1018q4bh.sh (best v39q8a4b, 25 files in the
# skip list, v40b8all included) with KRIG_LATEST -> kriging30 (= kriging29 + the v40b8all reading). Best unchanged.
# usage: source run_env_1019k30.sh v39q8a4b 9.9434377740   (then set BASE per line, as in run_env_1018q4b.sh)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018q4bh.sh "$@" >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging30.py
echo "env (1019k30): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v40b8all read 9.9465; 25 files in the skip list)"
