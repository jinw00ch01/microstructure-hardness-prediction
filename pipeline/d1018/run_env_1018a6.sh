# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q3a6
# (Q3A6 = [B8f, APBf, NS3Bf, AP6f], g1mix11 line A P6A = PALL; 10-18 file 4, SAFE) read 9.9443061756 (user, main thread 2026-10-18
# 10:58 KST) and became the best public file (-0.0947 public MSE vs Q8AB3 9.9490659450; kriging27 had predicted 9.9464, z -0.15;
# pub/read28_out.txt). Written 2026-10-18 11:00 KST by the main thread: = d1018/run_env_1018q3q.sh (23 submitted or handed-over files
# in the skip list, v39q3a6 included) with KRIG_LATEST -> kriging28 (= kriging27 + the v39q3a6 reading) and the v39q3a6 best case.
# BEST_RUNS = the runs_of([B8f, APBf, NS3Bf, AP6f]) order of build_g1mix11.py (3 + 3 + 3 + 3 runs, no repeats), the rule g1mix11
# line A P6A used to build blend_v39q3a6; the builders assert it reproduces the file bit for bit.
# Line A of the new best (BASE = B8f,APBf,NS3Bf,AP6f) cannot be built: AP6f is a g1mix11 new family and unknown to g1mix7..g1mix10,
# so every builder's BASE assert stops it (the default BASE below fails safe). Pool per d1018/ADDENDUM_pool1019.txt points 1-2:
# set after sourcing  BASE=B8f (line B) | BASE=B8f,APBf,NS3Bf (lineAq8ab3, g1mix11 only) | BASE=B8f,APf (lineAq8a).
# usage: source run_env_1018a6.sh v39q3a6 9.9443061756
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018q3q.sh v39q8ab3 9.9490659450 >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging28.py
case "${1:-v39q3a6}" in
  v39q3a6) B8=b5ns_locK_fsim; APB=b5ap_locK_fsim; N3B=b3ns_locK_fsim; A6=b6ap_locK_real5
           export BASE=B8f,APBf,NS3Bf,AP6f BEST_NAME=v39q3a6 BEST_PUB="$2" \
                  BEST_RUNS=$B8,${B8}_s1,${B8}_s2,$APB,${APB}_s1,${APB}_s2,$N3B,${N3B}_s1,${N3B}_s2,$A6,${A6}_s1,${A6}_s2 ;;
  *) echo "unknown best $1"; return 1 ;;
esac
echo "env (1018a6): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v39q3a6 read 9.9443, new best; set BASE per line)"
