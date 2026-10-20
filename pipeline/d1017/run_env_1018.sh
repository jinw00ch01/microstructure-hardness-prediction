# Env for d1016/g1mix7/build_g1mix7.py (the g1mix8 line: BASE=B8f) and the d1017/g1mix9 and g1mix10 builders after blend_v38q8a (Q8A =
# [B8f, APf]) read 9.9495571078 and became the best public file (user, main thread 2026-10-17 23:42 KST). Written 2026-10-17 23:44 KST
# by the main thread from d1016/g1mix8/run_env_extra2_k23.sh: KRIG_LATEST -> kriging24 (= kriging23 + the v38q8a reading), v38q8a
# added to SUBMITTED_EXTRA, and the v38q8a best case. BEST_RUNS = the runs_of([B8f, APf]) order of build_g1mix7.py (3 + 3 runs, no
# repeats); it reproduced blend_v38q8a bit for bit in a scratch build at 23:28 KST (before the reading).
# usage: source run_env_1018.sh v38q8a 9.9495571078   (BASE = the best file's families; set BASE=B8f after sourcing for line B / g1mix8)
source /mnt/project-files/work/hardness-cache/scripts/d1016/g1mix7/run_env_1017.sh v38xn8 9.9564798521 >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging24.py
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v38q86:8c5bba8b1d955f8c5355e91d52674957,v38q8all:0885bff85aaee60516f0c56e5e09f35d,v38q8a:78a31699cf40a8dcfdb5a29b0483afb2
case "${1:-v38q8a}" in
  v38q8a) B8=b5ns_locK_fsim; AP=b5ap_locK_real5
          export BASE=B8f,APf BEST_NAME=v38q8a BEST_PUB="$2" BEST_RUNS=$B8,${B8}_s1,${B8}_s2,$AP,${AP}_s1,${AP}_s2 ;;
  *) echo "unknown best $1"; return 1 ;;
esac
echo "env (1018): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST"
