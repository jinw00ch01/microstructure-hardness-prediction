# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v40b8a4
# (10-19 file 2 = g1mix11 q8a4b_k30 lineB P4A = [B8f, AP4f]) read 9.9433362872 (user, main thread 2026-10-19 01:08 KST) and became the
# best public file (-0.0020 public MSE vs v39q8a4b 9.9434377740; kriging30 had 9.9502, full predictive 9.9494, z -0.84); blend_v40q8an4b
# (file 3) read 9.9572738847 (+0.2753 vs v39q8a4b). pub/read31_out.txt. Written 2026-10-19 01:09 KST by the main thread:
# = d1018/run_env_1019k30h.sh (27 submitted or handed-over files in the skip list) with KRIG_LATEST -> kriging31 and the v40b8a4 best case.
# BEST_RUNS = the runs_of([B8f, AP4f]) order of build_g1mix11.py (3 + 3 runs), the rule g1mix11 lineB P4A used; the builders assert
# it reproduces the file bit for bit. Line A of the new best (BASE = B8f,AP4f) builds only in g1mix12 (AP4f is a g1mix11 new family).
# Pool per d1018/ADDENDUM_pool1019.txt: set after sourcing BASE=B8f (line B) | B8f,APBf,NS3Bf (lineAq8ab3) | B8f,APf (lineAq8a).
# usage: source run_env_1019b4.sh v40b8a4 9.9433362872
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1019k30h.sh v39q8a4b 9.9434377740 >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging31.py
case "${1:-v40b8a4}" in
  v40b8a4) B8=b5ns_locK_fsim; A4=b4ap_locK_real5
           export BASE=B8f,AP4f BEST_NAME=v40b8a4 BEST_PUB="$2" BEST_RUNS=$B8,${B8}_s1,${B8}_s2,$A4,${A4}_s1,${A4}_s2 ;;
  *) echo "unknown best $1"; return 1 ;;
esac
echo "env (1019b4): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v40b8a4 read 9.9433, new best; set BASE per line)"
