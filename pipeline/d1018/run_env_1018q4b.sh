# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q8a4b
# (Q8A4B = [B8f, APf, AP4Bf], g1mix11 lineAq8a P4AB; 10-18 file 5, the 3rd BOLD of 10-18) read 9.9434377740 (user, main thread
# 2026-10-18 14:48 KST) and became the best public file (-0.0173 public MSE vs Q3A6 9.9443061756, -0.1217 vs Q8A; kriging28 had
# predicted 9.9469, z -0.20; full predictive 9.9462, z -0.26; pub/read29_out.txt). Written 2026-10-18 14:50 KST by the main thread:
# = d1018/run_env_1018a6b.sh (24 submitted or handed-over files in the skip list, v39q8a4b included) with KRIG_LATEST -> kriging29
# (= kriging28 + the v39q8a4b reading) and the v39q8a4b best case. BEST_RUNS = the runs_of([B8f, APf, AP4Bf]) order of
# build_g1mix11.py (3 + 3 + 3 runs, no repeats), the rule g1mix11 lineAq8a P4AB used to build blend_v39q8a4b; the builders assert it
# reproduces the file bit for bit.
# Line A of the new best (BASE = B8f,APf,AP4Bf) cannot be built: AP4Bf is a g1mix11 new family and unknown to g1mix7..g1mix10, so every
# builder's BASE assert stops it (the default BASE below fails safe). Pool per d1018/ADDENDUM_pool1019.txt points 1-2 (unchanged):
# set after sourcing  BASE=B8f (line B) | BASE=B8f,APBf,NS3Bf (lineAq8ab3, g1mix11 only) | BASE=B8f,APf (lineAq8a).
# usage: source run_env_1018q4b.sh v39q8a4b 9.9434377740
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018a6b.sh v39q3a6 9.9443061756 >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging29.py
case "${1:-v39q8a4b}" in
  v39q8a4b) B8=b5ns_locK_fsim; AP=b5ap_locK_real5; A4B=b4ap_locK_fsim
            export BASE=B8f,APf,AP4Bf BEST_NAME=v39q8a4b BEST_PUB="$2" \
                   BEST_RUNS=$B8,${B8}_s1,${B8}_s2,$AP,${AP}_s1,${AP}_s2,$A4B,${A4B}_s1,${A4B}_s2 ;;
  *) echo "unknown best $1"; return 1 ;;
esac
echo "env (1018q4b): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v39q8a4b read 9.9434, new best; set BASE per line)"
