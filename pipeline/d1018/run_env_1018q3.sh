# Env for the d1016/g1mix7 (g1mix8 line), d1017/g1mix9, d1017/g1mix10, d1018/g1mix11 and d1018/g1mix12 builders after blend_v39q8ab3
# (Q8AB3 = [B8f, APBf, NS3Bf], g1mix10 line B PALL; 10-18 file 3, BOLD 2 of 2) read 9.9490659450 (user, main thread 2026-10-18 09:51
# KST) and became the best public file (-0.0098 public MSE vs Q8A 9.9495571078; kriging26 had predicted 9.9623, z -0.53;
# pub/read27_out.txt). Written 2026-10-18 09:55 KST by the main thread: = d1018/run_env_1018k26q.sh (22 submitted or handed-over files
# in the skip list, v39q8ab3 included) with KRIG_LATEST -> kriging27 (= kriging26 + the v39q8ab3 reading) and the v39q8ab3 best case.
# BEST_RUNS = the runs_of([B8f, APBf, NS3Bf]) order of build_g1mix10.py / build_g1mix11.py (3 + 3 + 3 runs, no repeats), the rule
# g1mix10 line B PALL used to build blend_v39q8ab3; the builders assert it reproduces the file bit for bit.
# usage: source run_env_1018q3.sh v39q8ab3 9.9490659450   (line A BASE = [B8f, APBf, NS3Bf]; set BASE=B8f after sourcing for line B,
#        or BASE=B8f,APf for the earlier Q8A line A kept in the pool, NOTE_slate1018 rule 2(d))
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1018k26q.sh v38q8a 9.9495571078 >/dev/null || return 1
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging27.py
case "${1:-v39q8ab3}" in
  v39q8ab3) B8=b5ns_locK_fsim; APB=b5ap_locK_fsim; N3B=b3ns_locK_fsim
            export BASE=B8f,APBf,NS3Bf BEST_NAME=v39q8ab3 BEST_PUB="$2" \
                   BEST_RUNS=$B8,${B8}_s1,${B8}_s2,$APB,${APB}_s1,${APB}_s2,$N3B,${N3B}_s1,${N3B}_s2 ;;
  *) echo "unknown best $1"; return 1 ;;
esac
echo "env (1018q3): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (v39q8ab3 read 9.9491, new best)"
