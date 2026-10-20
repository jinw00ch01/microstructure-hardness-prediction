#!/bin/bash
# Rebuild the whole SAFE/BOLD pool of d1018/ADDENDUM_pool1019.txt under one env and run both judges (main thread, 2026-10-18 14:46 KST).
# Same builders, BASEs and order as the q3a6_k28 / q3a6_ap4b rebuilds; no bar, option, weight, step or seed changes. No ALLOW_PARTIAL.
# usage (from anywhere): bash rebuild_pool1019.sh <env.sh> <best name> <best pub> <tag> [KRIG for the FP judge, default = the env's KRIG_LATEST]
# writes  d1018/g1mix11/<tag>/{lineB,lineAq8ab3,lineAq8a}, d1017/g1mix10/<tag>/{lineB,lineAq8a}, d1017/g1mix9/<tag>/{lineB,lineAq8a},
#         d1016/g1mix8/best_<tag>, and d1018/g1mix11/<tag>/judge_slots_pool.txt + judge_bold_fp_pool.txt.
# A line whose BASE a builder cannot take stops at the builder's own BASE assert (rc != 0, reported); the other lines still build.
set -u
ENV=$1; BN=$2; BP=$3; TAG=$4
REPO=/home/claude/microstructure-hardness-prediction
S=/mnt/project-files/work/hardness-cache/scripts
G11=$S/d1018/g1mix11; G10=$S/d1017/g1mix10; G9=$S/d1017/g1mix9; G7=$S/d1016/g1mix7; G8=$S/d1016/g1mix8
for d in $G11/$TAG $G10/$TAG $G9/$TAG $G8/best_$TAG; do [ -e $d ] && { echo "exists: $d (pick a new tag)"; exit 1; }; done
mkdir -p $G11/$TAG/lineB $G11/$TAG/lineAq8ab3 $G11/$TAG/lineAq8a $G10/$TAG/lineB $G10/$TAG/lineAq8a $G9/$TAG/lineB $G9/$TAG/lineAq8a $G8/best_$TAG
run() {  # run <builder> <BASE> <OUT>
  ( cd $REPO && source $ENV $BN $BP >/dev/null && export BASE=$2 && unset ALLOW_PARTIAL && \
    OUT=$3 nice -n 5 python3 -B $1 > $3/build.log 2>&1; echo "rc $? $3" )
}
run $G11/build_g1mix11.py B8f            $G11/$TAG/lineB &
run $G11/build_g1mix11.py B8f,APBf,NS3Bf $G11/$TAG/lineAq8ab3 &
run $G11/build_g1mix11.py B8f,APf        $G11/$TAG/lineAq8a &
run $G10/build_g1mix10.py B8f            $G10/$TAG/lineB &
wait
run $G10/build_g1mix10.py B8f,APf        $G10/$TAG/lineAq8a &
run $G9/build_g1mix9.py   B8f            $G9/$TAG/lineB &
run $G9/build_g1mix9.py   B8f,APf        $G9/$TAG/lineAq8a &
run $G7/build_g1mix7.py   B8f            $G8/best_$TAG &
wait
for d in $G11/$TAG/lineB $G11/$TAG/lineAq8ab3 $G11/$TAG/lineAq8a $G10/$TAG/lineB $G10/$TAG/lineAq8a $G9/$TAG/lineB $G9/$TAG/lineAq8a $G8/best_$TAG; do
  printf '%s | %s | %s\n' "${d#$S/}" "$(grep -m1 -o 'BEST reference reproduces [a-z0-9_]* bit for bit' $d/build.log || echo 'NO BEST LINE')" "$(grep -m1 '^latest kriging' $d/build.log)"
done
KRIG=${5:-$(cd $REPO && source $ENV $BN $BP >/dev/null && echo $KRIG_LATEST)}
POOL="$G11/$TAG/lineB $G11/$TAG/lineAq8ab3 $G11/$TAG/lineAq8a $G10/$TAG/lineB $G10/$TAG/lineAq8a $G9/$TAG/lineB $G9/$TAG/lineAq8a $G8/best_$TAG"
cd /tmp && python3 -B $S/d1018/judge_slots1018.py $POOL > $G11/$TAG/judge_slots_pool.txt 2>&1; echo "judge_slots rc $?"
cd /tmp && KRIG=$KRIG python3 -B $S/d1018/judge_bold_fp1018.py $POOL > $G11/$TAG/judge_bold_fp_pool.txt 2>&1; echo "judge_bold_fp rc $? (KRIG=$KRIG)"
