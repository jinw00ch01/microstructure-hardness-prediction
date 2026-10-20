# Env after 10-19 files 2 and 3 were handed over together (main thread, 2026-10-19 00:57 KST): blend_v40b8a4 (g1mix11 q8a4b_k30 lineB P4A
# = [B8f, AP4f]) and blend_v40q8an4b (g1mix9 q8a4b_k30 lineAq8a P4B = [B8f, APf, NS4Bf]); verification d1018/verify_1019_files23/ (critic
# DELIVER for both); decision d1018/DECISION_slots1019.txt. = d1018/run_env_1019k30.sh (best v39q8a4b 9.9434377740, kriging30) plus both
# files in SUBMITTED_EXTRA (27 files in the skip list). After their readings: mk_pubN.py 31 (and 32), a new env (with a new best case if
# one of them beats 9.9434377740), rebuild_pool1019.sh with a new tag.
# usage: source run_env_1019k30h.sh v39q8a4b 9.9434377740   (then set BASE per line)
source /mnt/project-files/work/hardness-cache/scripts/d1018/run_env_1019k30.sh "$@" >/dev/null || return 1
export SUBMITTED_EXTRA=$SUBMITTED_EXTRA,v40b8a4:eac611155489d3ab70018297f5b694ce,v40q8an4b:2f77349f9919bd535bd82b4eb710d136
echo "env (1019k30h): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST (+ v40b8a4, v40q8an4b handed over; 27 files in the skip list)"
