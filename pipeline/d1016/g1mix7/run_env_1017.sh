# Env for d1016/g1mix7/build_g1mix7.py after blend_v38xn8 (XN8) read 9.9564798521 and became the best public file (user, main thread
# 2026-10-17 21:49 KST). Written 2026-10-17 21:51 KST by the main thread as a copy of run_env_1016.sh (left unchanged for the 21:10 KST
# builds) with: KRIG_LATEST -> kriging22 (= kriging21 + the v38xn8 reading), v38xn8 added to SUBMITTED_EXTRA, and the v38xn8 best case.
# usage: source run_env_1017.sh v38xn8 9.9564798521   (then: python build_g1mix7.py; OUT as needed)
export KRIG_LATEST=/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging22.py   # kriging22 = kriging21 + v38xn8 9.9564798521 (10-17 21:49 KST)
export SUBMITTED_EXTRA=v30b:26291df03fcddd4a6418be9dd72dc132,v30a:ad21ece3cb46157cbeb7ca8bd1970824,v31s:c50a5e8b1a21d57ba95f17896f212ccd,v31f:d4c1babf90badc95c6d230b019375b10,v32m:0a6a41901090f34f7bf55d24c1bf1aac,v36l7:4ed3ca53c55338572bfb7691fbeff522,v37x78:93cf38f0dbb61cc14f41c20e645b55d1,v37n2:d30852bad8500126c09364260837c3e7,v38q8:e68c6808fff5774da4ac47d0422bc655,v38xn8:8d7bbd7a63f4604f7bfd3b1f6dd311c3
case "${1:-v38xn8}" in
  v38xn8) export BASE=B1f,B4f,NSf,B8f BEST_NAME=v38xn8 BEST_PUB="$2"
          B1=ev2s_locK_fsim; B4=ev2m_locK_real5; NS=b5ns_locK_real5; B8=b5ns_locK_fsim
          export BEST_RUNS=$B1,$B1,${B1}_s1,${B1}_s1,${B1}_s2,${B1}_s2,$B4,$B4,$B4,${B4}_s1,${B4}_s1,${B4}_s1,$NS,$NS,${NS}_s1,${NS}_s1,${NS}_s2,${NS}_s2,$B8,$B8,${B8}_s1,${B8}_s1,${B8}_s2,${B8}_s2 ;;
  *) echo "unknown best $1"; return 1 ;;
esac
echo "g1mix7 env (1017): best=$BEST_NAME pub=$BEST_PUB base=$BASE krig=$KRIG_LATEST"
