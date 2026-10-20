"""kriging30 (forensics30: + v40b8all, 10-19 00:35 KST). kriging29 (forensics29: + v39q8a4b, 10-18 14:48 KST). kriging28 (forensics28: + v39q3a6, 10-18 10:58 KST). kriging27 (forensics27: + v39q8ab3, 10-18 09:51 KST). kriging26 (forensics26: + v39q8ab, 10-18 07:25 KST). kriging25 (forensics25: + v39p7b, 10-18 05:24 KST). kriging24 (forensics24: + v38q8a, 10-17 23:42 KST). kriging23 (forensics23: + v38q86, 10-17 23:05 KST). kriging22 (forensics22: + v38xn8, 10-17 21:49 KST). kriging21 (forensics21: + v38q8, 10-17 19:18 KST). kriging20 (forensics20: + v37x78 and v37n2, 10-16 21:30 KST). kriging19 (forensics19: + v36l7, 10-16 19:34 KST). kriging18 (forensics18: + v34l2 and v35xn, 10-16). Kriging of the public reading history (N6/pub/forensics.py Part 2 with v32m added; label-free). Defines kpred(u) for a
test change u vs blend_v30b: predicted public dMSE = m2 + 2 h_post, sd = 2 h_sd_post (not inflated; LOO says x1.34)."""
exec(open('/mnt/project-files/work/hardness-cache/scripts/d1016/pub/forensics30.py').read())


def kpred(u):
    r = krig(u); m2 = r['m2']; p = m2 + 2 * r['h_post']; sd = 2 * r['h_sd_post']
    return dict(m2=m2, pred=p, sd=sd, sd_x134=1.34 * sd, info=r['info_share'], rmse=float(np.sqrt(PUB['v30b'] ** 2 + p)))
