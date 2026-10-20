"""kriging18 (forensics18: + v34l2 and v35xn, 10-16). Kriging of the public reading history (N6/pub/forensics.py Part 2 with v32m added; label-free). Defines kpred(u) for a
test change u vs blend_v30b: predicted public dMSE = m2 + 2 h_post, sd = 2 h_sd_post (not inflated; LOO says x1.34)."""
exec(open('/mnt/project-files/work/hardness-cache/scripts/d1016/pub/forensics18.py').read())


def kpred(u):
    r = krig(u); m2 = r['m2']; p = m2 + 2 * r['h_post']; sd = 2 * r['h_sd_post']
    return dict(m2=m2, pred=p, sd=sd, sd_x134=1.34 * sd, info=r['info_share'], rmse=float(np.sqrt(PUB['v30b'] ** 2 + p)))
