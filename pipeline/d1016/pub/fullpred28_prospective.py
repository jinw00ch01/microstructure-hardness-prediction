import io, contextlib
C = '/mnt/project-files/work/hardness-cache'
K = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(C + '/scripts/d1016/pub/kriging28.py').read(), K)
np, PUB, sig_t, names, T, Dm, o, Kr = K['np'], K['PUB'], K['sig_t'], K['names'], K['T'], K['Dm'], K['o'], K['K']
f, n, N = 0.3, 300, 1000; t30 = T['v30b']
Cs = -f * (1 - f) / (N - 1) * np.ones((N, N)); np.fill_diagonal(Cs, f * (1 - f))
def full(x, idx, kp=False):
    u = x - t30; D = Dm[idx]; X2 = D ** 2; u2 = u ** 2
    kco = (u2 @ Cs @ X2.T + f * 4 * (u * sig_t ** 2) @ D.T) / n ** 2
    kcc = (u2 @ Cs @ u2 + f * 4 * np.sum(u ** 2 * sig_t ** 2)) / n ** 2
    Ki = np.linalg.inv(Kr[np.ix_(idx, idx)])
    mu = float(kco @ Ki @ o[idx]); var = float(kcc - kco @ Ki @ kco)
    # kpred-style (forensics krig): only the h term
    kuo = f * 2 * (D * sig_t ** 2) @ u / n ** 2; kuu = f * np.sum(u ** 2 * sig_t ** 2) / n ** 2
    hmu = float(kuo @ Ki @ o[idx]); hvar = float(kuu - kuo @ Ki @ kuo)
    return float(np.mean(u2)) + mu, np.sqrt(max(var, 0)), float(np.mean(u2)) + 2 * hmu, 2 * np.sqrt(max(hvar, 0))
order = list(PUB)  # chronological insertion order
zs_f, zs_k = [], []
for j, nm in enumerate(order):
    if nm == 'v30b' or order.index(nm) < order.index('v33xb'): continue
    prev = [i for i, q in enumerate(names) if order.index(q) < order.index(nm)]
    pf, sf, pk, sk = full(T[nm], prev)
    obs = PUB[nm] ** 2 - PUB['v30b'] ** 2
    zf, zk = (obs - pf) / sf, (obs - pk) / sk
    zs_f.append(zf); zs_k.append(zk)
    print(f"{nm:9s} prev {len(prev):2d} | full pred {np.sqrt(PUB['v30b']**2+pf):.4f} sd {sf/(2*np.sqrt(PUB['v30b']**2+pf)):.4f} z {zf:+6.2f} | kpred {np.sqrt(PUB['v30b']**2+pk):.4f} sd {sk/(2*np.sqrt(PUB['v30b']**2+pk)):.4f} z {zk:+6.2f} | read {PUB[nm]:.4f}")
zs_f, zs_k = np.array(zs_f), np.array(zs_k)
print(f"prospective z (n={len(zs_f)}): full rms {np.sqrt(np.mean(zs_f**2)):.2f} mean {zs_f.mean():+.2f} | kpred rms {np.sqrt(np.mean(zs_k**2)):.2f} mean {zs_k.mean():+.2f}")
