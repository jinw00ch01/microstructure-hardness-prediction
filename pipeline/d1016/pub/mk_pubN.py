"""Generator (main thread, 2026-10-17): forensicsN.py + krigingN.py = forensics(N-1) + krigingN-1 with new public readings added,
each BUILT_ON v30b (every file since v33xb is a v30b-base file), exactly as forensics20 added v37x78 and v37n2. Text edits only
(each anchor asserted unique); the submitted file must exist in cache submissions/ with the given md5 prefix.
usage: python3 mk_pubN.py N name=reading:md5 [name=reading:md5 ...]   e.g. python3 mk_pubN.py 21 v38q8=9.95xx:e68c6808"""
import hashlib, sys, re
P = '/mnt/project-files/work/hardness-cache/scripts/d1016/pub/'
SUB = '/mnt/project-files/work/hardness-cache/submissions/'
N = int(sys.argv[1]); M = N - 1
new = []
for a in sys.argv[2:]:
    n, rest = a.split('='); r, m = rest.split(':')
    assert hashlib.md5(open(f'{SUB}blend_{n}.csv', 'rb').read()).hexdigest().startswith(m), n
    float(r); new.append((n, r))
f = open(f'{P}forensics{M}.py').read()
names = ', '.join(n for n, _ in new)
assert f.count('"""forensics%d = ' % M) == 1
f = f.replace('"""forensics%d = ' % M, '"""forensics%d = forensics%d plus the %s reading(s) (user; main thread, mk_pubN.py). forensics%d = ' % (N, M, names, M), 1)
o = f"OUT = '/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging_out{M}'"
assert f.count(o) == 1; f = f.replace(o, o.replace(f'kriging_out{M}', f'kriging_out{N}'))
i = f.index('PUB = dict('); j = f.index(')', f.index('v37n2=9.9594601661', i))   # end of the PUB dict
add = ''.join(f', {n}={r}' for n, r in new)
f = f[:j] + add + f[j:]
i = f.index('BUILT_ON = dict('); j = f.index(')', f.index("v37n2='v30b'", i))
f = f[:j] + ''.join(f", {n}='v30b'" for n, _ in new) + f[j:]
for n, r in new:
    assert f.count(f'{n}={r}') == 1 and f.count(f"{n}='v30b'") == 1, n
open(f'{P}forensics{N}.py', 'w').write(f)
k = open(f'{P}kriging{M}.py').read()
o2 = f"exec(open('/mnt/project-files/work/hardness-cache/scripts/d1016/pub/forensics{M}.py').read())"
assert k.count(o2) == 1
k = k.replace(o2, o2.replace(f'forensics{M}.py', f'forensics{N}.py'))
k = k.replace('"""kriging%d (' % M, '"""kriging%d (forensics%d: + %s). kriging%d (' % (N, N, names, M), 1)
open(f'{P}kriging{N}.py', 'w').write(k)
print('wrote', f'{P}forensics{N}.py', f'{P}kriging{N}.py', 'added', new)
