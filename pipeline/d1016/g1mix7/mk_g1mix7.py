"""Generate d1016/g1mix7/build_g1mix7.py from d1016/g1mix6/build_g1mix6.py (sha asserted) for g1mix7/options.txt."""
import hashlib
G6 = '/mnt/project-files/work/hardness-cache/scripts/d1016/g1mix6'
G7 = '/mnt/project-files/work/hardness-cache/scripts/d1016/g1mix7'
src = open(G6 + '/build_g1mix6.py').read()
assert hashlib.sha256(src.encode()).hexdigest() == '3408758b03acf296693dd71f32dd0c5a1204ebc5c62ec07327b6da9bdfc0f3a6'
def sub(a, b, n=1):
    global src
    assert src.count(a) == n, (a[:80], src.count(a))
    src = src.replace(a, b)
i = src.index('"""', 3) + 3
src = ('"""d1016/g1mix7 builder (options.txt, sha256 asserted) = d1016/g1mix6/build_g1mix6.py (sha 3408758b...) with the g1mix7 reader\n'
       'families (NS6f, APf, NS6Bf, NS4f), options BASE + family / BASE + all complete new families, the BESTfixed reference (the exact\n'
       'rule of the best public file, reproduced bit for bit) and the g1mix7 bars (the g1mix6 rule re-based on the best file and the\n'
       'latest kriging). Main thread, 2026-10-16.\n'
       'usage (repo root): nice -n 5 python /mnt/project-files/work/hardness-cache/scripts/d1016/g1mix7/build_g1mix7.py\n'
       'env: BASE=B1f,B4f,NSf (families of the best file); BEST_NAME=v35xn, BEST_PUB=9.9722, BEST_RUNS=<exact reader list of the best\n'
       'file, comma-separated, with repeats> (default = the v35xn rule); SUBMITTED_EXTRA=name:md5,... (files submitted after v35xn,\n'
       'read from hardness-cache submissions/blend_<name>.csv); KRIG_LATEST=<pub/krigingNN.py> (default kriging18.py);\n'
       'RENAME=old=new,... (OOM-renamed runs, same seed of the family); OUT, ALLOW_PARTIAL as in build_g1mix6.py.\n'
       'Writes <OUT>/<opt>/{oof.csv,test.csv,result.json} and <OUT>/summary.json (test files stay in the cache; never in the repo).\n'
       'Labels: train only. Test rows enter only as label-free group membership, the T-CP gate flag and the reader test outputs."""'
       + src[i:])
sub("SPEC = '/mnt/project-files/work/hardness-cache/scripts/d1016/g1mix6'", "SPEC = '/mnt/project-files/work/hardness-cache/scripts/d1016/g1mix7'")
sub("OPT_SHA = 'a585cfbdca25435a4eb8bbf81208cd43f1c55543fc5a7451d0095f8b6fbbbecf'", "OPT_SHA = '2a7f15a0434901b481bfba12d6cace9ffca6ba2791088ce3ea52e451f4d40f5a'")
sub("""                B8f=['b5ns_locK_fsim', 'b5ns_locK_fsim_s1', 'b5ns_locK_fsim_s2'])
NEW_RUNS = set(FAM_RUNS['B1f'][1:] + FAM_RUNS['B4f'][1:] + FAM_RUNS['NSf'] + FAM_RUNS['B7f'] + FAM_RUNS['B8f'])
OPTIONS = dict(XN8=['B1f', 'B4f', 'NSf', 'B8f'], XN7=['B1f', 'B4f', 'NSf', 'B7f'], XN78=['B1f', 'B4f', 'NSf', 'B7f', 'B8f'],
               X8=['B1f', 'B4f', 'B8f'], N2=['NSf', 'B8f'])
DIAG = dict(XNf=['B1f', 'B4f', 'NSf'], NSf=['NSf'], B8f=['B8f'])""",
"""                B8f=['b5ns_locK_fsim', 'b5ns_locK_fsim_s1', 'b5ns_locK_fsim_s2'],
                NS6f=['b6ns_locK_real5', 'b6ns_locK_real5_s1', 'b6ns_locK_real5_s2'],
                APf=['b5ap_locK_real5', 'b5ap_locK_real5_s1', 'b5ap_locK_real5_s2'],
                NS6Bf=['b6ns_locK_fsim', 'b6ns_locK_fsim_s1', 'b6ns_locK_fsim_s2'],
                NS4f=['b4ns_locK_real5', 'b4ns_locK_real5_s1', 'b4ns_locK_real5_s2'])
for _x in os.environ.get('RENAME', '').split(','):  # OOM-renamed runs (laptop plan: '_b<batch>' appended) = the same seed of the family
    if _x:
        _a, _b = _x.split('=')
        assert sum(r == _a for v in FAM_RUNS.values() for r in v) == 1, _x
        FAM_RUNS = {f: [_b if r == _a else r for r in v] for f, v in FAM_RUNS.items()}
NEWF = ['NS6f', 'APf', 'NS6Bf', 'NS4f']
NEW_RUNS = set(FAM_RUNS['B1f'][1:] + FAM_RUNS['B4f'][1:] + FAM_RUNS['NSf'] + FAM_RUNS['B7f'] + FAM_RUNS['B8f'] + [r for f in NEWF for r in FAM_RUNS[f]])
BASE = [f for f in os.environ.get('BASE', 'B1f,B4f,NSf').split(',') if f]
assert BASE and all(f in FAM_RUNS for f in BASE) and not set(BASE) & set(NEWF), BASE
OPTIONS = dict(P6=BASE + ['NS6f'], PA=BASE + ['APf'], P6B=BASE + ['NS6Bf'], P4=BASE + ['NS4f'], PALL=BASE + NEWF)
DIAG = dict(NS6f=['NS6f'], APf=['APf'], NS6Bf=['NS6Bf'], NS4f=['NS4f'], BASEf=list(BASE))""")
sub("""print('families:', json.dumps(FAM), flush=True)
""", """print('families:', json.dumps(FAM), flush=True)
print('BASE:', BASE, flush=True)
if not ALLOW_PARTIAL:  # options.txt: PALL uses only the new families whose 3 seeds are all in
    _done = [f for f in NEWF if len(FAM.get(f, [])) == len(FAM_RUNS[f])]
    OPTIONS['PALL'] = BASE + _done
    if not _done:
        del OPTIONS['PALL']; print('PALL: no complete new family, not built', flush=True)
""")
sub("""g1core.EST_DIRS['G6_REF_XN'] = REF_XN_RUNS; g1core.HSPEC['replace_G6_REF_XN'] = (['G6_REF_XN'], False)
C = Ctx(ests=('LOC', 'A2', 'A3', 'A23', 'G3_REF_XB4', 'G6_REF_XN') + tuple('G6_' + k for k in ALLOPT))""",
"""g1core.EST_DIRS['G6_REF_XN'] = REF_XN_RUNS; g1core.HSPEC['replace_G6_REF_XN'] = (['G6_REF_XN'], False)
# BESTfixed = the exact rule of the best public file (default: the v35xn rule); reproduced bit for bit below
REF_BEST_RUNS = [r for r in os.environ.get('BEST_RUNS', ','.join(REF_XN_RUNS)).split(',') if r]
assert all(present(r) for r in REF_BEST_RUNS), 'BESTfixed readers missing'
g1core.EST_DIRS['G7_REF_BEST'] = REF_BEST_RUNS; g1core.HSPEC['replace_G7_REF_BEST'] = (['G7_REF_BEST'], False)
C = Ctx(ests=('LOC', 'A2', 'A3', 'A23', 'G3_REF_XB4', 'G6_REF_XN', 'G7_REF_BEST') + tuple('G6_' + k for k in ALLOPT))""")
sub("KNS17, KNS18 = {}, {}", "KNS17, KNS18, KNSL = {}, {}, {}")
sub("""    exec(open(os.environ.get('KRIG18', CACHE + '/scripts/d1016/pub/kriging18.py')).read(), KNS18)  # kriging18 = kriging17 + v35xn (the g1mix6 bars)
""", """    exec(open(os.environ.get('KRIG18', CACHE + '/scripts/d1016/pub/kriging18.py')).read(), KNS18)  # kriging18 = kriging17 + v35xn (the g1mix6 bars)
    exec(open(os.environ.get('KRIG_LATEST', CACHE + '/scripts/d1016/pub/kriging18.py')).read(), KNSL)  # the latest kriging (the g1mix7 bars)
print('latest kriging:', os.environ.get('KRIG_LATEST', CACHE + '/scripts/d1016/pub/kriging18.py'), flush=True)
""")
sub("""TSUB = {}
""", """for _x in os.environ.get('SUBMITTED_EXTRA', '').split(','):
    if _x:
        _n, _m = _x.split(':'); assert _n not in SUBMITTED; SUBMITTED[_n] = _m
BEST_NAME = os.environ.get('BEST_NAME', 'v35xn')
assert BEST_NAME in SUBMITTED, BEST_NAME
TSUB = {}
""")
sub("BEST_PUB, BOLD_KR = 9.9722, 9.9854", "BEST_PUB = float(os.environ.get('BEST_PUB', '9.9722')); BOLD_KR = round(BEST_PUB + 0.0132, 4)\nprint('best file', BEST_NAME, 'public', BEST_PUB, '| BOLD kriging bar', BOLD_KR, flush=True)")
sub("""REFS = {'A3fixed': ('replace_A3', 0.75), 'XB4fixed': ('replace_G3_REF_XB4', 0.75), 'XNfixed': ('replace_G6_REF_XN', 0.75)}""",
    """REFS = {'A3fixed': ('replace_A3', 0.75), 'XB4fixed': ('replace_G3_REF_XB4', 0.75), 'XNfixed': ('replace_G6_REF_XN', 0.75),
        'BESTfixed': ('replace_G7_REF_BEST', 0.75)}""")
sub("""print('XN reference test md5', xnmd5, '(blend_v35xn: d1eab1318577e0b8f962e253632b9e8d)', flush=True)
""", """print('XN reference test md5', xnmd5, '(blend_v35xn: d1eab1318577e0b8f962e253632b9e8d)', flush=True)
tbest = mapt(test_g1_clip(REFS['BESTfixed'])[0])
assert np.array_equal(tbest, TSUB[BEST_NAME]), 'BEST_RUNS do not reproduce the best file'
print('BEST reference reproduces blend_' + BEST_NAME + ' bit for bit (md5', SUBMITTED[BEST_NAME] + ')', flush=True)
""")
sub("""    VA, VX, VN = vs_ref(q, opt, 'A3fixed'), vs_ref(q, opt, 'XB4fixed'), vs_ref(q, opt, 'XNfixed')""",
    """    VA, VX, VN = vs_ref(q, opt, 'A3fixed'), vs_ref(q, opt, 'XB4fixed'), vs_ref(q, opt, 'XNfixed')
    VB = vs_ref(q, opt, 'BESTfixed')""")
sub("""    kr18 = KNS18['kpred'](u)
""", """    kr18 = KNS18['kpred'](u)
    krL = KNSL['kpred'](u)
""")
sub("""    safe = VN['oof_dmse'] < 0 and VN['folds_better'] >= 3 and VN['resplit40']['P_ge0'] <= 0.5 and kr18['rmse'] < BEST_PUB
    bold = VN['oof_dmse'] <= 0.42 and kr18['rmse'] <= BOLD_KR and rms_sub['v35xn'] >= 0.5""",
    """    safe = VB['oof_dmse'] < 0 and VB['folds_better'] >= 3 and VB['resplit40']['P_ge0'] <= 0.5 and krL['rmse'] < BEST_PUB
    bold = VB['oof_dmse'] <= 0.42 and krL['rmse'] <= BOLD_KR and rms_sub[BEST_NAME] >= 0.5""")
sub("""               kriging_public=kr, kriging17_public=kr17, kriging18_public=kr18, rms_G1t_vs_submitted=rms_sub, slate_flag=flag,""",
    """               kriging_public=kr, kriging17_public=kr17, kriging18_public=kr18, rms_G1t_vs_submitted=rms_sub, slate_flag=flag,
               vs_BESTfixed=VB, kriging_latest_public=krL, best_name=BEST_NAME, best_pub=BEST_PUB, base=BASE,""")
sub("""| rms vs35xn {rms_sub['v35xn']:.3f} min-sub {min(rms_sub.values()):.3f} | {flag} | md5 {res['test_md5']}", flush=True)""",
    """| rms vs35xn {rms_sub['v35xn']:.3f} min-sub {min(rms_sub.values()):.3f} | "
          f"vs BESTfixed({BEST_NAME}) {VB['oof_dmse']:+.3f} {VB['folds_better']}/5 rs {VB['resplit40']['mean']:+.3f} (P {VB['resplit40']['P_ge0']:.2f}) | kL {krL['pred']:+.3f} +- {krL['sd']:.3f} -> {krL['rmse']:.4f} | rms vs best {rms_sub[BEST_NAME]:.3f} | {flag} | md5 {res['test_md5']}", flush=True)""")
sub("""                   kr18_rmse=v['kriging18_public']['rmse'], rms_sub=v['rms_G1t_vs_submitted'], flag=v['slate_flag'],""",
    """                   kr18_rmse=v['kriging18_public']['rmse'], rms_sub=v['rms_G1t_vs_submitted'], flag=v['slate_flag'],
                   vs_BESTfixed=v['vs_BESTfixed'], krL=v['kriging_latest_public']['pred'], krL_sd=v['kriging_latest_public']['sd'],
                   krL_rmse=v['kriging_latest_public']['rmse'], best_name=v['best_name'], best_pub=v['best_pub'], base=v['base'],""")
open(G7 + '/build_g1mix7.py', 'w').write(src)
print('wrote', G7 + '/build_g1mix7.py', hashlib.sha256(src.encode()).hexdigest())
