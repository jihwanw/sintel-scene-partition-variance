#!/usr/bin/env python3
"""analyze_josaa.py — aggregate the full self-consistent grid (results_t4/)
into LaTeX \\newcommand macros for the JOSA A manuscript.

All numbers in the manuscript are generated here from per-run JSON records;
nothing is typed by hand. Protocols:
  random  (genuinely leaky: frames shuffled, same scene both sides)
  temporal(mid-strictness: within-scene past/future)
  scene   (strict: whole scenes held out)
  frame   (alphabetical cut; near-scene-level on Sintel -- appendix note)

Design: 4 architectures x {random,scene}: 5 partitions x 6 seeds;
        {temporal,frame}: 6 seeds.  Uncertainty: 3 cfg x 3 part x 4 seed.

Usage: python3 analyze_josaa.py <results_dir> <out_numbers.tex>
"""
import json, glob, sys, itertools, os
import numpy as np
from scipy import stats

RES = sys.argv[1] if len(sys.argv) > 1 else 'results_t4'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper_josaa/numbers_josaa.tex'
BOOT = 10000
RNG = np.random.RandomState(0)

_seen = {}
for f in sorted(glob.glob(f'{RES}/shard*/*.json')):
    _seen[os.path.basename(f)] = f  # dedup by run tag (same run may appear under 2 shard prefixes)
runs = [json.load(open(f)) for f in _seen.values()]
prot = [r for r in runs if r.get('kind') == 'protocol']
unc = [r for r in runs if r.get('kind') == 'uncertainty']
cost = [r for r in runs if r.get('kind') == 'cost']
print(f'loaded {len(runs)} runs: {len(prot)} protocol, {len(unc)} uncertainty, {len(cost)} cost')

ARCH = ['direct', 'unet', 'transformer', 'proposed']
TAG = {'direct': 'Direct', 'unet': 'Unet', 'transformer': 'Trans', 'proposed': 'Prop'}
M = {}  # macros

def put(k, v, fmt='{:.2f}'):
    M[k] = fmt.format(v) if isinstance(v, float) else str(v)

def by(arch, protocol):
    return [r for r in prot if r['arch'] == arch and r['protocol'] == protocol]

def parts(arch, protocol):
    d = {}
    for r in by(arch, protocol):
        d.setdefault(r['partition'], []).append(r['R_PSNR'])
    return {k: np.array(v) for k, v in d.items()}

def pooled(arch, protocol):
    return np.array([r['R_PSNR'] for r in by(arch, protocol)])

def boot_gap(a_parts, b_parts):
    """two-level bootstrap of mean(a)-mean(b); resample partitions then seeds."""
    ak, bk = list(a_parts), list(b_parts)
    out = []
    for _ in range(BOOT):
        ba = np.concatenate([RNG.choice(a_parts[ak[i]], len(a_parts[ak[i]]), replace=True)
                             for i in RNG.choice(len(ak), len(ak), replace=True)])
        bb = np.concatenate([RNG.choice(b_parts[bk[i]], len(b_parts[bk[i]]), replace=True)
                             for i in RNG.choice(len(bk), len(bk), replace=True)])
        out.append(ba.mean() - bb.mean())
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

# ---- 1) protocol means + leakage gradient (random>temporal>scene) ----
_MPREF = {'random': 'rand', 'temporal': 'temp', 'scene': 'scene', 'frame': 'frame'}
for a in ARCH:
    for pr, prtag in [('random', 'Rand'), ('temporal', 'Temp'), ('scene', 'Scene'), ('frame', 'Frame')]:
        v = pooled(a, pr)
        if len(v):
            put(f'{_MPREF[pr]}Mean{TAG[a]}', float(v.mean()))
            put(f'{pr}Sd{TAG[a]}', float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    # leakage gaps vs scene (partition-aware CI where both have partitions)
    sc = parts(a, 'scene')
    rn = parts(a, 'random')
    if sc and rn:
        g, lo, hi = boot_gap(rn, sc)
        put(f'leakGap{TAG[a]}', g, '{:+.2f}'); put(f'leakLo{TAG[a]}', lo, '{:+.2f}'); put(f'leakHi{TAG[a]}', hi, '{:+.2f}')
    tp = parts(a, 'temporal')
    if sc and tp:
        g, lo, hi = boot_gap(tp, sc)
        put(f'tempGap{TAG[a]}', g, '{:+.2f}'); put(f'tempLo{TAG[a]}', lo, '{:+.2f}'); put(f'tempHi{TAG[a]}', hi, '{:+.2f}')

# ---- 2) variance decomposition (scene and random) ----
for a in ARCH:
    for pr, prtag in [('scene', 'Scene'), ('random', 'Rand')]:
        p = parts(a, pr)
        if len(p) >= 2:
            pmeans = np.array([v.mean() for v in p.values()])
            bpv = float(np.var(pmeans, ddof=1))
            wsv = float(np.mean([np.var(v, ddof=1) for v in p.values() if len(v) > 1]))
            put(f'varPart{prtag}{TAG[a]}', bpv, '{:.3f}')
            put(f'varSeed{prtag}{TAG[a]}', wsv, '{:.3f}')
            put(f'varRatio{prtag}{TAG[a]}', bpv / wsv if wsv > 0 else float('nan'), '{:.1f}')

# ---- 3) false-positive / unreliability study ----
# (a) single-partition 3v3 recipe for RANDOM vs SCENE (true effect exists):
#     fraction significant AND fraction with CORRECT (positive) sign.
# (b) scene P_i vs scene P_j, 3v3 within pooled seeds (true zero): false-positive rate.
def recipe_random_vs_scene(a):
    rn = parts(a, 'random'); sc = parts(a, 'scene')
    sig = []; signpos = []; gaps = []
    for pr in rn:
        rv = rn[pr]
        for sp in sc:
            sv = sc[sp]
            for ri in itertools.combinations(range(len(rv)), 3):
                for si in itertools.combinations(range(len(sv)), 3):
                    x, y = rv[list(ri)], sv[list(si)]
                    t, p = stats.ttest_rel(x, y)
                    sig.append(p < .05); signpos.append(x.mean() > y.mean()); gaps.append(x.mean()-y.mean())
    return np.array(sig), np.array(signpos), np.array(gaps)

def recipe_scene_vs_scene(a):
    sc = parts(a, 'scene')
    ps = list(sc)
    sig = []; gaps = []
    for i in range(len(ps)):
        for j in range(len(ps)):
            if i == j: continue
            vi, vj = sc[ps[i]], sc[ps[j]]
            for ai in itertools.combinations(range(len(vi)), 3):
                for bj in itertools.combinations(range(len(vj)), 3):
                    x, y = vi[list(ai)], vj[list(bj)]
                    t, p = stats.ttest_rel(x, y)
                    sig.append(p < .05); gaps.append(abs(x.mean()-y.mean()))
    return np.array(sig), np.array(gaps)

for a in ARCH:
    sig, sp, gp = recipe_random_vs_scene(a)
    if len(sig):
        put(f'recipePow{TAG[a]}', 100*sig.mean(), '{:.0f}')         # "power" to call the real leak
        put(f'recipeSign{TAG[a]}', 100*sp.mean(), '{:.0f}')         # fraction correct sign
    s2, g2 = recipe_scene_vs_scene(a)
    if len(s2):
        put(f'fpScene{TAG[a]}', 100*s2.mean(), '{:.1f}')            # false positives, true zero
        put(f'fpSceneMax{TAG[a]}', float(g2.max()), '{:.2f}')

# same-partition architecture-comparison sign flips across partitions (method-ranking unreliability)
# count, per arch pair, how many of the 5 scene partitions give p<.05 and the sign
def arch_pair_flips():
    scp = {a: parts(a, 'scene') for a in ARCH}
    commonparts = sorted(set.intersection(*[set(scp[a]) for a in ARCH])) if all(scp[a] for a in ARCH) else []
    rows = []
    for a, b in itertools.combinations(ARCH, 2):
        signs = []; sigs = []
        for p in commonparts:
            x, y = scp[a][p], scp[b][p]
            n = min(len(x), len(y))
            t, pv = stats.ttest_rel(x[:n], y[:n])
            signs.append(np.sign(x.mean()-y.mean())); sigs.append(pv < .05)
        rows.append((a, b, signs, sigs))
    return commonparts, rows

cp, rows = arch_pair_flips()
# headline: number of arch pairs whose significant verdicts disagree in sign across partitions
flipn = 0
for a, b, signs, sigs in rows:
    sigsigns = [s for s, g in zip(signs, sigs) if g]
    if len(set(sigsigns)) > 1:
        flipn += 1
put('archFlipPairs', flipn); put('archPairTotal', len(rows)); put('nPart', len(cp))

# ---- 4) uncertainty under controls ----
def uagg(cfg, key):
    v = [r[key] for r in unc if r['arch'] == cfg and key in r]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)

for cfg, t in [('proposed_full', 'Full'), ('perchannel', 'PC'), ('single_scalar', 'SS')]:
    m, s, n = uagg(cfg, 'uq_corr_total')
    if m is not None:
        put(f'uq{t}', m, '{:.3f}'); put(f'uqSd{t}', s, '{:.3f}'); put(f'uqN{t}', n)
    if cfg != 'single_scalar':
        for key, mac in [('sig_nl->err_H', f'nlH{t}'), ('sig_tex->err_R', f'texR{t}'),
                         ('sig_light->err_S', f'lightS{t}'), ('permuted_signl->err_H', f'perm{t}'),
                         ('sig_nl_auroc_H', f'aurocNl{t}'), ('sig_light_auroc_H', f'aurocLight{t}'),
                         ('lum_auroc_H', f'aurocLum{t}'), ('sat_auroc_H', f'aurocSat{t}')]:
            m, s, n = uagg(cfg, key)
            if m is not None:
                put(mac, m, '{:.3f}')
    else:
        m, s, n = uagg(cfg, 'scalar_auroc_H')
        if m is not None: put('aurocScalarSS', m, '{:.3f}')

# per-run bimodality of proposed_full nl channel (how many runs have active nl)
pf = [r for r in unc if r['arch'] == 'proposed_full']
if pf:
    nlvals = np.array([r.get('sig_nl->err_H', 0.0) for r in pf])
    put('pfActiveN', int((np.abs(nlvals) > 0.05).sum())); put('pfTotalN', len(pf))

# ---- 5) cost ----
if cost:
    c = cost[0]
    put('costParams', float(c.get('params_M', 0)), '{:.2f}')
    put('costTrainMin', float(c.get('train_min_single', 0)), '{:.1f}')
    put('costInferSingle', float(c.get('infer_ms_single', 0)), '{:.0f}')
    put('costInferMC', float(c.get('infer_ms_mcdrop_T20', 0)), '{:.0f}')
    put('costInferEns', float(c.get('infer_ms_deepens5', 0)), '{:.0f}')
    put('costTrainEns', float(c.get('train_min_deepens5', 0)), '{:.1f}')

# totals
put('nRuns', len(runs)); put('nProtocol', len(prot)); put('nUnc', len(unc))

with open(OUT, 'w') as f:
    for k in sorted(M):
        f.write(f'\\newcommand{{\\{k}}}{{{M[k]}}}\n')
print(f'wrote {len(M)} macros to {OUT}')
# also dump a readable summary
for a in ARCH:
    print(a, 'rand', M.get(f'randMean{TAG[a]}'), 'temp', M.get(f'tempMean{TAG[a]}'),
          'scene', M.get(f'sceneMean{TAG[a]}'), '| leakGap', M.get(f'leakGap{TAG[a]}'),
          M.get(f'leakLo{TAG[a]}'), M.get(f'leakHi{TAG[a]}'),
          '| varRatioScene', M.get(f'varRatioScene{TAG[a]}'))
