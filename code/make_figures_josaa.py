#!/usr/bin/env python3
"""make_figures_josaa.py (v2) — figures for the JOSA A / IEEE Access paper.

F1 partition dominance (2x2, one panel per architecture)
F2 leakage gradient (grouped bars: random/temporal/scene/frame x arch)
F3 variance decomposition (partition vs seed variance, log scale)
F4 false-positive distributions under the 3-seed recipe (2x2)
F5 uncertainty channels vs trivial cues (AUROC)

All from results_t4/ per-run JSON (dedup by basename). Placeholders if absent.
Usage: python3 make_figures_josaa.py <results_dir> <out_fig_dir>
"""
import json, glob, sys, os, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'legend.fontsize': 10, 'figure.dpi': 150})

RES = sys.argv[1] if len(sys.argv) > 1 else 'results_t4'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper_josaa/figures'
os.makedirs(OUT, exist_ok=True)
_seen = {}
for f in sorted(glob.glob(f'{RES}/shard*/*.json')):
    _seen[os.path.basename(f)] = f
runs = [json.load(open(f)) for f in _seen.values()]
prot = [r for r in runs if r.get('kind') == 'protocol']
unc = [r for r in runs if r.get('kind') == 'uncertainty']
ARCH = ['direct', 'unet', 'transformer', 'proposed']
LBL = {'direct': 'DirectCNN', 'unet': 'UNet', 'transformer': 'TransUNet', 'proposed': 'Proposed'}
BLUE, RED, GRgray = '#2166ac', '#b2182b', '#999999'


def ph(path, text):
    fig, ax = plt.subplots(figsize=(6, 3)); ax.text(0.5, 0.5, text, ha='center', va='center'); ax.axis('off')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig); print('placeholder', path)


def cells(arch, protocol):
    d = {}
    for r in prot:
        if r['arch'] == arch and r['protocol'] == protocol:
            d.setdefault(r['partition'], []).append(r['R_PSNR'])
    return {k: np.array(v) for k, v in d.items()}


def pooled(arch, protocol):
    return np.array([r['R_PSNR'] for r in prot if r['arch'] == arch and r['protocol'] == protocol])


has = {pr: any(r['protocol'] == pr for r in prot) for pr in ['random', 'temporal', 'scene', 'frame']}

# ---------- F1: partition dominance, 2x2 ----------
if prot and has['scene']:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharey=False)
    for ax, a in zip(axes.ravel(), ARCH):
        byp = cells(a, 'scene'); parts = sorted(byp)
        rnd = pooled(a, 'random')
        for i, p in enumerate(parts):
            v = byp[p]
            ax.scatter(np.full(len(v), i) + np.random.RandomState(p).uniform(-.08, .08, len(v)),
                       v, s=30, alpha=0.8, color=BLUE, zorder=3, edgecolor='white', linewidth=0.4)
            ax.hlines(v.mean(), i - 0.33, i + 0.33, color=BLUE, lw=2.5, zorder=4)
        if len(rnd):
            xr = len(parts)
            ax.scatter(np.full(len(rnd), xr) + np.random.RandomState(1).uniform(-.08, .08, len(rnd)),
                       rnd, s=26, alpha=0.7, color=RED, zorder=3, edgecolor='white', linewidth=0.4)
            ax.hlines(rnd.mean(), xr - 0.33, xr + 0.33, color=RED, lw=2.5, zorder=4)
        ax.set_xticks(range(len(parts) + 1))
        ax.set_xticklabels([f'P{j+1}' for j in range(len(parts))] + ['rand'])
        ax.set_title(LBL[a]); ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Test $R$ PSNR (dB)')
    fig.suptitle('Scene partitions (blue, 6 seeds each) vs.\\ random leaky split (red)', y=0.995)
    fig.tight_layout(); fig.savefig(f'{OUT}/F1_partition_dominance.pdf', bbox_inches='tight'); plt.close(fig)
    print('F1 (2x2) saved')
else:
    ph(f'{OUT}/F1_partition_dominance.pdf', 'F1 pending')

# ---------- F2: leakage gradient grouped bars ----------
if prot and has['random']:
    protos = ['random', 'temporal', 'scene', 'frame']
    plabs = ['Random\n(leaky)', 'Temporal', 'Scene\n(strict)', 'Frame\n(alph.)']
    colors = [RED, '#ef8a62', BLUE, '#4393c3']
    x = np.arange(len(ARCH)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for k, (pr, c) in enumerate(zip(protos, colors)):
        means = [pooled(a, pr).mean() if len(pooled(a, pr)) else np.nan for a in ARCH]
        sds = [pooled(a, pr).std(ddof=1) if len(pooled(a, pr)) > 1 else 0 for a in ARCH]
        ax.bar(x + (k - 1.5) * w, means, w, yerr=sds, capsize=3, color=c, label=plabs[k], edgecolor='k', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in ARCH])
    ax.set_ylabel('Test $R$ PSNR (dB)')
    ax.set_title('Protocol strictness gradient: random $>$ temporal $>$ scene $\\approx$ frame')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12)); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{OUT}/F2_leakage_gradient.pdf', bbox_inches='tight'); plt.close(fig)
    print('F2 saved')
else:
    ph(f'{OUT}/F2_leakage_gradient.pdf', 'F2 pending')

# ---------- F3: variance decomposition ----------
if prot and has['scene']:
    x = np.arange(len(ARCH)); w = 0.36
    pv, sv = [], []
    for a in ARCH:
        byp = cells(a, 'scene')
        pm = np.array([v.mean() for v in byp.values()])
        pv.append(np.var(pm, ddof=1))
        sv.append(np.mean([v.var(ddof=1) for v in byp.values() if len(v) > 1]))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - w/2, pv, w, color=BLUE, label='Between-partition variance', edgecolor='k', linewidth=0.4)
    ax.bar(x + w/2, sv, w, color='#f4a582', label='Between-seed variance', edgecolor='k', linewidth=0.4)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in ARCH])
    ax.set_ylabel('Variance of test $R$ PSNR (dB$^2$, log)')
    for i, a in enumerate(ARCH):
        ax.text(i, max(pv[i], sv[i]) * 1.3, f'{pv[i]/sv[i]:.1f}$\\times$', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Variance decomposition: partition vs.\\ seed (ratio annotated)')
    ax.legend(); ax.grid(axis='y', alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f'{OUT}/F3_variance_decomp.pdf', bbox_inches='tight'); plt.close(fig)
    print('F3 saved')
else:
    ph(f'{OUT}/F3_variance_decomp.pdf', 'F3 pending')

# ---------- F4: false-positive distributions (2x2) ----------
if prot and has['scene']:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6))
    for ax, a in zip(axes.ravel(), ARCH):
        byp = cells(a, 'scene'); ps = list(byp)
        gaps, sig = [], []
        for i in range(len(ps)):
            for j in range(len(ps)):
                if i == j: continue
                vi, vj = byp[ps[i]], byp[ps[j]]
                for ai in itertools.combinations(range(len(vi)), 3):
                    for bj in itertools.combinations(range(len(vj)), 3):
                        xx, yy = vi[list(ai)], vj[list(bj)]
                        t, p = stats.ttest_rel(xx, yy)
                        gaps.append(xx.mean() - yy.mean()); sig.append(p < .05)
        gaps = np.array(gaps); sig = np.array(sig)
        ax.hist(gaps[~sig], bins=35, color=GRgray, alpha=0.75, label='n.s.')
        ax.hist(gaps[sig], bins=35, color=RED, alpha=0.8, label='$p<.05$')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(f'{LBL[a]}  ({100*sig.mean():.0f}\\% "significant")')
        ax.set_xlabel('Apparent scene-pair gap (dB)'); ax.grid(alpha=0.3)
    axes.ravel()[0].legend(loc='upper right')
    fig.suptitle('Single-partition 3-seed recipe on a TRUE-ZERO effect (scene vs scene)', y=0.995)
    fig.tight_layout(); fig.savefig(f'{OUT}/F4_false_positives.pdf', bbox_inches='tight'); plt.close(fig)
    print('F4 saved')
else:
    ph(f'{OUT}/F4_false_positives.pdf', 'F4 pending')

# ---------- F5: uncertainty channels vs trivial cues ----------
pf = [r for r in unc if r['arch'] == 'proposed_full']
pc = [r for r in unc if r['arch'] == 'perchannel']
if pf and pc:
    g = lambda rows, k: float(np.mean([r[k] for r in rows if k in r]))
    labels = ['$\\sigma_{\\mathrm{nl}}$', '$\\sigma_{\\mathrm{light}}$', 'luminance\n(trivial)', 'saturation\n(trivial)']
    full_v = [g(pf, 'sig_nl_auroc_H'), g(pf, 'sig_light_auroc_H'), g(pf, 'lum_auroc_H'), g(pf, 'sat_auroc_H')]
    pc_v = [g(pc, 'sig_nl_auroc_H'), g(pc, 'sig_light_auroc_H'), g(pc, 'lum_auroc_H'), g(pc, 'sat_auroc_H')]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.3))
    b1 = ax.bar(x - w/2, full_v, w, color=BLUE, label='Proposed', edgecolor='k', linewidth=0.4)
    b2 = ax.bar(x + w/2, pc_v, w, color='#92c5de', label='+ per-channel sup.', edgecolor='k', linewidth=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + .015, f'{b.get_height():.2f}', ha='center', fontsize=10)
    ax.axhline(0.5, color='k', ls='--', lw=0.8); ax.text(3.35, 0.52, 'chance', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('AUROC (top-decile $\\mathrm{err}_H$)')
    ax.set_ylim(0, 1.05); ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Learned uncertainty channels vs.\\ trivial input cues')
    fig.tight_layout(); fig.savefig(f'{OUT}/F5_cue_comparison.pdf', bbox_inches='tight'); plt.close(fig)
    print('F5 saved')
else:
    ph(f'{OUT}/F5_cue_comparison.pdf', 'F5 pending')
