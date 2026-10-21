#!/usr/bin/env python3
"""make_mit_figure.py (v2) — MIT cross-domain panels with baseline comparison.

F9_mit_crossdomain.pdf : rows Input / GT R / DirectCNN R̂ / Proposed R̂ ; columns = objects.
F11_mit_zoom.pdf       : per object, the 48x48 patch (inside the mask) where the two
                         methods' reflectance errors differ most (auto-selected);
                         shows GT / DirectCNN / Proposed zooms with masked patch MAE.
Writes results_t4/mit_zoom_stats.json (per-object whole-object masked MAE and patch MAE).
Falls back to the proposed-only layout if directR is absent.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'results_t4/mit_qual2.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper_josaa/figures'
STATS = sys.argv[3] if len(sys.argv) > 3 else 'results_t4/mit_zoom_stats.json'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11})
hwc = lambda a: np.clip(np.transpose(np.asarray(a), (1, 2, 0)), 0, 1)

d = np.load(NPZ, allow_pickle=True)
objs_all = sorted(set(k.rsplit('_', 1)[0] for k in d.keys()))
has_direct = any(k.endswith('_directR') for k in d.keys())
pref = [o for o in ['apple', 'dinosaur', 'panther', 'turtle', 'cup1', 'frog2', 'paper1', 'teabag1'] if f'{o}_input' in d]
cols = pref[:4] if pref else objs_all[:4]

# ---------- F9: comparison panel ----------
rows = [('input', 'Input (photo)'), ('gtR', 'GT reflectance')]
if has_direct:
    rows.append(('directR', 'DirectCNN $\\hat R$'))
rows.append(('predR', 'Proposed $\\hat R$'))
fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 2.25 * len(rows)))
for r, (ck, clab) in enumerate(rows):
    for c, o in enumerate(cols):
        ax = axes[r, c]; ax.imshow(hwc(d[f'{o}_{ck}'])); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(o, fontsize=12)
    axes[r, 0].set_ylabel(clab, fontsize=12)
fig.suptitle('Zero-shot cross-domain check: Sintel-trained models applied to real MIT-intrinsic objects (no fine-tuning)',
             y=1.0, fontsize=12)
fig.tight_layout(); fig.savefig(f'{OUT}/F9_mit_crossdomain.pdf', bbox_inches='tight'); plt.close(fig)
print('F9 saved (direct present:', has_direct, ')')

# ---------- F11: crop-zoom with method comparison ----------
if has_direct:
    P = 48
    stats = {}
    fig, axes = plt.subplots(len(cols), 4, figsize=(10.4, 2.6 * len(cols)))
    for r, o in enumerate(cols):
        gt, da, pr, inp = d[f'{o}_gtR'], d[f'{o}_directR'], d[f'{o}_predR'], d[f'{o}_input']
        mk = d[f'{o}_mask'][0] if f'{o}_mask' in d else (gt.sum(0) > 0).astype(float)
        ea = (np.abs(da - gt).sum(0)) * mk; eb = (np.abs(pr - gt).sum(0)) * mk
        diff = ea - eb
        H, W = diff.shape; cs = np.pad(diff, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        cm = np.pad(mk, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        best = (-1e9, 0, 0)
        for y in range(0, H - P + 1, 4):
            for x in range(0, W - P + 1, 4):
                cov = cm[y+P, x+P] - cm[y, x+P] - cm[y+P, x] + cm[y, x]
                if cov < 0.6 * P * P:   # require patch mostly on the object
                    continue
                s = cs[y+P, x+P] - cs[y, x+P] - cs[y+P, x] + cs[y, x]
                if s > best[0]: best = (s, y, x)
        _, y, x = best
        sl = (slice(None), slice(y, y+P), slice(x, x+P)); m = mk[y:y+P, x:x+P] > 0
        mae_d = float(np.abs(da[sl]-gt[sl]).mean(0)[m].mean()); mae_p = float(np.abs(pr[sl]-gt[sl]).mean(0)[m].mean())
        full_d = float((np.abs(da-gt).mean(0))[mk > 0].mean()); full_p = float((np.abs(pr-gt).mean(0))[mk > 0].mean())
        stats[o] = dict(y=int(y), x=int(x), patch_mae_direct=mae_d, patch_mae_proposed=mae_p,
                        object_mae_direct=full_d, object_mae_proposed=full_p)
        ax = axes[r, 0]; ax.imshow(hwc(inp)); ax.add_patch(Rectangle((x, y), P, P, fill=False, color='red', lw=2))
        ax.set_title(f'Input ({o})', fontsize=11)
        for c, (arr, lab, mae) in enumerate([(gt, 'GT $R$ (zoom)', None), (da, 'DirectCNN $\\hat R$ (zoom)', mae_d),
                                             (pr, 'Proposed $\\hat R$ (zoom)', mae_p)], start=1):
            ax = axes[r, c]; ax.imshow(hwc(arr[sl]), interpolation='nearest'); ax.set_title(lab, fontsize=11)
            if mae is not None:
                ax.text(0.03, 0.05, f'MAE {mae:.3f}', transform=ax.transAxes, color='w', fontsize=10,
                        bbox=dict(facecolor='k', alpha=0.6, pad=2))
        for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('MIT objects: crop-zoom at the on-object patch where the two methods differ most (auto-selected)',
                 y=1.0, fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/F11_mit_zoom.pdf', bbox_inches='tight'); plt.close(fig)
    json.dump(stats, open(STATS, 'w'), indent=1)
    print('F11 saved'); print(json.dumps(stats, indent=1))
else:
    print('no directR in npz -> F11 skipped')
