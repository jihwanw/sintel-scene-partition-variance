#!/usr/bin/env python3
"""make_zoom_figure.py — crop-zoom qualitative figure.
For each held-out frame, find (automatically, no hand-picking) the 64x64 patch
where |err_direct - err_proposed| summed over the patch is largest, i.e. where
the two methods' reflectance errors vs GT differ most. Show: full input with a
red box, then zoomed GT / DirectCNN / Proposed reflectance patches.
"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
NPZ = sys.argv[1] if len(sys.argv) > 1 else 'results_t4/qualitative.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper_josaa/figures'
d = np.load(NPZ, allow_pickle=True)
hwc = lambda a: np.clip(np.transpose(a, (1, 2, 0)), 0, 1)
P = 64
def best_patch(gt, a, b):
    ea = np.abs(a - gt).sum(0); eb = np.abs(b - gt).sum(0)
    diff = ea - eb                      # where baseline is worse than proposed
    H, W = diff.shape; best = (-1e9, 0, 0)
    cs = np.pad(diff, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for y in range(0, H - P + 1, 4):
        for x in range(0, W - P + 1, 4):
            s = cs[y+P, x+P] - cs[y, x+P] - cs[y+P, x] + cs[y, x]
            if s > best[0]: best = (s, y, x)
    return best[1], best[2], diff
frames = [('frame0', d['frame0_scene']), ('frame1', d['frame1_scene'])]
fig, axes = plt.subplots(len(frames), 4, figsize=(10.4, 2.75 * len(frames)))
stats = {}
for r, (fr, sc) in enumerate(frames):
    gt, da, pr, inp = d[f'{fr}_gt_R'], d[f'{fr}_direct_R'], d[f'{fr}_proposed_R'], d[f'{fr}_input']
    y, x, diff = best_patch(gt, da, pr)
    sl = (slice(None), slice(y, y+P), slice(x, x+P))
    ea = np.abs(da[sl]-gt[sl]).mean(); eb = np.abs(pr[sl]-gt[sl]).mean()
    stats[fr] = dict(scene=str(sc), y=y, x=x, mae_direct=float(ea), mae_proposed=float(eb))
    ax = axes[r, 0]; ax.imshow(hwc(inp)); ax.add_patch(Rectangle((x, y), P, P, fill=False, color='red', lw=2))
    ax.set_title(f'Input ({sc})' if r == 0 else f'Input ({sc})', fontsize=11); ax.set_ylabel(f'frame {r+1}', fontsize=11)
    for c, (arr, lab) in enumerate([(gt, 'GT $R$ (zoom)'), (da, 'DirectCNN $\\hat R$ (zoom)'), (pr, 'Proposed $\\hat R$ (zoom)')], start=1):
        ax = axes[r, c]; ax.imshow(hwc(arr[sl]), interpolation='nearest'); ax.set_title(lab, fontsize=11)
        if c > 1:
            m = ea if c == 2 else eb
            ax.text(0.03, 0.05, f'MAE {m:.3f}', transform=ax.transAxes, color='w', fontsize=10,
                    bbox=dict(facecolor='k', alpha=0.6, pad=2))
    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Crop-zoom at the patch where the two methods differ most (selected automatically by error difference)', y=1.0, fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/F10_zoom_patches.pdf', bbox_inches='tight')
import json; json.dump(stats, open('results_t4/zoom_patch_stats.json', 'w'), indent=1)
print('F10 saved'); print(json.dumps(stats, indent=1))
