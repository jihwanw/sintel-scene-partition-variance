#!/usr/bin/env python3
"""make_qual_figure.py — render the qualitative decomposition figures from
qualitative.npz:
  F6_decomposition.pdf : Input + (R,S,H) for GT and each method (grid).
  F7_leakage_visual.pdf: reflectance on a leaky random-split frame vs an
                         unseen scene-split frame (why leakage inflates PSNR).
Placeholders if the npz is absent.
Usage: python3 make_qual_figure.py <qualitative.npz> <out_fig_dir>
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'results_t4/qualitative.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper_josaa/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11})


def chw2hwc(a):
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return np.clip(a, 0, 1)


def ph(path, text):
    fig, ax = plt.subplots(figsize=(7, 3)); ax.text(0.5, 0.5, text, ha='center', va='center'); ax.axis('off')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig); print('placeholder', path)


if not os.path.exists(NPZ):
    ph(f'{OUT}/F6_decomposition.pdf', 'F6 decomposition\n(pending qualitative run)')
    ph(f'{OUT}/F7_leakage_visual.pdf', 'F7 leakage visualization\n(pending qualitative run)')
    sys.exit(0)

d = np.load(NPZ, allow_pickle=True)
methods = [('direct', 'DirectCNN'), ('unet', 'UNet'), ('transformer', 'TransUNet'), ('proposed', 'Proposed')]
comps = [('R', 'Reflectance $\\hat R$'), ('S', 'Shading $\\hat S$'), ('H', 'Non-Lamb. $\\hat H$')]
fr = 'frame0'

# ---------- F6: decomposition grid ----------
rows = [('gt', 'Ground truth')] + methods
ncol = 1 + len(comps)          # input + R,S,H
nrow = len(rows)
fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.9 * nrow))
for r, (mk, mlab) in enumerate(rows):
    # column 0: input (only on first row); keep axis so the row label shows
    ax = axes[r, 0]
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if r == 0:
        ax.imshow(chw2hwc(d[f'{fr}_input'])); ax.set_title('Input $I$', fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(True)
    ax.set_ylabel(mlab, fontsize=12, rotation=90, labelpad=6)
    for c, (ck, clab) in enumerate(comps):
        ax = axes[r, c + 1]
        key = f'{fr}_gt_{ck}' if mk == 'gt' else f'{fr}_{mk}_{ck}'
        arr = d[key] if key in d else None
        # DirectCNN does not model the non-Lambertian residual (H = 0)
        if arr is not None and not (mk == 'direct' and ck == 'H'):
            ax.imshow(chw2hwc(arr))
        else:
            ax.imshow(np.ones((8, 8, 3)) * 0.85)
            ax.text(0.5, 0.5, 'not\nmodeled', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='0.3')
        if r == 0:
            ax.set_title(clab, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Intrinsic decomposition of a held-out Sintel test frame (scene {str(d['frame0_scene'])})", y=1.005, fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/F6_decomposition.pdf', bbox_inches='tight'); plt.close(fig)
print('F6 saved')

# ---------- F7: leakage visualization (2x2, single-column friendly) ----------
if 'leak_input' in d:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8))
    # row 0: leaky random-split frame (scene also in train)
    axes[0, 0].imshow(chw2hwc(d['leak_gt_R'])); axes[0, 0].set_title('GT reflectance')
    axes[0, 1].imshow(chw2hwc(d['leak_pred_R'])); axes[0, 1].set_title('Predicted $\\hat R$')
    axes[0, 0].set_ylabel('Random split\n(scene seen in train)', fontsize=11)
    # row 1: unseen scene (scene-split held-out) = frame0 proposed
    axes[1, 0].imshow(chw2hwc(d['frame0_gt_R']))
    axes[1, 1].imshow(chw2hwc(d['frame0_proposed_R']))
    axes[1, 0].set_ylabel('Scene split\n(unseen scene)', fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Reflectance: seen-scene frame vs.\\ unseen scene\n(Proposed model)',
                 y=1.01, fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/F7_leakage_visual.pdf', bbox_inches='tight'); plt.close(fig)
    print('F7 (2x2) saved')
else:
    ph(f'{OUT}/F7_leakage_visual.pdf', 'F7 pending (no leak keys)')

# ---------- F8: second held-out frame, baseline vs physics reflectance (2x2) ----------
if 'frame1_input' in d:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8))
    axes[0,0].imshow(chw2hwc(d['frame1_input'])); axes[0,0].set_title('Input $I$')
    axes[0,1].imshow(chw2hwc(d['frame1_gt_R'])); axes[0,1].set_title('GT reflectance $R$')
    axes[1,0].imshow(chw2hwc(d['frame1_direct_R'])); axes[1,0].set_title('DirectCNN $\\hat R$')
    axes[1,1].imshow(chw2hwc(d['frame1_proposed_R'])); axes[1,1].set_title('Proposed $\\hat R$')
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Second held-out frame (scene {str(d['frame1_scene'])}): the\n"
                 "baseline retains shading; the physics model is flatter", y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/F8_decomp_frame2.pdf', bbox_inches='tight'); plt.close(fig)
    print('F8 (frame2) saved')
else:
    ph(f'{OUT}/F8_decomp_frame2.pdf', 'F8 pending')
