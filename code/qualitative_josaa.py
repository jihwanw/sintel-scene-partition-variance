#!/usr/bin/env python3
"""qualitative_josaa.py — produce qualitative decomposition arrays for the
JOSA A / IEEE Access paper.

Trains the four study architectures under the paper's reference configuration
(scene partition seed 101, model seed 42, same epochs as the main study) and
dumps, for a few representative TEST frames:
  input I, ground-truth (R,S,H), and each method's (Rhat,Shat,Hhat).
Also dumps a leakage-visualization pair for the Proposed model:
  - a random-split test frame (its scene is also in train: leakage), and
  - a scene-split test frame (held-out scene),
so the figure can show WHY the leaky split looks better.

Outputs a single npz: qualitative.npz  (uploaded to S3 by the runner).
Deterministic; faithful to the configuration of a main-study run.
"""
import json, sys, time
import numpy as np
import torch
import v7_base as B

DEV = B.DEVICE
DATA = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/data/training'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/home/ubuntu/qualitative.npz'
PART, SEED = 101, 42


def train_arch(arch, trl):
    B.set_seed(SEED)
    if arch == 'direct':
        m = B.DirectCNN(); B.train_2head(m, trl, 'direct', epochs=60)
    elif arch == 'unet':
        m = B.UNetPhysics(); B.train_3head(m, trl, 'unet', epochs=60)
    elif arch == 'transformer':
        import experiment_v8 as V8
        m = V8.TransformerUNet(); B.train_3head(m, trl, 'transformer', epochs=60)
    else:
        m = B.ProposedFull(); B.train_proposed(m, trl, 'proposed', epochs=80, warmup=56)
    return m.eval().to(DEV)


@torch.no_grad()
def predict(arch, m, I):
    x = I.unsqueeze(0).to(DEV)
    if arch == 'direct':
        R, S = m(x); H = torch.zeros_like(R)
    elif arch == 'proposed':
        R, S, H, _ = m(x)
    else:
        R, S, H = m(x)
    return (R[0].cpu().numpy(), S[0].cpu().numpy(), H[0].cpu().numpy())


def chw(t):
    return np.clip(t.cpu().numpy() if torch.is_tensor(t) else t, 0, 1)


def main():
    # scene-split datasets (train for fitting; test for display)
    trl, tel, ntr, nte = B.make_loaders(DATA, 'scene', PART, batch_size=16,
                                        size=256, num_workers=4)
    cache = B.SintelDataset._GLOBAL_CACHE[(DATA, 256)]
    meta = cache['meta']
    test_scene = B.SintelDataset(root=DATA, split='test', size=256,
                                 protocol='scene', seed=PART)
    # pick 2 representative test frames (middle of the test list)
    idxs = test_scene.indices
    pick = [idxs[len(idxs) // 3], idxs[2 * len(idxs) // 3]]

    out = {}
    for j, gi in enumerate(pick):
        I, R, S, H = cache['F'][gi], cache['R'][gi], cache['S'][gi], cache['H'][gi]
        out[f'frame{j}_input'] = chw(I); out[f'frame{j}_scene'] = meta[gi][0]
        out[f'frame{j}_gt_R'] = chw(R); out[f'frame{j}_gt_S'] = chw(S); out[f'frame{j}_gt_H'] = chw(H)

    archs = ['direct', 'unet', 'transformer', 'proposed']
    models = {}
    for a in archs:
        t0 = time.time()
        m = train_arch(a, trl); models[a] = m
        for j, gi in enumerate(pick):
            Rp, Sp, Hp = predict(a, m, cache['F'][gi])
            out[f'frame{j}_{a}_R'] = np.clip(Rp, 0, 1)
            out[f'frame{j}_{a}_S'] = np.clip(Sp, 0, 1)
            out[f'frame{j}_{a}_H'] = np.clip(Hp, 0, 1)
        print(f'[{a}] trained+predicted in {(time.time()-t0)/60:.1f} min', flush=True)
        np.savez(OUT, **out)  # checkpoint after each method

    # leakage visualization with the Proposed model: random-split test frame
    # whose scene is in train (leaky) vs the scene-split held-out frame above.
    rnd_tr = B.SintelDataset(root=DATA, split='train', size=256, protocol='random', seed=PART)
    rnd_te = B.SintelDataset(root=DATA, split='test', size=256, protocol='random', seed=PART)
    tr_scenes = set(meta[i][0] for i in rnd_tr.indices)
    leaky = next(i for i in rnd_te.indices if meta[i][0] in tr_scenes)
    Rp, Sp, Hp = predict('proposed', models['proposed'], cache['F'][leaky])
    out['leak_input'] = chw(cache['F'][leaky]); out['leak_scene'] = meta[leaky][0]
    out['leak_gt_R'] = chw(cache['R'][leaky]); out['leak_pred_R'] = np.clip(Rp, 0, 1)
    # (the scene-split unseen case is frame0_proposed_R above)
    np.savez(OUT, **out)
    print('saved', OUT, 'keys', len(out), flush=True)


if __name__ == '__main__':
    main()
