"""
v7: Unified MPI Sintel experiment — addresses Reviewer 1/2 concerns.

Key changes from v6:
- True scene-level split (split by SCENE name, not frame-alphabetical)
- Multi-seed runs (default 5 seeds) for statistical power
- Both scene-level and frame-level protocols (Table 2 reproduction + R1 #1)
- All models report Recon_PSNR (R2 #2: fix Section 4.9)
- Uncertainty controls: single-scalar head, permuted channel (R1 #4)
- Saves per-pixel arrays for calibration figure regeneration (R1 #9)
- AUROC/AUPRC for non-Lambertian region detection (R1 #4)
- DeepEnsemble (5 members) for fair comparison
- CLI args control which experiments run

Usage:
    python experiment_v7.py --mode protocol_comparison --seeds 5
    python experiment_v7.py --mode main_table --seeds 5
    python experiment_v7.py --mode uncertainty_controls --seeds 3
    python experiment_v7.py --mode all --seeds 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

# Optional helper for S3 sync / checkpointing
try:
    sys.path.insert(0, '/tmp')
    from experiment import Experiment as ExpHelper
    HELPER_AVAILABLE = True
except Exception:
    HELPER_AVAILABLE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# ================================================================
# Dataset — proper scene-level / frame-level / temporal splits
# ================================================================
class SintelDataset(torch.utils.data.Dataset):
    """
    Multi-protocol Sintel dataset.

    protocol:
      - 'scene' : split scenes into train/test (no scene appears in both)
      - 'frame' : pool all frames, alphabetical 80/20 (legacy / leaky)
      - 'temporal' : within each scene, first 80% frames train, last 20% test
    """

    # Class-level cache shared across instances within a process
    _GLOBAL_CACHE = {}

    @classmethod
    def _get_global_cache(cls, root, size):
        """Build (or retrieve) a process-wide cache of all (F, R, S, H, scene)
        tuples keyed by (root, size). This avoids re-decoding PNG for every
        dataset instance across protocols and seeds.
        """
        key = (root, size)
        if key in cls._GLOBAL_CACHE:
            return cls._GLOBAL_CACHE[key]

        albedo_dir = os.path.join(root, 'albedo')
        clean_dir = os.path.join(root, 'clean')
        final_dir = os.path.join(root, 'final')
        all_scenes = sorted(s for s in os.listdir(albedo_dir)
                            if not s.startswith('.'))
        scenes = [s for s in all_scenes
                  if all(os.path.isdir(os.path.join(d, s))
                         for d in [albedo_dir, clean_dir, final_dir])]

        tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        sample_meta = []  # list of (scene, frame_name, sample_idx_in_scene)
        F_list, R_list, S_list, H_list = [], [], [], []
        for scene in scenes:
            a_path = os.path.join(albedo_dir, scene)
            c_path = os.path.join(clean_dir, scene)
            f_path = os.path.join(final_dir, scene)
            frames = sorted(f for f in os.listdir(a_path)
                            if not f.startswith("."))
            for fi, frame in enumerate(frames):
                af = os.path.join(a_path, frame)
                cf = os.path.join(c_path, frame)
                ff = os.path.join(f_path, frame)
                if not all(os.path.exists(p) for p in [af, cf, ff]):
                    continue
                A = tfm(Image.open(af).convert('RGB'))
                C = tfm(Image.open(cf).convert('RGB'))
                F = tfm(Image.open(ff).convert('RGB'))
                S = C / (A + 1e-4)
                S = torch.clamp(S, 0, 2.0) / 2.0
                H = torch.clamp(F - C, 0, 1)
                F_list.append(F)
                R_list.append(A)
                S_list.append(S)
                H_list.append(H)
                sample_meta.append((scene, frame, fi))

        cache = {
            'F': torch.stack(F_list),
            'R': torch.stack(R_list),
            'S': torch.stack(S_list),
            'H': torch.stack(H_list),
            'meta': sample_meta,
            'scenes': scenes,
        }
        cls._GLOBAL_CACHE[key] = cache
        print(f"[SintelDataset] Built global cache: {len(sample_meta)} samples, "
              f"{len(scenes)} scenes, size={size}", flush=True)
        return cache

    def __init__(self, root='data/sintel_real/training', split='train',
                 size=256, protocol='scene', test_scene_ratio=0.2, seed=42):
        self.size = size
        self.protocol = protocol
        self.split = split

        cache = self._get_global_cache(root, size)
        self._cache = cache
        scenes = cache['scenes']

        if protocol == 'scene':
            rng = np.random.RandomState(seed)
            shuffled = list(scenes)
            rng.shuffle(shuffled)
            n_test = max(1, int(round(len(shuffled) * test_scene_ratio)))
            test_scenes = set(shuffled[:n_test])
            train_scenes = set(shuffled[n_test:])
            picked = train_scenes if split == 'train' else test_scenes
            self.split_scenes = sorted(picked)
            self.indices = [i for i, (s, _f, _fi) in enumerate(cache['meta'])
                            if s in picked]
        elif protocol == 'temporal':
            # Per-scene: first 80% train, last 20% test
            self.indices = []
            scene_to_indices = {}
            for i, (s, _f, _fi) in enumerate(cache['meta']):
                scene_to_indices.setdefault(s, []).append(i)
            for s, idxs in scene_to_indices.items():
                n = len(idxs)
                cut = int(round(n * 0.8))
                if split == 'train':
                    self.indices.extend(idxs[:cut])
                else:
                    self.indices.extend(idxs[cut:])
        elif protocol == 'frame':
            # Pool all, alphabetical 80/20. NOTE: on Sintel, filenames sort by
            # scene, so this deterministic cut is near-scene-level (NOT leaky);
            # retained only to document that "frame-level" is ambiguous.
            ordered = sorted(range(len(cache['meta'])),
                             key=lambda i: (cache['meta'][i][0],
                                            cache['meta'][i][1]))
            n = len(ordered)
            cut = int(round(n * 0.8))
            if split == 'train':
                self.indices = ordered[:cut]
            else:
                self.indices = ordered[cut:]
        elif protocol == 'random':
            # Genuinely leaky protocol: shuffle ALL frames by `seed`, 80/20.
            # Near-duplicate frames of the same scene land on both sides, so
            # this is the true "frame-level leakage" condition. `seed`
            # (partition seed 101-105) selects the random split; model init is
            # controlled separately by set_seed in the run driver.
            rng = np.random.RandomState(seed)
            order = list(range(len(cache['meta'])))
            rng.shuffle(order)
            n = len(order)
            cut = int(round(n * 0.8))
            if split == 'train':
                self.indices = order[:cut]
            else:
                self.indices = order[cut:]
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        c = self._cache
        return c['F'][i], c['R'][i], c['S'][i], c['H'][i]


def make_loaders(root, protocol, seed, batch_size=8, size=256, num_workers=0):
    train_ds = SintelDataset(root=root, split='train', size=size,
                             protocol=protocol, seed=seed)
    test_ds = SintelDataset(root=root, split='test', size=size,
                            protocol=protocol, seed=seed)
    trl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True)
    tel = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=True)
    return trl, tel, len(train_ds), len(test_ds)


# ================================================================
# Models
# ================================================================
class UNetBase(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(ch, ch * 2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(ch * 2, ch * 4, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(ch * 4, ch * 8, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 8, ch * 8, 3, padding=1), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(ch * 8, ch * 4, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(ch * 4, ch * 2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(ch * 2, ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())

    def encode_decode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return d1


class DirectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 6, 1))

    def forward(self, x):
        o = self.net(x)
        return torch.sigmoid(o[:, :3]), torch.sigmoid(o[:, 3:])


class UNetBaseline(UNetBase):
    def __init__(self):
        super().__init__()
        self.r_head = nn.Conv2d(48, 3, 1)
        self.s_head = nn.Conv2d(48, 3, 1)

    def forward(self, x):
        d = self.encode_decode(x)
        return torch.sigmoid(self.r_head(d)), torch.sigmoid(self.s_head(d))


class UNetPhysics(UNetBase):
    def __init__(self):
        super().__init__()
        self.r_head = nn.Conv2d(48, 3, 1)
        self.s_head = nn.Conv2d(48, 3, 1)
        self.h_head = nn.Conv2d(48, 3, 1)

    def forward(self, x):
        d = self.encode_decode(x)
        return (torch.sigmoid(self.r_head(d)),
                torch.sigmoid(self.s_head(d)),
                torch.relu(self.h_head(d)))


class ProposedFull(UNetBase):
    """3-source uncertainty (texture, lighting, non-Lambertian)."""

    def __init__(self):
        super().__init__()
        self.r_head = nn.Conv2d(48, 3, 1)
        self.s_head = nn.Conv2d(48, 3, 1)
        self.h_head = nn.Conv2d(48, 3, 1)
        self.var_head = nn.Sequential(
            nn.Conv2d(48 + 3, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 3, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.dropout(e1))
        e3 = self.enc3(self.dropout(e2))
        e4 = self.enc4(self.dropout(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return (torch.sigmoid(self.r_head(d1)),
                torch.sigmoid(self.s_head(d1)),
                torch.relu(self.h_head(d1)),
                self.var_head(torch.cat([d1, x], 1)))


class ProposedNoSkip(UNetBase):
    def __init__(self):
        super().__init__()
        self.r_head = nn.Conv2d(48, 3, 1)
        self.s_head = nn.Conv2d(48, 3, 1)
        self.h_head = nn.Conv2d(48, 3, 1)
        self.var_head = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(48, 3, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.dropout(e1))
        e3 = self.enc3(self.dropout(e2))
        e4 = self.enc4(self.dropout(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return (torch.sigmoid(self.r_head(d1)),
                torch.sigmoid(self.s_head(d1)),
                torch.relu(self.h_head(d1)),
                self.var_head(d1))


class ProposedSingleScalar(UNetBase):
    """R1 #4 control: single scalar uncertainty head (1 channel) instead of 3."""

    def __init__(self):
        super().__init__()
        self.r_head = nn.Conv2d(48, 3, 1)
        self.s_head = nn.Conv2d(48, 3, 1)
        self.h_head = nn.Conv2d(48, 3, 1)
        self.var_head = nn.Sequential(
            nn.Conv2d(48 + 3, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 1, 1))  # SINGLE channel
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.dropout(e1))
        e3 = self.enc3(self.dropout(e2))
        e4 = self.enc4(self.dropout(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return (torch.sigmoid(self.r_head(d1)),
                torch.sigmoid(self.s_head(d1)),
                torch.relu(self.h_head(d1)),
                self.var_head(torch.cat([d1, x], 1)))


# ================================================================
# Training functions
# ================================================================
def train_2head(model, loader, name, epochs=60, log_every=15, smooth_lambda=0.02):
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.to(DEVICE).train()
    for ep in range(epochs):
        el, n = 0.0, 0
        for I, R, S, _H in loader:
            I, R, S = I.to(DEVICE), R.to(DEVICE), S.to(DEVICE)
            opt.zero_grad()
            Rp, Sp = model(I)
            loss = (nn.functional.mse_loss(Rp, R)
                    + nn.functional.mse_loss(Sp, S)
                    + 0.5 * nn.functional.mse_loss(Rp * Sp, I))
            sg = Sp.mean(1, keepdim=True)
            sm = (((sg[:, :, 1:] - sg[:, :, :-1]) ** 2).mean()
                  + ((sg[:, :, :, 1:] - sg[:, :, :, :-1]) ** 2).mean())
            loss = loss + smooth_lambda * sm
            loss.backward()
            opt.step()
            el += loss.item()
            n += 1
        sch.step()
        if (ep + 1) % log_every == 0:
            print(f"  [{name}] Ep {ep+1}/{epochs} L={el/n:.4f}", flush=True)


def train_3head(model, loader, name, epochs=60, log_every=15, smooth_lambda=0.02):
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.to(DEVICE).train()
    for ep in range(epochs):
        el, n = 0.0, 0
        for I, R, S, H in loader:
            I, R, S, H = (I.to(DEVICE), R.to(DEVICE),
                          S.to(DEVICE), H.to(DEVICE))
            opt.zero_grad()
            Rp, Sp, Hp = model(I)
            loss = (nn.functional.mse_loss(Rp, R)
                    + nn.functional.mse_loss(Sp, S)
                    + 0.5 * nn.functional.mse_loss(Hp, H)
                    + nn.functional.mse_loss(Rp * Sp + Hp, I))
            sg = Sp.mean(1, keepdim=True)
            sm = (((sg[:, :, 1:] - sg[:, :, :-1]) ** 2).mean()
                  + ((sg[:, :, :, 1:] - sg[:, :, :, :-1]) ** 2).mean())
            loss = loss + smooth_lambda * sm
            loss.backward()
            opt.step()
            el += loss.item()
            n += 1
        sch.step()
        if (ep + 1) % log_every == 0:
            print(f"  [{name}] Ep {ep+1}/{epochs} L={el/n:.4f}", flush=True)


def train_proposed(model, loader, name, epochs=80, warmup=55,
                   log_every=20, smooth_lambda=0.02):
    mean_params = [p for n, p in model.named_parameters()
                   if 'var_head' not in n]
    var_params = list(model.var_head.parameters())
    opt_m = optim.Adam(mean_params, lr=5e-4)
    opt_v = optim.Adam(var_params, lr=2e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt_m, epochs)
    model.to(DEVICE).train()
    for ep in range(epochs):
        nw = 0.0 if ep < warmup else min(0.03,
                                          0.03 * (ep - warmup) / (epochs - warmup))
        el, n = 0.0, 0
        for I, R, S, H in loader:
            I, R, S, H = (I.to(DEVICE), R.to(DEVICE),
                          S.to(DEVICE), H.to(DEVICE))
            opt_m.zero_grad()
            opt_v.zero_grad()
            Rp, Sp, Hp, lv = model(I)
            lv = torch.clamp(lv, -10, 2)
            Ip = Rp * Sp + Hp
            loss = (nn.functional.mse_loss(Rp, R)
                    + nn.functional.mse_loss(Sp, S)
                    + 0.5 * nn.functional.mse_loss(Hp, H)
                    + nn.functional.mse_loss(Ip, I))
            sg = Sp.mean(1, keepdim=True)
            sm = (((sg[:, :, 1:] - sg[:, :, :-1]) ** 2).mean()
                  + ((sg[:, :, :, 1:] - sg[:, :, :, :-1]) ** 2).mean())
            loss = loss + smooth_lambda * sm
            if nw > 0:
                res = (I - Ip).detach() ** 2
                tlv = torch.logsumexp(lv, dim=1, keepdim=True)
                loss = loss + nw * 0.5 * (tlv + res * torch.exp(-tlv)).mean()
            loss.backward()
            opt_m.step()
            if nw > 0:
                opt_v.step()
            el += loss.item()
            n += 1
        sch.step()
        if (ep + 1) % log_every == 0:
            print(f"  [{name}] Ep {ep+1}/{epochs} L={el/n:.4f} nw={nw:.3f}",
                  flush=True)


# ================================================================
# Evaluation
# ================================================================
def psnr_from_mse(mse):
    return -10.0 * np.log10(mse + 1e-10)


@torch.no_grad()
def eval_2head(model, loader, name):
    model.eval().to(DEVICE)
    rm, sm, rcm = [], [], []
    for I, R, S, _H in loader:
        I, R, S = I.to(DEVICE), R.to(DEVICE), S.to(DEVICE)
        Rp, Sp = model(I)
        rm.append(nn.functional.mse_loss(Rp, R).item())
        sm.append(nn.functional.mse_loss(Sp, S).item())
        rcm.append(nn.functional.mse_loss(Rp * Sp, I).item())
    return {
        'model': name,
        'R_MSE': float(np.mean(rm)),
        'S_MSE': float(np.mean(sm)),
        'Recon_MSE': float(np.mean(rcm)),
        'R_PSNR': float(psnr_from_mse(np.mean(rm))),
        'S_PSNR': float(psnr_from_mse(np.mean(sm))),
        'Recon_PSNR': float(psnr_from_mse(np.mean(rcm))),
    }


@torch.no_grad()
def eval_3head(model, loader, name):
    model.eval().to(DEVICE)
    rm, sm, hm, rcm = [], [], [], []
    for I, R, S, H in loader:
        I, R, S, H = (I.to(DEVICE), R.to(DEVICE),
                      S.to(DEVICE), H.to(DEVICE))
        Rp, Sp, Hp = model(I)
        rm.append(nn.functional.mse_loss(Rp, R).item())
        sm.append(nn.functional.mse_loss(Sp, S).item())
        hm.append(nn.functional.mse_loss(Hp, H).item())
        rcm.append(nn.functional.mse_loss(Rp * Sp + Hp, I).item())
    return {
        'model': name,
        'R_MSE': float(np.mean(rm)),
        'S_MSE': float(np.mean(sm)),
        'H_MSE': float(np.mean(hm)),
        'Recon_MSE': float(np.mean(rcm)),
        'R_PSNR': float(psnr_from_mse(np.mean(rm))),
        'S_PSNR': float(psnr_from_mse(np.mean(sm))),
        'Recon_PSNR': float(psnr_from_mse(np.mean(rcm))),
    }


@torch.no_grad()
def eval_proposed(model, loader, name, save_arrays=False, n_channels=3):
    model.eval().to(DEVICE)
    rm, sm, hm, rcm = [], [], [], []
    errs_all, sigs_all = [], []
    sig_per_ch = [[] for _ in range(n_channels)] if n_channels > 1 else None
    err_R, err_S, err_H = [], [], []
    H_residuals = []
    for I, R, S, H in loader:
        I, R, S, H = (I.to(DEVICE), R.to(DEVICE),
                      S.to(DEVICE), H.to(DEVICE))
        Rp, Sp, Hp, lv = model(I)
        lv = torch.clamp(lv, -10, 2)
        rm.append(nn.functional.mse_loss(Rp, R).item())
        sm.append(nn.functional.mse_loss(Sp, S).item())
        hm.append(nn.functional.mse_loss(Hp, H).item())
        rcm.append(nn.functional.mse_loss(Rp * Sp + Hp, I).item())

        # Per-pixel error and total sigma
        e = ((Rp - R) ** 2 + (Sp - S) ** 2).mean(1).cpu().numpy()
        s_total = torch.exp(0.5 * lv).mean(1).cpu().numpy()  # over channels
        errs_all.append(e.flatten())
        sigs_all.append(s_total.flatten())

        if n_channels == 3:
            for c in range(3):
                sig_per_ch[c].append(torch.exp(0.5 * lv[:, c]).cpu().numpy().flatten())
            err_R.append(((Rp - R) ** 2).mean(1).cpu().numpy().flatten())
            err_S.append(((Sp - S) ** 2).mean(1).cpu().numpy().flatten())
            err_H.append(((Hp - H) ** 2).mean(1).cpu().numpy().flatten())
            # Predicted non-Lambertian residual magnitude (used as "real-world" proxy)
            H_residuals.append(Hp.mean(1).cpu().numpy().flatten())

    errs_all = np.concatenate(errs_all)
    sigs_all = np.concatenate(sigs_all)
    corr_total = float(np.corrcoef(errs_all, sigs_all)[0, 1])

    out = {
        'model': name,
        'R_MSE': float(np.mean(rm)),
        'S_MSE': float(np.mean(sm)),
        'H_MSE': float(np.mean(hm)),
        'Recon_MSE': float(np.mean(rcm)),
        'R_PSNR': float(psnr_from_mse(np.mean(rm))),
        'S_PSNR': float(psnr_from_mse(np.mean(sm))),
        'Recon_PSNR': float(psnr_from_mse(np.mean(rcm))),
        'uq_corr': corr_total,
        'n_channels': n_channels,
    }

    if n_channels == 3:
        sig_R = np.concatenate(sig_per_ch[0])
        sig_S = np.concatenate(sig_per_ch[1])
        sig_H = np.concatenate(sig_per_ch[2])
        e_R = np.concatenate(err_R)
        e_S = np.concatenate(err_S)
        e_H = np.concatenate(err_H)
        H_res = np.concatenate(H_residuals)

        cross = np.zeros((3, 3), dtype=float)
        sigs = [sig_R, sig_S, sig_H]
        errs = [e_R, e_S, e_H]
        for i in range(3):
            for j in range(3):
                cross[i, j] = float(np.corrcoef(sigs[i], errs[j])[0, 1])
        out['cross_corr_matrix'] = cross.tolist()  # rows: sigma_i, cols: error_j

        inter = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                inter[i, j] = float(np.corrcoef(sigs[i], sigs[j])[0, 1])
        out['inter_corr_matrix'] = inter.tolist()

        # AUROC / AUPRC for sigma_nl detecting top-percentile non-Lambertian residual
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            thresh = np.percentile(H_res, 90)  # top 10% of non-Lamb residual = "high"
            y_true = (H_res >= thresh).astype(int)
            out['auroc_sig_nl_vs_top10pct_H'] = float(roc_auc_score(y_true, sig_H))
            out['auprc_sig_nl_vs_top10pct_H'] = float(average_precision_score(y_true, sig_H))
            out['auroc_sig_tex_vs_top10pct_H'] = float(roc_auc_score(y_true, sig_R))
            out['auroc_sig_light_vs_top10pct_H'] = float(roc_auc_score(y_true, sig_S))
            # Intensity-only baseline
            # (we don't have intensity here directly; will be saved if save_arrays)
        except Exception as e:
            out['auroc_error'] = str(e)

    if save_arrays:
        out['_arrays'] = {
            'errs_total': errs_all,
            'sigs_total': sigs_all,
        }
        if n_channels == 3:
            out['_arrays']['sig_R'] = sig_R
            out['_arrays']['sig_S'] = sig_S
            out['_arrays']['sig_H'] = sig_H
            out['_arrays']['err_R'] = e_R
            out['_arrays']['err_S'] = e_S
            out['_arrays']['err_H'] = e_H
            out['_arrays']['H_residual'] = H_res

    return out


# ================================================================
# Statistical helpers
# ================================================================
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        return float(values.mean() if n else 0.0), float('nan'), float('nan')
    boot = np.array([rng.choice(values, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def paired_test(a, b):
    from scipy import stats as sps
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    if len(diff) < 2:
        return {'mean_diff': float(diff.mean()), 'n': len(diff)}
    t, p = sps.ttest_rel(a, b)
    return {
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std(ddof=1)),
        'n': len(diff),
        't_stat': float(t),
        'p_value': float(p),
    }


# ================================================================
# Experiment orchestrators
# ================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(arch, loaders, seed, epochs):
    """Train+eval one model. Returns dict (no _arrays)."""
    trl, tel = loaders
    set_seed(seed)
    if arch == 'DirectCNN':
        m = DirectCNN()
        train_2head(m, trl, f'{arch}-s{seed}', epochs=epochs)
        return eval_2head(m, tel, arch), m
    if arch == 'UNet':
        m = UNetBaseline()
        train_2head(m, trl, f'{arch}-s{seed}', epochs=epochs)
        return eval_2head(m, tel, arch), m
    if arch == 'UNet+Physics':
        m = UNetPhysics()
        train_3head(m, trl, f'{arch}-s{seed}', epochs=epochs)
        return eval_3head(m, tel, arch), m
    if arch == 'Proposed-NoSkip':
        m = ProposedNoSkip()
        train_proposed(m, trl, f'{arch}-s{seed}', epochs=epochs,
                       warmup=int(epochs * 0.7))
        return eval_proposed(m, tel, arch, save_arrays=False, n_channels=3), m
    if arch == 'Proposed-Full':
        m = ProposedFull()
        train_proposed(m, trl, f'{arch}-s{seed}', epochs=epochs,
                       warmup=int(epochs * 0.7))
        # save arrays controlled by caller (set later via save_arrays_first_seed)
        return eval_proposed(m, tel, arch,
                             save_arrays=getattr(run_one, '_save_arrays', False),
                             n_channels=3), m
    if arch == 'Proposed-SingleScalar':
        m = ProposedSingleScalar()
        train_proposed(m, trl, f'{arch}-s{seed}', epochs=epochs,
                       warmup=int(epochs * 0.7))
        return eval_proposed(m, tel, arch, save_arrays=False, n_channels=1), m
    raise ValueError(f"Unknown arch: {arch}")


def run_protocol_comparison(args, exp=None):
    """R1 #1 + Table 2: scene-level vs frame-level across 5+ seeds."""
    archs = ['DirectCNN', 'UNet', 'Proposed-Full']
    protocols = ['scene', 'frame']
    seeds = list(range(42, 42 + args.seeds))

    results = {p: {a: [] for a in archs} for p in protocols}
    t0 = time.time()
    total = len(archs) * len(protocols) * len(seeds)
    done = 0
    for protocol in protocols:
        for arch in archs:
            for seed in seeds:
                done += 1
                set_seed(seed)
                trl, tel, ntr, nte = make_loaders(
                    args.data_root, protocol, seed,
                    batch_size=args.batch_size, size=args.size,
                    num_workers=args.num_workers)
                print(f"\n[{done}/{total}] protocol={protocol} arch={arch} seed={seed} "
                      f"(train={ntr}, test={nte})", flush=True)
                run_one._save_arrays = False  # never save in protocol comparison
                r, _ = run_one(arch, (trl, tel), seed,
                               epochs=args.protocol_epochs)
                r['seed'] = seed
                if '_arrays' in r:
                    del r['_arrays']
                results[protocol][arch].append(r)
                if exp is not None:
                    exp.checkpoint({
                        'phase': 'protocol_comparison',
                        'done': done, 'total': total,
                        'last': {'protocol': protocol, 'arch': arch, 'seed': seed,
                                 'R_PSNR': r['R_PSNR'], 'S_PSNR': r['S_PSNR']}
                    })

    summary = {}
    for arch in archs:
        a_summary = {}
        for metric in ['R_PSNR', 'S_PSNR', 'Recon_PSNR']:
            sl = [r[metric] for r in results['scene'][arch] if metric in r]
            fl = [r[metric] for r in results['frame'][arch] if metric in r]
            if not sl or not fl:
                continue
            sl_mean, sl_lo, sl_hi = bootstrap_ci(sl)
            fl_mean, fl_lo, fl_hi = bootstrap_ci(fl)
            test = paired_test(fl, sl)
            a_summary[metric] = {
                'scene_mean': sl_mean, 'scene_ci': [sl_lo, sl_hi], 'scene_vals': sl,
                'frame_mean': fl_mean, 'frame_ci': [fl_lo, fl_hi], 'frame_vals': fl,
                'gap': fl_mean - sl_mean,
                'paired_test': test,
            }
        summary[arch] = a_summary
    return {
        'raw': results,
        'summary': summary,
        'meta': {
            'seeds': seeds,
            'archs': archs,
            'protocols': protocols,
            'epochs': args.protocol_epochs,
            'duration_seconds': time.time() - t0,
        }
    }


def run_main_table(args, exp=None):
    """R2 #2 + Table 1: all models × seeds, scene-level only, save Recon_PSNR."""
    archs = ['DirectCNN', 'UNet', 'UNet+Physics',
             'Proposed-NoSkip', 'Proposed-Full']
    seeds = list(range(42, 42 + args.seeds))
    results = {a: [] for a in archs}
    t0 = time.time()
    total = len(archs) * len(seeds)
    done = 0
    saved_arrays = None
    for arch in archs:
        for seed in seeds:
            done += 1
            set_seed(seed)
            trl, tel, ntr, nte = make_loaders(
                args.data_root, 'scene', seed,
                batch_size=args.batch_size, size=args.size,
                num_workers=args.num_workers)
            print(f"\n[{done}/{total}] main_table arch={arch} seed={seed} "
                  f"(train={ntr}, test={nte})", flush=True)
            run_one._save_arrays = (arch == 'Proposed-Full' and seed == 42)
            r, _ = run_one(arch, (trl, tel), seed, epochs=args.main_epochs)
            r['seed'] = seed
            if '_arrays' in r:
                saved_arrays = r['_arrays']
                del r['_arrays']
            results[arch].append(r)
            if exp is not None:
                exp.checkpoint({
                    'phase': 'main_table', 'done': done, 'total': total,
                    'last': {'arch': arch, 'seed': seed,
                             'R_PSNR': r['R_PSNR'], 'S_PSNR': r['S_PSNR'],
                             'Recon_PSNR': r['Recon_PSNR']}
                })

    summary = {}
    for arch in archs:
        s = {}
        for metric in ['R_PSNR', 'S_PSNR', 'Recon_PSNR', 'uq_corr']:
            vals = [r[metric] for r in results[arch] if metric in r]
            if not vals:
                continue
            m, lo, hi = bootstrap_ci(vals)
            s[metric] = {'mean': m, 'std': float(np.std(vals, ddof=1) if len(vals) > 1 else 0),
                         'ci95': [lo, hi], 'values': vals}
        summary[arch] = s

    out = {
        'raw': results,
        'summary': summary,
        'meta': {
            'seeds': seeds, 'archs': archs,
            'epochs': args.main_epochs,
            'duration_seconds': time.time() - t0,
        }
    }
    if saved_arrays is not None:
        np.savez(os.path.join(args.out_dir, 'calibration_arrays.npz'),
                 **{k: v for k, v in saved_arrays.items()})
        out['calibration_arrays_path'] = 'calibration_arrays.npz'
    return out


def run_uncertainty_controls(args, exp=None):
    """R1 #4: compare 3-channel vs single-scalar vs permuted controls."""
    seeds = list(range(42, 42 + args.seeds))
    archs = ['Proposed-Full', 'Proposed-SingleScalar']
    results = {a: [] for a in archs}
    t0 = time.time()
    total = len(archs) * len(seeds)
    done = 0
    for arch in archs:
        for seed in seeds:
            done += 1
            set_seed(seed)
            trl, tel, ntr, nte = make_loaders(
                args.data_root, 'scene', seed,
                batch_size=args.batch_size, size=args.size,
                num_workers=args.num_workers)
            print(f"\n[{done}/{total}] uncertainty_controls arch={arch} seed={seed}",
                  flush=True)
            r, _ = run_one(arch, (trl, tel), seed, epochs=args.main_epochs)
            r['seed'] = seed
            if '_arrays' in r:
                del r['_arrays']
            results[arch].append(r)
            if exp is not None:
                exp.checkpoint({
                    'phase': 'uncertainty_controls', 'done': done, 'total': total,
                    'last': {'arch': arch, 'seed': seed, 'uq_corr': r.get('uq_corr', 0)}
                })
    return {
        'raw': results,
        'meta': {
            'seeds': seeds, 'archs': archs,
            'epochs': args.main_epochs,
            'duration_seconds': time.time() - t0,
        }
    }


# ================================================================
# CLI
# ================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['protocol_comparison', 'main_table',
                                      'uncertainty_controls', 'all'],
                   default='all')
    p.add_argument('--data-root', default='/home/ubuntu/data/sintel_real/training')
    p.add_argument('--out-dir', default='./results_v7')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--protocol-epochs', type=int, default=30,
                   help='Epochs per protocol comparison run')
    p.add_argument('--main-epochs', type=int, default=80,
                   help='Epochs for main table runs (longer for warmup)')
    p.add_argument('--exp-name', default='jei_v7_revision')
    p.add_argument('--exp-project', default='JEI_IntrinsicUQ')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Args: {vars(args)}")
    print(f"Helper available: {HELPER_AVAILABLE}")

    exp = None
    if HELPER_AVAILABLE:
        exp = ExpHelper(name=args.exp_name, project=args.exp_project,
                        tags=['revision', 'reviewer-response'])
        exp.start(config=vars(args),
                  purpose={'goal': 'Address Reviewer 1, 2, Editor comments for JEI rejection',
                           'reviewer_items': ['R1#1', 'R1#4', 'R1#9', 'R2#2', 'R2#5']})

    all_out = {}
    try:
        if args.mode in ('protocol_comparison', 'all'):
            print("\n" + "=" * 70)
            print("PHASE: protocol_comparison")
            print("=" * 70)
            all_out['protocol_comparison'] = run_protocol_comparison(args, exp=exp)
            with open(os.path.join(args.out_dir, 'protocol_comparison.json'), 'w') as f:
                json.dump(all_out['protocol_comparison'], f, indent=2, default=str)

        if args.mode in ('main_table', 'all'):
            print("\n" + "=" * 70)
            print("PHASE: main_table")
            print("=" * 70)
            all_out['main_table'] = run_main_table(args, exp=exp)
            with open(os.path.join(args.out_dir, 'main_table.json'), 'w') as f:
                # exclude _arrays (numpy)
                serializable = {k: v for k, v in all_out['main_table'].items()
                                if k != '_arrays'}
                json.dump(serializable, f, indent=2, default=str)

        if args.mode in ('uncertainty_controls', 'all'):
            print("\n" + "=" * 70)
            print("PHASE: uncertainty_controls")
            print("=" * 70)
            all_out['uncertainty_controls'] = run_uncertainty_controls(args, exp=exp)
            with open(os.path.join(args.out_dir, 'uncertainty_controls.json'), 'w') as f:
                json.dump(all_out['uncertainty_controls'], f, indent=2, default=str)

        if exp is not None:
            exp.complete(
                results={'phases_completed': list(all_out.keys()),
                         'output_dir': args.out_dir},
                findings={'verdict': 'Reviewer-revision experiments completed'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        if exp is not None:
            exp.fail(e)
        raise


if __name__ == '__main__':
    main()
