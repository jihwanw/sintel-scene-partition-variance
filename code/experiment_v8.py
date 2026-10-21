#!/usr/bin/env python3
"""experiment_v8.py — JEI revision experiments (2026-08).

리뷰 반영 설계:
- R1#1  : 고정 파티션 5개 × 모델 시드 6개, 파티션/시드 분산 분리 (mixed-effects용)
- R1#2  : TransformerUNet 추가 (4개 아키텍처)
- R1#4  : per-channel NLL 감독(ProposedPerChannel) + single-scalar + permuted
          control + 휘도/채도 단서 상관 + AUROC/AUPRC
- R1#7  : 단일패스 / MC-Dropout / DeepEnsemble 비용 분리 계측
- R2#5/6: 결과는 정직 보고 (효과 부재 시 그대로)

샤딩: --shard i --num-shards N 으로 런 그리드를 분할해 여러 EC2에서 병렬 실행.
출력: results/ 아래 run 단위 JSON (하나의 런 = 하나의 파일, 원자적).
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import v7_base as B  # SintelDataset, models, train/eval, bootstrap_ci, set_seed

DEVICE = B.DEVICE

PARTITION_SEEDS = [101, 102, 103, 104, 105]   # 고정 장면 파티션
MODEL_SEEDS = [42, 43, 44, 45, 46, 47]        # 모델 초기화/셔플 시드


# ================================================================
# R1#2 — Transformer bottleneck UNet
# ================================================================
class TransformerUNet(B.UNetBase):
    """UNet encoder/decoder + MHSA transformer bottleneck (2 blocks)."""

    def __init__(self, ch=48, heads=8, blocks=2):
        super().__init__(ch)
        d = ch * 8
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 2,
            dropout=0.1, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=blocks)
        self.pos = None  # lazy positional embedding
        self.r_head = nn.Conv2d(ch, 3, 1)
        self.s_head = nn.Conv2d(ch, 3, 1)
        self.h_head = nn.Conv2d(ch, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)                        # B, 8ch, H/8, W/8
        b, c, h, w = e4.shape
        t = e4.flatten(2).transpose(1, 2)          # B, HW, C
        if self.pos is None or self.pos.shape[1] != h * w:
            self.pos = nn.Parameter(
                torch.zeros(1, h * w, c, device=e4.device)).requires_grad_(True)
        t = self.attn(t + self.pos)
        e4 = t.transpose(1, 2).reshape(b, c, h, w)
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return (torch.sigmoid(self.r_head(d1)),
                torch.sigmoid(self.s_head(d1)),
                torch.relu(self.h_head(d1)))


# ================================================================
# R1#4 / Path 2.2 — per-channel supervised uncertainty
# ================================================================
class ProposedPerChannel(B.UNetBase):
    """ProposedFull과 동일 구조; per-channel NLL로 학습 (train_perchannel)."""

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


def train_perchannel(model, loader, name, epochs=80, warmup=55, nw=0.05,
                     smooth_lambda=0.02):
    """σ_tex→err_R, σ_light→err_S, σ_nl→err_H 를 각각 직접 감독."""
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.to(DEVICE).train()
    for ep in range(epochs):
        for F, R, S, H in loader:
            F, R, S, H = [t.to(DEVICE) for t in (F, R, S, H)]
            Rp, Sp, Hp, lv = model(F)
            loss = ((Rp - R).pow(2).mean() + (Sp - S).pow(2).mean()
                    + (Hp - H).pow(2).mean())
            recon = Rp * Sp * 2.0 + Hp
            loss = loss + 0.5 * (recon - F).pow(2).mean()
            loss = loss + smooth_lambda * (
                (lv[:, :, 1:, :] - lv[:, :, :-1, :]).abs().mean()
                + (lv[:, :, :, 1:] - lv[:, :, :, :-1]).abs().mean())
            if ep >= warmup:
                lvc = torch.clamp(lv, -6, 6)
                for k, (P, T) in enumerate([(Rp, R), (Sp, S), (Hp, H)]):
                    err = (P - T).pow(2).mean(1, keepdim=True).detach()
                    loss = loss + nw * 0.5 * (
                        lvc[:, k:k + 1]
                        + err * torch.exp(-lvc[:, k:k + 1])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return model


# ================================================================
# 평가 확장 — 단서 대조·permuted control (R1#4)
# ================================================================
def eval_uncertainty_full(model, loader, name):
    """proposed 계열 공통 평가 + 휘도/채도 단서 상관 + AUROC."""
    model.to(DEVICE).eval()
    sig_all, errR, errS, errH, lum_all, sat_all = [], [], [], [], [], []
    mR = mS = mRec = 0.0; n = 0
    with torch.no_grad():
        for F, R, S, H in loader:
            F, R, S, H = [t.to(DEVICE) for t in (F, R, S, H)]
            Rp, Sp, Hp, lv = model(F)
            mR += (Rp - R).pow(2).mean().item() * F.size(0)
            mS += (Sp - S).pow(2).mean().item() * F.size(0)
            rec = Rp * Sp * 2.0 + Hp
            mRec += (rec - F).pow(2).mean().item() * F.size(0)
            n += F.size(0)
            sig = torch.exp(0.5 * torch.clamp(lv, -6, 6))
            sig_all.append(sig.cpu().numpy())
            errR.append((Rp - R).pow(2).mean(1).cpu().numpy())
            errS.append((Sp - S).pow(2).mean(1).cpu().numpy())
            errH.append((Hp - H).pow(2).mean(1).cpu().numpy())
            lum = (F * torch.tensor([.2989, .587, .114], device=DEVICE
                                    ).view(1, 3, 1, 1)).sum(1)
            mx, mn = F.max(1).values, F.min(1).values
            sat = (mx - mn) / (mx + 1e-6)
            lum_all.append(lum.cpu().numpy()); sat_all.append(sat.cpu().numpy())
    sig = np.concatenate(sig_all); eR = np.concatenate(errR)
    eS = np.concatenate(errS); eH = np.concatenate(errH)
    lum = np.concatenate(lum_all); sat = np.concatenate(sat_all)
    nch = sig.shape[1]
    flat = lambda a: a.reshape(-1)
    def corr(a, b):
        a, b = flat(a), flat(b)
        return float(np.corrcoef(a, b)[0, 1])
    def auroc(score, err):
        from sklearn.metrics import roc_auc_score, average_precision_score
        y = (flat(err) > np.quantile(flat(err), 0.9)).astype(int)
        s = flat(score)
        return (float(roc_auc_score(y, s)),
                float(average_precision_score(y, s)))
    out = {'name': name,
           'r_psnr': B.psnr_from_mse(mR / n), 's_psnr': B.psnr_from_mse(mS / n),
           'recon_psnr': B.psnr_from_mse(mRec / n), 'n_channels': nch}
    errs = {'err_R': eR, 'err_S': eS, 'err_H': eH}
    total_err = eR + eS + eH
    if nch == 3:
        chn = ['sig_tex', 'sig_light', 'sig_nl']
        for k, cn in enumerate(chn):
            for en, ev in errs.items():
                out[f'{cn}->{en}'] = corr(sig[:, k], ev)
            out[f'{cn}->lum'] = corr(sig[:, k], lum)
            out[f'{cn}->sat'] = corr(sig[:, k], sat)
            a, p = auroc(sig[:, k], eH)
            out[f'{cn}_auroc_H'], out[f'{cn}_auprc_H'] = a, p
        # permuted-channel control: 채널 무작위 재배정
        rng = np.random.RandomState(0)
        perm = sig[:, rng.permutation(3)]
        out['permuted_signl->err_H'] = corr(perm[:, 2], eH)
        # 단서 자체의 AUROC (σ가 단서 이상인가)
        a, p = auroc(lum, eH); out['lum_auroc_H'] = a
        a, p = auroc(sat, eH); out['sat_auroc_H'] = a
        out['uq_corr_total'] = corr(sig.mean(1), total_err)
    else:
        out['uq_corr_total'] = corr(sig[:, 0], total_err)
        a, p = auroc(sig[:, 0], eH)
        out['scalar_auroc_H'], out['scalar_auprc_H'] = a, p
    return out


# ================================================================
# 런 그리드 정의 + 샤딩
# ================================================================
def build_grid(mode):
    runs = []
    if mode in ('protocol', 'all'):
        archs = ['direct', 'unet', 'proposed', 'transformer']
        for arch in archs:
            for seed in MODEL_SEEDS:
                runs.append(dict(kind='protocol', arch=arch, protocol='frame',
                                 partition=0, seed=seed))
                for part in PARTITION_SEEDS:
                    runs.append(dict(kind='protocol', arch=arch,
                                     protocol='scene', partition=part,
                                     seed=seed))
    if mode in ('leaky', 'all'):
        # New arms for the 2026-10 JOSA A revision: a genuinely leaky
        # random-frame split (the true "frame-level leakage" condition) and a
        # mid-strictness temporal split, forming the random > temporal > scene
        # strictness gradient. Same four architectures / seeds as the scene arm.
        archs = ['direct', 'unet', 'proposed', 'transformer']
        for arch in archs:
            for part in PARTITION_SEEDS:          # 5 independent random splits
                for seed in MODEL_SEEDS:          # model init seeds
                    runs.append(dict(kind='protocol', arch=arch,
                                     protocol='random', partition=part,
                                     seed=seed))
            for seed in MODEL_SEEDS:              # temporal split is deterministic
                runs.append(dict(kind='protocol', arch=arch,
                                 protocol='temporal', partition=0, seed=seed))
    if mode in ('uncertainty', 'all'):
        for cfg in ['perchannel', 'proposed_full', 'single_scalar']:
            for part in PARTITION_SEEDS[:3]:
                for seed in MODEL_SEEDS[:4]:
                    runs.append(dict(kind='uncertainty', arch=cfg,
                                     protocol='scene', partition=part,
                                     seed=seed))
    if mode in ('cost', 'all'):
        runs.append(dict(kind='cost', arch='proposed', protocol='scene',
                         partition=PARTITION_SEEDS[0], seed=MODEL_SEEDS[0]))
    return runs


def run_protocol_one(r, args):
    B.set_seed(r['seed'])
    part_seed = r['partition'] if r['protocol'] in ('scene', 'random') else 42
    trl, tel, ntr, nte = B.make_loaders(
        args.data_root, r['protocol'], part_seed,
        batch_size=args.batch_size, size=args.size,
        num_workers=args.num_workers)
    t0 = time.time()
    if r['arch'] == 'direct':
        m = B.DirectCNN(); B.train_2head(m, trl, 'direct', epochs=args.protocol_epochs)
        res = B.eval_2head(m, tel, 'direct')
    elif r['arch'] == 'unet':
        m = B.UNetPhysics(); B.train_3head(m, trl, 'unet', epochs=args.protocol_epochs)
        res = B.eval_3head(m, tel, 'unet')
    elif r['arch'] == 'transformer':
        m = TransformerUNet(); B.train_3head(m, trl, 'transformer',
                                             epochs=args.protocol_epochs)
        res = B.eval_3head(m, tel, 'transformer')
    else:
        m = B.ProposedFull()
        B.train_proposed(m, trl, 'proposed', epochs=args.protocol_epochs,
                         warmup=int(args.protocol_epochs * 0.7))
        res = B.eval_proposed(m, tel, 'proposed')
    res.update(r, train_n=ntr, test_n=nte, minutes=(time.time() - t0) / 60)
    return res


def run_uncertainty_one(r, args):
    B.set_seed(r['seed'])
    trl, tel, ntr, nte = B.make_loaders(
        args.data_root, 'scene', r['partition'],
        batch_size=args.batch_size, size=args.size,
        num_workers=args.num_workers)
    t0 = time.time()
    if r['arch'] == 'perchannel':
        m = ProposedPerChannel()
        train_perchannel(m, trl, 'perchannel', epochs=args.main_epochs)
    elif r['arch'] == 'single_scalar':
        m = B.ProposedSingleScalar()
        B.train_proposed(m, trl, 'single_scalar', epochs=args.main_epochs)
    else:
        m = B.ProposedFull()
        B.train_proposed(m, trl, 'proposed_full', epochs=args.main_epochs)
    res = eval_uncertainty_full(m, tel, r['arch'])
    res.update(r, train_n=ntr, test_n=nte, minutes=(time.time() - t0) / 60)
    return res


def run_cost_one(r, args):
    """R1#7 — 단일패스 / MC-Dropout(T=20) / DeepEns(5) 비용 분리 계측."""
    B.set_seed(r['seed'])
    trl, tel, *_ = B.make_loaders(args.data_root, 'scene', r['partition'],
                                  batch_size=args.batch_size, size=args.size)
    out = {'kind': 'cost'}
    m = B.ProposedFull()
    t0 = time.time()
    B.train_proposed(m, trl, 'cost_single', epochs=args.main_epochs)
    out['train_min_single'] = (time.time() - t0) / 60
    out['params_M'] = sum(p.numel() for p in m.parameters()) / 1e6
    m.eval()
    xb = next(iter(tel))[0].to(DEVICE)
    with torch.no_grad():
        for _ in range(3): m(xb)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): m(xb)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        out['infer_ms_single'] = (time.time() - t0) / 20 * 1000
        m.train()  # dropout on
        t0 = time.time()
        for _ in range(20):
            for _ in range(20): m(xb)  # T=20 MC samples
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        out['infer_ms_mcdrop_T20'] = (time.time() - t0) / 20 * 1000
    # DeepEnsemble(5): 학습 5×, 추론 5×
    out['train_min_deepens5'] = out['train_min_single'] * 5
    out['infer_ms_deepens5'] = out['infer_ms_single'] * 5
    out['params_M_deepens5'] = out['params_M'] * 5
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='all',
                   choices=['protocol', 'leaky', 'uncertainty', 'cost', 'all'])
    p.add_argument('--data-root', default='/home/ubuntu/data/training')
    p.add_argument('--out-dir', default='/home/ubuntu/results')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--protocol-epochs', type=int, default=60)
    p.add_argument('--main-epochs', type=int, default=80)
    p.add_argument('--s3-prefix', default='')
    args = p.parse_args()

    grid = build_grid(args.mode)
    mine = [r for i, r in enumerate(grid) if i % args.num_shards == args.shard]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)}/{len(grid)} runs",
          flush=True)
    for i, r in enumerate(mine):
        tag = (f"{r['kind']}_{r['arch']}_{r['protocol']}"
               f"_p{r['partition']}_s{r['seed']}")
        fp = out / f"{tag}.json"
        if fp.exists():
            print(f"skip {tag}", flush=True); continue
        print(f"[{i+1}/{len(mine)}] {tag}", flush=True)
        try:
            if r['kind'] == 'protocol':
                res = run_protocol_one(r, args)
            elif r['kind'] == 'uncertainty':
                res = run_uncertainty_one(r, args)
            else:
                res = run_cost_one(r, args)
            fp.write_text(json.dumps(res, indent=1))
            if args.s3_prefix:
                os.system(f"aws s3 cp {fp} {args.s3_prefix}/{fp.name} --quiet")
        except Exception as e:
            print(f"FAILED {tag}: {e}", flush=True)
    print("shard done", flush=True)


if __name__ == '__main__':
    main()
