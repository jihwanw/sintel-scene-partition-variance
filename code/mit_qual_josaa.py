#!/usr/bin/env python3
"""mit_qual_josaa.py — cross-domain qualitative check.

Train the Proposed model on Sintel (scene partition 101, seed 42, 80 epochs,
identical to the study's reference config), then run it ZERO-SHOT on real
photographed MIT-intrinsic objects (no fine-tuning). Save, for a few objects:
  input (original.png), GT reflectance (reflectance.png), predicted Rhat, mask.
This illustrates that strong Sintel-benchmark behavior does not transfer to
real captured objects -- i.e. Sintel PSNR is benchmark-internal.
Outputs mit_qual.npz.
"""
import os, sys, numpy as np, torch
from PIL import Image
from torchvision import transforms
import v7_base as B

DEV = B.DEVICE
SINTEL = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/data/training'
MIT = sys.argv[2] if len(sys.argv) > 2 else '/home/ubuntu/data/MIT-intrinsic/data'
OUT = sys.argv[3] if len(sys.argv) > 3 else '/home/ubuntu/mit_qual2.npz'
PART, SEED = 101, 42
OBJECTS = ['apple', 'cup1', 'dinosaur', 'frog2', 'panther', 'paper1', 'teabag1', 'turtle']


def load_mit(obj, size=256):
    d = os.path.join(MIT, obj)
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    inp = tfm(Image.open(os.path.join(d, 'original.png')).convert('RGB'))
    refl = tfm(Image.open(os.path.join(d, 'reflectance.png')).convert('RGB'))
    mask = tfm(Image.open(os.path.join(d, 'mask.png')).convert('L'))
    return inp, refl, (mask > 0.5).float()


@torch.no_grad()
def predict_R(m, inp):
    R, S, H, _ = m(inp.unsqueeze(0).to(DEV))
    return R[0].cpu().numpy()


@torch.no_grad()
def predict_R_direct(m, inp):
    R, S = m(inp.unsqueeze(0).to(DEV))
    return R[0].cpu().numpy()


def main():
    B.set_seed(SEED)
    trl, _, _, _ = B.make_loaders(SINTEL, 'scene', PART, batch_size=16, size=256, num_workers=4)
    m = B.ProposedFull(); B.train_proposed(m, trl, 'proposed', epochs=80, warmup=56)
    m.eval().to(DEV)
    B.set_seed(SEED)
    md = B.DirectCNN(); B.train_2head(md, trl, 'direct', epochs=60)
    md.eval().to(DEV)
    out = {}
    avail = [o for o in OBJECTS if os.path.isdir(os.path.join(MIT, o))]
    if not avail:
        avail = sorted(x for x in os.listdir(MIT) if os.path.isdir(os.path.join(MIT, x)))[:6]
    print('MIT objects:', avail, flush=True)
    for o in avail:
        inp, refl, mask = load_mit(o)
        Rp = np.clip(predict_R(m, inp), 0, 1)
        Rd = np.clip(predict_R_direct(md, inp), 0, 1)
        mk = mask.numpy()
        out[f'{o}_input'] = np.clip(inp.numpy(), 0, 1)
        out[f'{o}_gtR'] = np.clip(refl.numpy(), 0, 1) * mk
        out[f'{o}_predR'] = Rp * mk
        out[f'{o}_directR'] = Rd * mk
        out[f'{o}_mask'] = mk
        np.savez(OUT, **out)
    print('saved', OUT, 'objects', len(avail), flush=True)


if __name__ == '__main__':
    main()
