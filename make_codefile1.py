#!/usr/bin/env python3
"""make_codefile1.py — build the public "Code File 1" archive for JOSA A 621704.

1. Reproduce the frame enumeration of v7_base.SintelDataset._get_global_cache
   (scenes = sorted albedo dirs present in albedo/clean/final; frames = sorted
   albedo files present in all three passes; global index = enumeration order)
   from the MPI-Sintel training_images zip, and emit explicit split files for
   every protocol used in the paper:
     splits/frame_index.json                global index -> (scene, frame)
     splits/scene_p101..105.json            held-out scenes + train/test indices
     splits/random_p101..105.json           shuffled 80/20 train/test indices
     splits/temporal.json                   last 20% of each scene -> test
     splits/frame_alphabetical.json         alphabetical pooled 80/20 cut
2. Copy code/, results/**/*.json (per-run records), LICENSE, requirements.txt,
   write README.md (with the archive DOI), and zip everything to
   Woo_JOSAA_CodeFile1.zip.  The *.npz image arrays are deliberately excluded:
   they contain MPI-Sintel and MIT-Intrinsic imagery that is not ours to
   redistribute; the scripts regenerate them from the public datasets.

Usage: python3 make_codefile1.py --doi 10.17605/OSF.IO/XXXXX [--sintel-zip PATH]
"""
import argparse, json, os, shutil, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.join(HERE, 'supp_code')
STAGE = os.path.join(HERE, 'codefile1_stage')
OUT_ZIP = os.path.join(HERE, 'Woo_JOSAA_CodeFile1.zip')
DEFAULT_SINTEL = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'MPI-Sintel-training_images.zip'))
PARTITION_SEEDS = [101, 102, 103, 104, 105]
TEST_RATIO = 0.2


def enumerate_frames(sintel_zip):
    z = zipfile.ZipFile(sintel_zip)
    names = set(z.namelist())
    passes = {}
    for n in names:
        parts = n.split('/')
        if len(parts) == 4 and parts[0] == 'training' and parts[1] in ('albedo', 'clean', 'final') and parts[3]:
            passes.setdefault(parts[1], {}).setdefault(parts[2], set()).add(parts[3])
    all_scenes = sorted(s for s in passes['albedo'] if not s.startswith('.'))
    scenes = [s for s in all_scenes if all(s in passes[p] for p in ('albedo', 'clean', 'final'))]
    meta = []
    for s in scenes:
        frames = sorted(f for f in passes['albedo'][s] if not f.startswith('.'))
        for fi, f in enumerate(frames):
            if f in passes['clean'][s] and f in passes['final'][s]:
                meta.append((s, f, fi))
    return scenes, meta


def build_splits(scenes, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    N = len(meta)
    json.dump({'n_frames': N, 'n_scenes': len(scenes), 'scenes': scenes,
               'frames': [{'index': i, 'scene': s, 'frame': f} for i, (s, f, _) in enumerate(meta)]},
              open(os.path.join(out_dir, 'frame_index.json'), 'w'), indent=0)
    summary = {}
    for seed in PARTITION_SEEDS:
        rng = np.random.RandomState(seed)
        sh = list(scenes); rng.shuffle(sh)
        n_test = max(1, int(round(len(sh) * TEST_RATIO)))
        test_sc, train_sc = set(sh[:n_test]), set(sh[n_test:])
        te = [i for i, (s, _f, _fi) in enumerate(meta) if s in test_sc]
        tr = [i for i, (s, _f, _fi) in enumerate(meta) if s in train_sc]
        json.dump({'protocol': 'scene', 'partition_seed': seed,
                   'test_scenes': sorted(test_sc), 'train_scenes': sorted(train_sc),
                   'test_indices': te, 'train_indices': tr},
                  open(os.path.join(out_dir, f'scene_p{seed}.json'), 'w'))
        summary[f'scene_p{seed}'] = {'test_scenes': sorted(test_sc), 'n_test': len(te), 'n_train': len(tr)}
        rng = np.random.RandomState(seed)
        order = list(range(N)); rng.shuffle(order)
        cut = int(round(N * 0.8))
        json.dump({'protocol': 'random', 'partition_seed': seed,
                   'train_indices': order[:cut], 'test_indices': order[cut:]},
                  open(os.path.join(out_dir, f'random_p{seed}.json'), 'w'))
        summary[f'random_p{seed}'] = {'n_test': N - cut, 'n_train': cut,
                                      'n_test_scenes_also_in_train': len({meta[i][0] for i in order[cut:]} & {meta[i][0] for i in order[:cut]})}
    # temporal: per scene, first 80% train / last 20% test (round)
    by_scene = {}
    for i, (s, _f, _fi) in enumerate(meta):
        by_scene.setdefault(s, []).append(i)
    tr, te = [], []
    for s, idxs in by_scene.items():
        cut = int(round(len(idxs) * 0.8)); tr += idxs[:cut]; te += idxs[cut:]
    json.dump({'protocol': 'temporal', 'train_indices': tr, 'test_indices': te},
              open(os.path.join(out_dir, 'temporal.json'), 'w'))
    summary['temporal'] = {'n_test': len(te), 'n_train': len(tr)}
    # alphabetical pooled cut
    ordered = sorted(range(N), key=lambda i: (meta[i][0], meta[i][1]))
    cut = int(round(N * 0.8))
    json.dump({'protocol': 'frame_alphabetical', 'train_indices': ordered[:cut], 'test_indices': ordered[cut:]},
              open(os.path.join(out_dir, 'frame_alphabetical.json'), 'w'))
    summary['frame_alphabetical'] = {'n_test': N - cut, 'n_train': cut,
                                     'test_scenes': sorted({meta[i][0] for i in ordered[cut:]})}
    json.dump(summary, open(os.path.join(out_dir, 'SUMMARY.json'), 'w'), indent=1)
    return summary


README = """# Code File 1 — Scene-partition variance dominates learned intrinsic-image-decomposition benchmarks

Author: Jihwan Woo. Archive DOI: https://doi.org/{doi}
Companion to the manuscript "Scene-partition variance dominates learned
intrinsic-image-decomposition benchmarks: a reanalysis of evaluation on MPI
Sintel" (Journal of the Optical Society of America A, manuscript 621704, under review).

## Contents
- `code/` — training/evaluation code (`v7_base.py`, `experiment_v8.py`), the
  analysis that generates every number in the paper (`analyze_josaa.py`), the
  figure scripts (`make_figures_josaa.py`, `make_qual_figure.py`,
  `make_zoom_figure.py`, `make_mit_figure.py`) and the qualitative-render
  scripts (`qualitative_josaa.py`, `mit_qual_josaa.py`).
- `results/shard*/` — one JSON per training run ({n_runs} runs): architecture,
  protocol, partition seed, model seed and all component metrics. These are
  the per-run records behind every table, figure and test in the paper.
- `splits/` — explicit frame-index lists for every split protocol, generated
  by `make_codefile1.py` with the same enumeration and seeds as
  `code/v7_base.py` (`SintelDataset`): `frame_index.json` (global index ->
  scene/frame), `scene_p101..105.json` (held-out scenes), `random_p101..105.json`
  (leaky shuffled splits), `temporal.json`, `frame_alphabetical.json`,
  `SUMMARY.json`.
- `LICENSE` (MIT), `requirements.txt`.

## Not included, and why
The image arrays used to render the qualitative figures (`qualitative.npz`,
`mit_qual*.npz`) contain frames of the MPI-Sintel dataset (Butler et al.,
ECCV 2012; renders of the Blender Foundation's open film *Sintel*, CC BY 3.0)
and photographs from the MIT Intrinsic Images dataset (Grosse et al., ICCV
2009). They are not ours to redistribute and are therefore omitted; both
datasets are publicly downloadable and `code/qualitative_josaa.py` /
`code/mit_qual_josaa.py` regenerate the arrays deterministically (reference
configuration: partition seed 101, model seed 42).

## Reproducing the paper
```
pip install -r requirements.txt
# MPI-Sintel training_images.zip extracted to data/training (albedo/clean/final)
python code/experiment_v8.py --mode all --data-root data/training --num-shards 1 --shard 0
python code/analyze_josaa.py results paper/numbers_josaa.tex     # every reported number
python code/make_figures_josaa.py results paper/figures           # quantitative figures
```
Fixed partition seeds are 101–105 and model seeds 42–47 (hard-coded in the
scripts). All quantitative runs in the paper were executed on NVIDIA T4 GPUs.

## Citation
J. Woo, "Code, split-protocol definitions, and per-run result records for
'Scene-partition variance dominates learned intrinsic-image-decomposition
benchmarks'," OSF (2026), https://doi.org/{doi}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--doi', required=True)
    ap.add_argument('--sintel-zip', default=DEFAULT_SINTEL)
    a = ap.parse_args()
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    shutil.copytree(os.path.join(SUPP, 'code'), os.path.join(STAGE, 'code'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    n_runs = 0
    for root, _, files in os.walk(os.path.join(SUPP, 'results')):
        for f in files:
            if f.endswith('.json'):
                rel = os.path.relpath(os.path.join(root, f), SUPP)
                os.makedirs(os.path.dirname(os.path.join(STAGE, rel)), exist_ok=True)
                shutil.copy2(os.path.join(root, f), os.path.join(STAGE, rel)); n_runs += 1
    for f in ['LICENSE', 'requirements.txt']:
        shutil.copy2(os.path.join(SUPP, f), os.path.join(STAGE, f))
    scenes, meta = enumerate_frames(a.sintel_zip)
    summary = build_splits(scenes, meta, os.path.join(STAGE, 'splits'))
    open(os.path.join(STAGE, 'README.md'), 'w').write(README.format(doi=a.doi, n_runs=n_runs))
    shutil.copy2(os.path.abspath(__file__), os.path.join(STAGE, 'make_codefile1.py'))
    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(STAGE):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, STAGE))
    print(f'frames={len(meta)} scenes={len(scenes)} runs={n_runs}')
    print(json.dumps({k: v for k, v in summary.items() if k in ('scene_p101', 'random_p101', 'temporal', 'frame_alphabetical')}, indent=1))
    print('zip:', OUT_ZIP, os.path.getsize(OUT_ZIP) // 1024, 'KB')


if __name__ == '__main__':
    main()
