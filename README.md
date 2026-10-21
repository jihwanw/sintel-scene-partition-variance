# Code File 1 — Scene-partition variance dominates learned intrinsic-image-decomposition benchmarks

Author: Jihwan Woo. Archive DOI: https://doi.org/10.17605/OSF.IO/75MS8 (OSF, archival copy of this repository at tag `v1.0-josaa-621704`)
GitHub: https://github.com/jihwanw/sintel-scene-partition-variance
Companion to the manuscript "Scene-partition variance dominates learned
intrinsic-image-decomposition benchmarks: a reanalysis of evaluation on MPI
Sintel" (Journal of the Optical Society of America A, manuscript 621704, under review).

## Contents
- `code/` — training/evaluation code (`v7_base.py`, `experiment_v8.py`), the
  analysis that generates every number in the paper (`analyze_josaa.py`), the
  figure scripts (`make_figures_josaa.py`, `make_qual_figure.py`,
  `make_zoom_figure.py`, `make_mit_figure.py`) and the qualitative-render
  scripts (`qualitative_josaa.py`, `mit_qual_josaa.py`).
- `results/shard*/` — one JSON per training run (404 runs): architecture,
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
benchmarks'," OSF (2026), https://doi.org/10.17605/OSF.IO/75MS8
