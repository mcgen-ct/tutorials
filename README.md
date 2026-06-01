# 2026 CFNS-MCgen Tutorials

These are the tutorials for the Monte Carlo general education network (MCgen). The primary repository for these tutorials is [gitlab.cern.ch/mcgen-ct/tutorials](https://gitlab.com/mcgen-ct/tutorials), which is mirrored at [github.com/mcgen-ct/tutorials](https://github.com/mcgen-ct/tutorials) so they can be directly accessed via Google's Colab. This project is supported by NSF grant OAC-2417682.

This branch of `mcgen-ct/tutorials` is a snapshot of the tutorials being used during the [2026 CFNS summer school](https://indico.cfnssbu.physics.sunysb.edu/event/604/) which is running from 2026/06/01 - 2026/06/12.

## Lecturers

* Philip Ilten (University of Cincinnati)

## Schedule

* day 1 - introduction to MC
  - [visualization](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/pythia/vistas.ipynb)
  - [Pythia](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/pythia/worksheet.ipynb)
  - [tuning](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/pythia/tuning.ipynb)
* day 2 - general MC techniques
  - [random number generation](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/rng.ipynb)
  - [integration and sampling](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/integrate.ipynb)
* day 3 - particle-physics specific algorithms
  - [hard process](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/hard_process.ipynb)
  - [parton showers](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/parton_shower.ipynb)
  - [hadronization](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/hadronization.ipynb)

## Complete List of Notebooks

### Monte Carlo Notebooks

* [visualization](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/pythia/vistas.ipynb)
* [Pythia](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/pythia/worksheet.ipynb)
* [tuning](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/pythia/tuning.ipynb)
* [random number generation](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/rng.ipynb)
* [integration and sampling](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/integrate.ipynb)
* [hard process](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/hard_process.ipynb)
* [parton showers](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/parton_shower.ipynb)
* [hadronization](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/mc/hadronization.ipynb)

### Machine Learning Notebooks

* [linear and logistic regression](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/regression.ipynb)
* [classification](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/classify.ipynb)
* [auto-differentiation](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/autodiff.ipynb)
* [neural networks](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/neural_networks.ipynb)
* [neural networks with standard tools](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/neural_networks_jax_pytorch_tensorflow.ipynb)
* [unsupervised learning](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/unsupervised.ipynb)
* [normalizing flows](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/flows.ipynb)
* [boosted decision trees](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/trees.ipynb)
* [top tagging](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2026-cfns/ml/top_tagging.ipynb)

## Notebook Requirements

A number of external packages are used in the notebooks above. For each notebook, the following packages are required. Note, in some cases, large portions of the notebook can be completed without external packages.

* [`mc/hadronization.ipynb`](mc/hadronization.ipynb): `math`, `matplotlib`, `numpy`
* [`mc/hard_process.ipynb`](mc/hard_process.ipynb): `math`, `numpy`, `pythia8mc`, `wurlitzer`
* [`mc/integrate.ipynb`](mc/integrate.ipynb): `math`, `matplotlib`, `numpy`
* [`mc/parton_shower.ipynb`](mc/parton_shower.ipynb): `matplotlib`, `numpy`
* [`mc/rng.ipynb`](mc/rng.ipynb): `inspect`, `math`, `matplotlib`, `numpy`, `scipy`, `sys`, `time`
* [`ml/autodiff.ipynb`](ml/autodiff.ipynb): `graphviz`, `jax`, `matplotlib`, `numpy`, `torch`
* [`ml/classify.ipynb`](ml/classify.ipynb): `bs4`, `matplotlib`, `numpy`, `numpy`, `os`, `pandas`, `sklearn`, `urllib`
* [`ml/decision_trees_random_forests_boosted_decision_trees.ipynb`](ml/decision_trees_random_forests_boosted_decision_trees.ipynb): `cv2`, `matplotlib`, `numpy`, `os`, `pandas`, `scipy`, `sklearn`, `sys`, `xgboost`
* [`ml/example_top_dataset.ipynb`](ml/example_top_dataset.ipynb): `h5py`, `matplotlib`, `numpy`, `os`, `pandas`, `scipy`, `sklearn`, `torch`
* [`ml/flows.ipynb`](ml/flows.ipynb): `matplotlib`, `nflows`, `numpy`, `os`, `scipy`, `sys`, `torch`, `tqdm`
* [`ml/neural_networks.ipynb`](ml/neural_networks.ipynb): `matplotlib`, `numpy`
* [`ml/neural_networks_jax_pytorch_tensorflow.ipynb`](ml/neural_networks_jax_pytorch_tensorflow.ipynb): `jax`, `math`, `matplotlib`, `numpy`, `os`, `scipy`, `sklearn`, `sys`, `tensorflow`, `time`, `torch`
* [`ml/regression.ipynb`](ml/regression.ipynb): `matplotlib`, `numpy`, `os`, `scipy`, `sklearn`
* [`ml/unsupervised.ipynb`](ml/unsupervised.ipynb): `matplotlib`, `numpy`, `os`, `sklearn`
* [`pythia/tuning.ipynb`](pythia/tuning.ipynb): `math`, `matplotlib`, `numpy`, `pythia8mc`, `wurlitzer`
* [`pythia/worksheet.ipynb`](pythia/worksheet.ipynb): `argparse`, `matplotlib`, `os`, `pythia8mc`, `wurlitzer`

At the time of running this school the following package versions were used in Colab.

* `argparse`: `1.4.0`
* `bs4`: `0.0.2`
* `cv2`: `4.11.0`
* `graphviz`: `0.21`
* `h5py`: `3.14.0`
* `inspect`: `built-in`
* `jax`: `0.5.2`
* `math`: `built-in`
* `matplotlib`: `3.10.0`
* `nflows`: `0.14`
* `numpy`: `2.0.2`
* `os`: `built-in`
* `pandas`: `2.2.2`
* `pythia8mc`: `8.315.0`
* `python`: `3.11.13 (main, Jun  4 2025, 08:57:29) [GCC 11.4.0]`
* `scipy`: `1.15.3`
* `sklearn`: `1.6.1`
* `sys`: `built-in`
* `tensorflow`: `2.18.0`
* `time`: `built-in`
* `torch`: `2.6.0+cu124`
* `tqdm`: `4.67.1`
* `urllib`: `built-in`
* `wurlitzer`: `3.1.1`
* `xgboost`: `2.1.4`

## License

These tutorials are licensed under the GNU GPL version 2, or later and are copyrighted (C) 2025 by the MCgen authors. These tutorials are free software; you can redistribute them and/or modify them under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or any later version. These tutorials are distributed in the hope that they will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License or [`LICENSE.md`](LICENSE.md) for more details.

## MCgen Members

* Benoit Assi
* Christian Bierlich
* Rikab Gambhir
* Philip Ilten
* Tony Menzo
* Steve Mrenna
* Ben Nachman
* Manuel Szewc
* Michael Wilkinson
* Jure Zupan

## Former MCgen Members

* Ahmed Youssef

## MCgen Schools

These tutorials have been in used in the following schools, either directly run by MCgen, or with support from MCgen. Each school listed below has a frozen branch which archives the tutorials as used for that specific school.

* 2025/06/15 - 2025/06/27 [CTEQ-MCgen](https://indico.cern.ch/event/1497407/) with branch [`2025-cteq`](../../tree/2025-cteq)
  - `2025-cteq-v0`: tag for initial archived branch commit
* 2026/06/01 - 2026/06/12 [CFNS](https://indico.cfnssbu.physics.sunysb.edu/event/604/) with branch [`2026-cfns`](../../tree/2026-cfns)