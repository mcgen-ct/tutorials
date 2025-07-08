# 2025 CTEQ-MCgen Tutorials

These are the tutorials for the Monte Carlo general education network (MCgen). The primary repository for these tutorials is [gitlab.cern.ch/mcgen-ct/tutorials](https://gitlab.com/mcgen-ct/tutorials), which is mirrored at [github.com/mcgen-ct/tutorials](https://github.com/mcgen-ct/tutorials) so they can be directly accessed via Google's Colab. This project is supported by NSF grant OAC-2417682.

This branch of `mcgen-ct/tutorials` is a snapshot of the tutorials used during the [2025 CTEQ-MCgen summer school](https://indico.cern.ch/event/1497407/) which ran from 2025/06/15 - 2025/06/27.

## Lecturers

* Rikab Gambhir (MIT)
* Philip Ilten (University of Cincinnati)
* Tony Menzo (University of Cincinnati)
* Manuel Szewc (University of Cincinnati)

## Lecture Slides

* [MC1](pdf/lectures/mc1.pdf) - introduction and Monte Carlo techniques
* [MC2](pdf/lectures/mc2.pdf) - matrix elements and parton showers
* [MC3](pdf/lectures/mc3.pdf) - multi-parton interactions, hadronization, and non-perurbative effects
* [ML1](pdf/lectures/ml1.pdf) - introduction to machine learning
* [ML2](pdf/lectures/ml2.pdf) - advanced machine learning techniques

## Monte Carlo Notebooks

* [Pythia](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/pythia/worksheet.ipynb) [\[archived pdf\]](pdf/pythia/worksheet.pdf)
* [tuning](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/pythia/tuning.ipynb) [\[archived pdf\]](pdf/pythia/tuning.pdf)
* [random number generation](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/mc/rng.ipynb) [\[archived pdf\]](pdf/mc/rng.pdf)
* [integration and sampling](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/mc/integrate.ipynb) [\[archived pdf\]](pdf/mc/integreate.pdf)
* [hard process](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/mc/hard_process.ipynb) [\[archived pdf\]](pdf/mc/hard_process.pdf)
* [parton showers](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/mc/parton_shower.ipynb) [\[archived pdf\]](pdf/mc/parton_shower.pdf)
* [hadronization](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/mc/hadronization.ipynb) [\[archived pdf\]](pdf/mc/hadronization.pdf)
* [visualization](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/vistas/vistas.ipynb) [\[archived pdf\]](pdf/vistas/vistas.pdf)

## Machine Learning Notebooks

* [linear and logistic regression](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/regression.ipynb) [\[archived pdf\]](pdf/ml/regression.pdf)
* [classification](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/classify.ipynb) [\[archived pdf\]](pdf/ml/classify.pdf)
* [auto-differentiation](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/autodiff.ipynb) [\[archived pdf\]](pdf/ml/autodiff.pdf)
* [neural networks](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/neural_networks.ipynb) [\[archived pdf\]](pdf/ml/neural_networks.pdf)
* [neural networks with standard tools](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/neural_networks_jax_pytorch_tensorflow.ipynb) [\[archived pdf\]](pdf/ml/neural_networks_jax_pytorch_tensorflow.pdf)
* [unsupervised learning](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/unsupervised.ipynb) [\[archived pdf\]](pdf/ml/unsupervised.pdf)
* [normalizing flows](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/flows.ipynb) [\[archived pdf\]](pdf/ml/flows.pdf)
* [boosted decision trees](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/trees.ipynb) [\[archived pdf\]](pdf/ml/trees.pdf)
* [top tagging](https://colab.research.google.com/github/mcgen-ct/tutorials/blob/2025-cteq/ml/top_tagging.ipynb) [\[archived pdf\]](pdf/ml/top_tagging.pdf)

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
* Ahmed Youssef
* Jure Zupan

## MCgen Schools

These tutorials have been in used in the following schools, either directly run by MCgen, or with support from MCgen. Each school listed below has a frozen branch which archives the tutorials as used for that specific school.

* 2025/06/15 - 2025/06/27 [CTEQ-MCgen](https://indico.cern.ch/event/1497407/) with branch [`2025-cteq`](../../tree/2025-cteq)
  - `2025-cteq-v0`: tag for initial archived branch commit
