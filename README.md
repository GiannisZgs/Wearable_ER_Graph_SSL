# Graph_SSL_WearableEmoRec

**Self-Supervised Graph Representation Learning for In-The-Wild Wearable and Smartphone based Emotion Recognition**

Official code for our ICASSP 2025 paper. We formulate wearable/smartphone emotion recognition (arousal & valence) as an **inductive graph node classification** problem and train a single graph neural network with a **joint supervised + semi-supervised + self-supervised loss**, sampling small subgraphs of labeled and unlabeled subjects at every training epoch.

📄 Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10888648) · DOI: [10.1109/ICASSP49660.2025.10888648](https://doi.org/10.1109/ICASSP49660.2025.10888648)
📚 Core methodological contribution: [METHODOLOGY.md](METHODOLOGY.md) · Annotated code walkthrough: [example_methodology.py](example_methodology.py)

![Our proposed SSL-based graph representation learning framework](Fig1_SSL_graph.png)
*Fig. 1 — Features from continuous wearable/smartphone time series are embedded as node features; a small subgraph is sampled every epoch and connected via a signed k-nearest/m-farthest-neighbor graph. A shared GCN encoder is trained jointly on the raw subgraph (supervised classification loss `L_sup` + semi-supervised pseudo-labeling loss `L_semi`) and on an augmented view of the same subgraph (self-supervised consistency loss `L_self`).*

## Core contributions

The paper's core contributions — the signed nearest/farthest-neighbor graph construction, the four graph augmentation (SSL pretext) tasks, the shared-encoder dual-head architecture, and the combined supervised + semi-supervised + self-supervised subgraph-sampling training loop — are documented in detail, with exact file/line references, in **[METHODOLOGY.md](METHODOLOGY.md)**. **[example_methodology.py](example_methodology.py)** is a walkthrough of a single training fold/epoch using the real functions and hyperparameters, cross-referenced against the paper's equations.

## Table of contents
- [Core Contributions](#core-contributions)
- [Repository guide](#repository-guide)
- [Setup](#setup)
- [Data](#data)
- [Pipeline / usage](#pipeline--usage)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Repository guide

**Core pipeline** — the path from raw data to a trained model:
| Stage | File(s) |
|---|---|
| Data preprocessing | [`first_processing.py`](first_processing.py) (see [Acknowledgements](#acknowledgements)) → [`processing_utils.py`](processing_utils.py) → [`feature_extraction.py`](feature_extraction.py) → [`analysis.py`](analysis.py); [`explore_data.py`](explore_data.py) is a standalone EDA helper |
| Graph-ready preprocessing | [`preproc_forGraph_v2.py`](preproc_forGraph_v2.py) (feature selection, subject splits) + [`graph_utils.py`](graph_utils.py) (graph-density diagnostics) |
| Graph construction | [`construct_graph_utils.py`](construct_graph_utils.py) |
| Graph augmentations | [`transforms.py`](transforms.py) + external [`GCL.augmentors`](https://github.com/PyGCL/PyGCL) |
| Model | [`models.py`](models.py) → `GCN_basic_projection` |
| Training engine | [`training_utils.py`](training_utils.py) → `cv_split()`, `train()`, `train_SSL()`, `test()` |
| **Main training script** | [`self_and_semi_supervised_inductive3.py`](self_and_semi_supervised_inductive3.py) 
| Result aggregation | [`see_results_graphs.py`](see_results_graphs.py), [`ablations.py`](ablations.py) |

See [METHODOLOGY.md](METHODOLOGY.md) for how these fit together.

**Companion / ablation scripts** — each isolates one arm of the method or one hyperparameter sweep reported in the paper:
| File | What it isolates |
|---|---|
| [`semi_supervised_inductive2.py`](semi_supervised_inductive2.py) (+ `_tests`, `_tests_bayes_opt`) | Semi-supervised-only arm (no self-supervised loss) + its hyperparameter search |
| [`self_and_supervised_inductive4.py`](self_and_supervised_inductive4.py) | Self-supervised + supervised, no semi-supervised term |
| [`supervised_induct_with_unlabeled.py`](supervised_induct_with_unlabeled.py) / [`supervised_induct_without_unlabeled.py`](supervised_induct_without_unlabeled.py) | Purely supervised arm through the same subgraph-sampling pipeline (`λ1=λ2=0`), with vs. without unlabeled subjects structurally present in the graph — Table I's supervised baseline rows |
| [`semi_supervised_induct_ablate_labeled_size.py`](semi_supervised_induct_ablate_labeled_size.py) | Labeled-subject-count ablation (Fig. 2a) |
| [`self_and_supervised_induct_ablate_aug_prob.py`](self_and_supervised_induct_ablate_aug_prob.py) | SSL augmentation-probability ablation (Fig. 2b/c) |
| [`self_and_semi_supervised_inductive3_tests_bayes_opt.py`](self_and_semi_supervised_inductive3_tests_bayes_opt.py) | Bayesian hyperparameter search (`λ1`, `λ2`, label smoothing, augmentation probability) for the full model |

**Classical ML baseline** (independent of the graph pipeline — a non-graph comparison track built on the same K-EmoPhone features): [`cross_val.py`](cross_val.py), [`classif_utils.py`](classif_utils.py), [`eval_utils.py`](eval_utils.py), [`supervised_ml.py`](supervised_ml.py), [`see_feature_importance.py`](see_feature_importance.py), [`see_results.py`](see_results.py) — Random Forest / XGBoost / dummy classifiers with LOGO cross-validation, extracted from `analysis.ipynb`'s cross-validation section.

## Setup

### Clone
```bash
git clone https://github.com/GiannisZgs/Wearable_ER_Graph_SSL.git
cd Wearable_ER_Graph_SSL
```

### Environments

The project spans two stages that were developed with two separate conda environments:

| Environment file | Env name | Python | Purpose |
|---|---|---|---|
| [`envGraphSSL.yml`](envGraphSSL.yml) | `graphSSL` | 3.12, CPU | Data preprocessing, feature extraction, EDA (pandas, scikit-learn, gensim, networkx) |
| [`cuda_env_files.yaml`](cuda_env_files.yaml) | `cuda_env` | 3.11, CUDA 12.1 | Graph construction, GNN training (PyTorch 2.1.0, PyTorch Geometric 2.5.1) |

```bash
conda env create -f envGraphSSL.yml
conda env create -f cuda_env_files.yaml
```

Moreover, the below packages should be installed manually:
```bash
conda activate cuda_env
pip install PyGCL               # imported as `GCL` (graph augmentations, e.g. GCL.augmentors.EdgeRemoving)
pip install bayesian-optimization  # imported as `bayes_opt` (only needed for the *_bayes_opt.py hyperparameter-search scripts)
```

## Data

The [K-EmoPhone dataset](https://www.nature.com/articles/s41597-023-02248-2) (Kang et al., 2023) is publicly available — see the paper for the access procedure. The preprocessing code expects it laid out one level above the repo as:
```
../data/
├── EsmResponse.csv
├── UserInfo.csv
└── Sensor/
    └── P<id>/
        ├── Acceleration.csv
        ├── EDA.csv
        ├── HR.csv
        ├── RRI.csv
        ├── SkinTemperature.csv
        └── ... (27 sensor/phone-log types total, see analysis.ipynb's `DATA_TYPES`)
```

## Pipeline / usage

There is no CLI — each script is a `#%%`-delimited cell script with hyperparameters as module-level variables near the top (e.g. `label = 'arousal_bin'`, `augmentations = ['edge_removing']`). Edit those, then run top-to-bottom. Run order:

1. **Preprocess raw sensor/phone/ESM data → features.** Run [`analysis.ipynb`](analysis.ipynb) end-to-end (recommended, matches the acknowledged reference pipeline), or the equivalent scripts `first_processing.py` → `feature_extraction.py` → `analysis.py`. Produces `intermediate/*_all_labels.pkl` (`X, y, group, t`).
2. **Prepare graph-ready splits.** Run [`preproc_forGraph_v2.py`](preproc_forGraph_v2.py) — per-fold feature normalization/selection and labeled/unlabeled subject splitting (via `training_utils.cv_split`).
3. **Train.** Run [`self_and_semi_supervised_inductive3.py`](self_and_semi_supervised_inductive3.py) for the full method (or one of the companion scripts above for a specific ablation/arm). See [METHODOLOGY.md](METHODOLOGY.md) for what happens inside. Set `label = 'arousal_bin'` or `'valence_bin'` and re-run for the other task.
4. **Inspect results.** [`see_results_graphs.py`](see_results_graphs.py) aggregates per-fold metrics with confidence intervals; [`ablations.py`](ablations.py) plots the labeled-size and augmentation-probability sweeps.

## Acknowledgements

- The data-processing pipeline in [`analysis.ipynb`](analysis.ipynb) (and its script equivalents `first_processing.py`, `processing_utils.py`, `feature_extraction.py`, `analysis.py`, and the classical-ML baseline `cross_val.py`/`classif_utils.py`/`eval_utils.py`/`supervised_ml.py`/`see_feature_importance.py`) is adapted from the official **[K-EmoPhone_SupplementaryCodes](https://github.com/Kaist-ICLab/K-EmoPhone_SupplementaryCodes)** repository by Kaist-ICLab, released alongside:
  > S. Kang, W. Choi, C. Y. Park, N. Cha, A. Kim, A. H. Khandoker, L. Hadjileontiadis, H. Kim, Y. Jeong, and U. Lee, "K-EmoPhone: A Mobile and Wearable Dataset with In-Situ Emotion, Stress, and Attention Labels," *Scientific Data*, vol. 10, no. 1, pp. 1–21, 2023.
- The subgraph-sampling training scheme is inspired by Shirian et al.'s self-supervised audio graphs (*IEEE JSTSP*, 2022) and by GraphSAINT (Zeng et al., ICLR 2020).
- Graph augmentations use [PyGCL](https://github.com/PyGCL/PyGCL) (`GCL.augmentors`), built on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ziogas2025selfsupervised,
  author    = {Ziogas, Ioannis and Hadjileontiadis, Leontios J. and Khandoker, Ahsan H. and Al Shehhi, Aamna},
  title     = {Self-Supervised Graph Representation Learning for In-The-Wild Wearable and Smartphone based Emotion Recognition},
  booktitle = {ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025},
  pages     = {1--5},
  doi       = {10.1109/ICASSP49660.2025.10888648}
}
```

I. Ziogas, L. J. Hadjileontiadis, A. H. Khandoker, and A. Al Shehhi, "Self-Supervised Graph Representation Learning for In-The-Wild Wearable and Smartphone based Emotion Recognition," in *2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025, pp. 1–5, doi: [10.1109/ICASSP49660.2025.10888648](https://doi.org/10.1109/ICASSP49660.2025.10888648).

Please also cite the K-EmoPhone dataset paper (above) if you use the data-processing pipeline.

## License

MIT — see [LICENSE](LICENSE).
