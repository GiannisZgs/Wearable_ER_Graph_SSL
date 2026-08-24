"""
example_methodology.py
=======================

Walkthrough of the paper's method:
graph construction -> graph augmentation -> shared-encoder model -> combined
supervised + semi-supervised + self-supervised loss -> inductive evaluation.

This script shows one fold and a handful of epochs instead of the real 47-fold x 200-epoch run, and drops the
logging/plotting/checkpointing scaffolding to keep the methodology visible.

NOT STANDALONE-RUNNABLE: `load(...)` below expects the pickle produced by
`preproc_forGraph_v2.py` (itself downstream of
`first_processing.py` + `feature_extraction.py` + `analysis.py`), i.e. a
processed K-EmoPhone feature matrix. Run the full data pipeline first (see
README.md, "Pipeline / usage"), then point PATH_INTERMEDIATE below at it.
For the complete 47-fold leave-one-subject-out loop with checkpointing,
pseudo-label majority voting, and result logging, see
`self_and_semi_supervised_inductive3.py` directly.
"""

import os
import numpy as np
import torch
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from torch_geometric.utils import from_networkx

from utils import load
from training_utils import cv_split, process_label_info, train_SSL, test
from construct_graph_utils import compute_adjacency, construct_from_np_adjacency, get_graph_masks, inductive_val
from models import GCN_basic_projection
import transforms as graph_transforms          # node/feature masking, Gaussian noise (paper Section II-B3, a/b/c)
import GCL.augmentors as A                      # edge removal (paper Section II-B3, d) -- pip install PyGCL

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Load processed features and pick a task (paper Section III-A/B)
# ---------------------------------------------------------------------------
PATH_INTERMEDIATE = os.path.join(os.path.dirname(os.getcwd()), 'intermediate_ass4')
label = 'arousal_bin'  # or 'valence_bin'

X, y, group, t, _ = load(os.path.join(PATH_INTERMEDIATE, 'all_ScheduledResponseValid_all_labels.pkl'))
X, y, group, old_index = process_label_info(X, y, group, label)
y_target = torch.from_numpy(y[label].values)

# ---------------------------------------------------------------------------
# 2. Config -- the exact values used for the paper's arousal results
#    (self_and_semi_supervised_inductive3.py:79-198)
# ---------------------------------------------------------------------------
subjects = 47
K_train_percentage = 0.25 if label == 'arousal_bin' else 0.2       # ~20%/25% of labels (paper abstract)
K_train = int(np.floor(K_train_percentage * subjects))             # labeled-subject POOL size for this fold
labeled_use_percentage, unlabeled_use_percentage = 1, 0.5
Ls_train = int(K_train * labeled_use_percentage)                   # labeled subjects sampled EACH epoch
Us_train = int(np.ceil(K_train * unlabeled_use_percentage))        # unlabeled subjects sampled EACH epoch

construction_method = 'nearest_farthest_neighbors'                 # paper Section II-A
distance_metric = 'cosine'
nearest_neighbors, farthest_neighbors = 2, 1                        # k=2, m=1 (paper Section III-C)
sup_mode = 'self_sup'                                                # activates the labeled/unlabeled split in compute_adjacency

hidden_channels, num_layers, activation, dropout, use_lnorm = 96, 3, 'tanh', 0.5, False
lr, weight_decay, label_smoothing = 0.0055, 1e-5, 0.1
lambda1, lambda2 = 0.3, 0.2                                          # semi-sup / self-sup loss weights (paper Eq. 1)

augmentation = 'edge_removing'                                      # the augmentation used for the headline result
aug_prob = 0.15 if label == 'arousal_bin' else 0.05
graph_aug = A.EdgeRemoving(pe=aug_prob)
# Alternatives (paper Section II-B3): graph_transforms.Completion(p=..., masking='node')      -> Node Masking
#                                      graph_transforms.Completion(p=..., masking='feature')   -> Node Attribute Masking
#                                      graph_transforms.Denoising(p=..., mu=0.0, sigma=1.0)    -> Gaussian Noise Addition

SELECT_SVC = SelectFromModel(
    estimator=LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=1e-3, C=15e-3, max_iter=5000),
    threshold=1e-5,
)

# ---------------------------------------------------------------------------
# 3. One leave-one-subject-out fold (paper Section III-C: 43/3/1 train/dev/test)
#    -- the real script loops over all 47 folds; here we just take the first.
# ---------------------------------------------------------------------------
fold_iter = cv_split(
    X, y[label], group, SELECT_SVC,
    train_size=0.60, val_size=0.15,
    cv_mode='leave_one_out', subgraph_sampling='random_both', K_train=K_train,
)
(X_sel, X_labeled_train, X_unlabeled_train, X_val, X_test,
 labeled_train_inds, unlabeled_train_inds, val_inds, test_inds, n_splits) = next(fold_iter)

labeled_train_groups_unique = np.unique(group[labeled_train_inds])
unlabeled_train_groups_unique = np.unique(group[unlabeled_train_inds])
val_labels, test_labels = y_target[val_inds], y_target[test_inds]

num_features, num_classes = X_sel.shape[1], len(np.unique(y_target.numpy()))
model = GCN_basic_projection(
    num_layers=num_layers, activation=activation, dropout=dropout,
    hidden_channels=hidden_channels, input_size=num_features, output_size=num_classes,
    use_lnorm=use_lnorm,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# ---------------------------------------------------------------------------
# 4. Epoch loop -- the real script runs `epochs=200`; a handful here is enough
#    to show that the SUBGRAPH IS RESAMPLED EVERY EPOCH (paper Section II-A,
#    "at each epoch, a single subgraph G_s is constructed from (L_s, U_s)").
# ---------------------------------------------------------------------------
N_DEMO_EPOCHS = 3

for epoch in range(N_DEMO_EPOCHS):

    # --- 4a. Subgraph sampling: draw L_s labeled + U_s unlabeled subjects from
    #     this fold's pools (self_and_semi_supervised_inductive3.py:397-422).
    #     The real script also tracks which unlabeled subjects were already
    #     used this fold so they aren't resampled until the pool is exhausted;
    #     omitted here (plain sampling-with-replacement-across-epochs) for clarity. ---
    Ls_groups = np.random.choice(labeled_train_groups_unique, size=Ls_train, replace=False)
    Us_groups = np.random.choice(unlabeled_train_groups_unique, size=Us_train, replace=False)
    Ls_inds = group[group.isin(Ls_groups)].index
    Us_inds = group[group.isin(Us_groups)].index

    Xs_L, Xs_U = X_sel.iloc[Ls_inds], X_sel.iloc[Us_inds]
    X_train = np.vstack((Xs_L, Xs_U))                      # this epoch's subgraph nodes: G_s = (L_s, U_s)

    # --- 4b. Graph construction (paper Section II-A, Eq. via k-NN/m-farthest) ---
    A_train = compute_adjacency(
        X_train, y_target[Ls_inds], construction_method, distance_metric,
        use_labels=True, nearest_neighbors=nearest_neighbors, farthest_neighbors=farthest_neighbors,
        sup_mode=sup_mode, X_labels=Xs_L, X_no_labels=Xs_U,
    )
    G_train = construct_from_np_adjacency(A_train, X_train)
    data = from_networkx(G_train)
    data.x = data.embedding.clone().detach()
    data.num_features = num_features
    data.num_classes = num_classes

    # Labeled nodes keep their true label; unlabeled nodes get a placeholder (2)
    # that is never read as ground truth -- only `train_mask`/`val_mask` matter.
    y_pseudo_placeholder = torch.full((len(Us_inds),), 2, dtype=torch.long)
    data.y = torch.cat((y_target[Ls_inds].clone().detach(), y_pseudo_placeholder), dim=0)

    # `train_mask` = labeled nodes (supervised loss), `val_mask` = unlabeled
    # nodes of THIS subgraph (semi-supervised loss) -- not a held-out split.
    data = get_graph_masks(
        data,
        train_inds=np.arange(len(Ls_inds)),
        val_inds=np.arange(len(Ls_inds), len(Ls_inds) + len(Us_inds)),
    )

    # --- 4c. Augmented view G~_s (paper Section II-B3). `data.weight` holds the
    #     signed +1/-1 edge weights from A_train (set by from_networkx, since
    #     construct_from_np_adjacency's nx.from_numpy_array stores A's values
    #     under the 'weight' edge attribute). ---
    x_aug, edge_index_aug, edge_weight_aug = graph_aug(x=data.x, edge_index=data.edge_index, edge_weight=data.weight)
    data_aug = data.clone()
    data_aug.x, data_aug.edge_index, data_aug.weight = x_aug, edge_index_aug, edge_weight_aug

    # --- 4d. Combined loss: L_sup + lambda1*L_semi + lambda2*L_self (paper Eq. 1) ---
    loss, pseudolabels = train_SSL(
        model, data, data_aug, optimizer, criterion, device,
        ssl_task=augmentation, lambda1=lambda1, lambda2=lambda2,
    )
    print(f'[epoch {epoch}] fold 0, label={label}: train loss = {loss.item():.4f}')

    # --- 4e. Inductive validation: attach held-out subjects to the JUST-TRAINED
    #     subgraph without any further gradient updates (paper Section II-C) ---
    y_train_with_pseudo = torch.cat((y_target[Ls_inds], pseudolabels.cpu()), dim=0)
    data_val, _ = inductive_val(
        A_train, X_val, X_train, y_target, y_train_with_pseudo, val_labels,
        construction_method, distance_metric, mode='val',
        use_labels=True, nearest_neighbors=nearest_neighbors, farthest_neighbors=farthest_neighbors,
    )
    val_metrics = test(data_val.val_mask, model, data_val, criterion, device)
    print(f'[epoch {epoch}] val accuracy = {val_metrics["acc"]:.4f}, val F1 macro = {val_metrics["f1_macro"]:.4f}')

# ---------------------------------------------------------------------------
# The real script additionally: checkpoints the model on every val-loss
# improvement, immediately scores the held-out TEST subject at that
# checkpoint via a second inductive_val(..., mode='test') call, repeats this
# for all 47 folds, takes a majority vote over each unlabeled subject's
# pseudolabels across epochs, and writes per-fold metrics to disk. See
# self_and_semi_supervised_inductive3.py for that full loop.
# ---------------------------------------------------------------------------
