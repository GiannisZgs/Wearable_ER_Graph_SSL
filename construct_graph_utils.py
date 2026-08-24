# Copyright (c) 2024 Ioannis Ziogas
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import torch
import pandas as pd
from torch_geometric.utils import from_networkx

def compute_adjacency(X,y,method,metric,use_labels = True,thresh = None,nearest_neighbors = None,farthest_neighbors = None,sup_mode = None,X_labels = None,X_no_labels = None):
    "Construct a graph from the data based on three methods: correlation, nearest neighbors, nearest and farthest neighbors"
    
    if method == 'corr':
        if metric == 'cosine':
            corr_mat = cosine_similarity(X)
        #Pearson correlation
        elif metric == 'pearson':
            try:
                corr_mat = np.corrcoef(X, rowvar=True)
            except AttributeError:
                corr_mat = np.corrcoef(X.astype(float), rowvar=True) #rowvar because we get correlation between samples
            #get the positive correlation matrix and assign as the adjacency matrix
        A = np.where(corr_mat > thresh, corr_mat, 0) #corr_mat.where(corr_mat>corr_thresh,0)

        #corr_thresholds = np.linspace(0, 1, 100)
        #explore_graph_features(X,corr_thresholds)
        #print(pos_corr_mat)
        #G = nx.from_pandas_adjacency(pos_corr_mat)

    elif method == 'nearest_neighbors':
        from sklearn.neighbors import kneighbors_graph
        if metric == 'cosine':
            A = kneighbors_graph(X, nearest_neighbors, mode='connectivity', metric = 'cosine', include_self=False)
        elif metric == 'euclidean':
            A = kneighbors_graph(X, nearest_neighbors, mode='connectivity', metric = 'minkowski',p = 2, include_self=False)
        elif metric == 'manhattan':
            A = kneighbors_graph(X, nearest_neighbors, mode='connectivity', metric = 'minkowski',p = 1, include_self=False)

        A = A.toarray()
    elif method == 'nearest_farthest_neighbors':
        #Cosine similarity
        #Construct a graph where each node is connected to its nearest neighbors and farthest neighbors
        #The nearest neighbors are the ones with the highest similarity if they also have the same class
        if metric == 'cosine':
            sim_mat = cosine_similarity(X)
        elif metric == 'euclidean':
            sim_mat = euclidean_distances(X)
        elif metric == 'manhattan':
            sim_mat = manhattan_distances(X)
        
        # Set similarities of the samples that don't have the same label to a low value
        sim_mat_masked = sim_mat.copy()
        if sup_mode == 'semi_sup' and X_labels is not None and use_labels == True:
            "Labeled submatrix"
            sim_mat_labeled = sim_mat[:X_labels.shape[0],:X_labels.shape[0]]
            sim_mat_labeled_masked = sim_mat_labeled.copy()
            same_label_mask = y[:, None] == y[None, :]
            sim_mat_labeled_masked[~same_label_mask] = 0
            "Unlabeled submatrix"
            sim_mat_unlabeled = sim_mat.copy()
            #Set the labeled-labeled part to 0
            sim_mat_unlabeled[:X_labels.shape[0],:X_labels.shape[0]] = 0
        elif sup_mode is None and use_labels == True:
            # Create a 2D mask where mask[i, j] is True if sample i and sample j have the same label
            same_label_mask = y[:, None] == y[None, :]
            sim_mat_masked[~same_label_mask] = 0

        
        if sup_mode == 'semi_sup' and X_labels is not None and use_labels == True:
            "Labeled - work with the labeled nodes only"
            "For farthest neighbors work with the unmasked matrix"
            #Set lower diagonal to 0
            sim_mat_far = sim_mat.copy()
            sim_mat_far[X_labels.shape[0]:,X_labels.shape[0]:] = 0
            sim_mat_far = np.triu(sim_mat_far)
            sorted_indices_far = np.argsort(sim_mat_far, axis=1)
            farthest_inds_labeled = sorted_indices_far[:,:farthest_neighbors]
            "For nearest neighbors work with the masked matrix"
            #Set diagonal to 0
            np.fill_diagonal(sim_mat_labeled_masked, 0)
            #Set lower diagonal to 0
            sim_mat_labeled_masked = np.triu(sim_mat_labeled_masked)
            #Sort distances
            sorted_indices_near = np.argsort(sim_mat_labeled_masked, axis=1)
            #Get the nearest and farthest neighbors - sort in ascending order
            #[0,...................,end]
            #[farthest_neighbor,...,nearest_neighbor]
            nearest_inds_labeled = sorted_indices_near[:,-nearest_neighbors:]

            "Unlabeled - work with all nodes"
            #Set lower diagonal to 0
            sim_mat_unlabeled = np.triu(sim_mat_unlabeled)
            #Set diagonal to 0
            np.fill_diagonal(sim_mat_unlabeled, 0)
            sorted_indices = np.argsort(sim_mat_unlabeled, axis=1)
            farthest_inds_unlabeled = sorted_indices[:,:farthest_neighbors]
            nearest_inds_unlabeled = sorted_indices[:,-nearest_neighbors:]

            
            A = np.zeros_like(sim_mat)
            for i in range(A.shape[0]):
                if i < X_labels.shape[0]:
                    A[i, nearest_inds_labeled[i]] = 1
                    A[nearest_inds_labeled[i], i] = 1
                    A[i, farthest_inds_labeled[i]] = -1
                    A[farthest_inds_labeled[i], i] = -1
                else:
                    A[i, nearest_inds_unlabeled[i]] = 1
                    A[nearest_inds_unlabeled[i], i] = 1
                    A[i, farthest_inds_unlabeled[i]] = -1
                    A[farthest_inds_unlabeled[i], i] = -1
            

        else:
            "For farthest neighbors work with the unmasked matrix"
            #Set lower diagonal to 0
            sim_mat = np.triu(sim_mat)
            sorted_indices_far = np.argsort(sim_mat, axis=1)
            farthest_inds = sorted_indices_far[:,:farthest_neighbors]

            "For nearest neighbors work with the masked matrix"
            #Set diagonal to 0
            np.fill_diagonal(sim_mat_masked, 0)
            #Set lower diagonal to 0
            sim_mat_masked = np.triu(sim_mat_masked)
            #Sort distances
            sorted_indices_near = np.argsort(sim_mat_masked, axis=1)
            #Get the nearest and farthest neighbors - sort in ascending order
            #[0,...................,end]
            #[farthest_neighbor,...,nearest_neighbor]
            nearest_inds = sorted_indices_near[:,-nearest_neighbors:]
            #farthest_inds = sorted_indices[:,:farthest_neighbors]
            #Create the adjacency matrix
            A = np.zeros_like(sim_mat_masked)
            for i in range(A.shape[0]):
                A[i, nearest_inds[i]] = 1
                A[nearest_inds[i], i] = 1
                A[i, farthest_inds[i]] = -1
                A[farthest_inds[i], i] = -1
    
    "Return the adjacency matrix of the graph"
    return A

def construct_from_np_adjacency(A,X):
    "Construct undirected graph from adjacency matrix A and node embeddings X"
    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))
    "Add node embeddings as node attributes"
    for i in range(len(X)):
        if isinstance(X,pd.DataFrame):
            try:
                G.nodes[i]['embedding'] = X.loc[i].tolist()
            except KeyError:
                G.nodes[i]['embedding'] = X.iloc[i].tolist()
        elif isinstance(X,np.ndarray):
            G.nodes[i]['embedding'] = X[i].tolist()
    return G

def get_graph_masks(data,train_inds = None,val_inds = None,test_inds = None):
    "Create masks"
    train_mask = torch.zeros(data.num_nodes, dtype=bool)
    val_mask = torch.zeros(data.num_nodes, dtype=bool)
    test_mask = torch.zeros(data.num_nodes, dtype=bool)

    if train_inds is not None:
        if len(train_inds) == data.num_nodes:
            train_mask[:] = True
        else:
            train_mask[train_inds] = True
    if val_inds is not None:
        val_mask[val_inds] = True
    if test_inds is not None:
        test_mask[test_inds] = True

    "Add the masks to the data object"
    if train_inds is not None:
        data.train_mask = train_mask
    if val_inds is not None:
        data.val_mask = val_mask
    if test_inds is not None:
        data.test_mask = test_mask

    return data

def _add_unseen_nodes(A_old,X_new, X_old, connection_method,metric,y = None,use_labels = False,thresh = None,nearest_neighbors = None,farthest_neighbors = None):
    # Compute the adjacency matrix for the new nodes by taking into account old nodes (all nodes)
    X_total = np.vstack((X_old,X_new))
    A_temp = compute_adjacency(X_total,y,connection_method,metric,use_labels = use_labels,thresh = thresh,nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors)
    # Adjacency matrix for validation nodes only 
    A_val = A_temp[A_old.shape[0]:,A_old.shape[0]:]
    expected_val_shape = (X_new.shape[0],X_new.shape[0])
    if not A_val.shape == expected_val_shape:
        raise ValueError(f"Expected shape {expected_val_shape} but got {A_val.shape}")
    #New validation indices: indices of the newly added nodes
    new_val_inds = pd.Index(range(A_old.shape[0],X_old.shape[0]+X_new.shape[0]))
    #New training indices: indices of the old nodes
    new_train_inds = pd.Index(range(0,X_old.shape[0]))
    # Create a block matrix with zeros for the off-diagonal blocks
    A_new = np.block([
        [A_old, A_temp[:A_old.shape[0],A_old.shape[0]:]],   #np.zeros((A_old.shape[0], A_val.shape[0]))],
        [A_temp[A_old.shape[0]:,:A_old.shape[0]], A_val] #np.zeros((A_val.shape[0], A_old.shape[0]))
    ])

    return A_new, new_train_inds, new_val_inds

def inductive_val(A_train,X_val,X_train,y_target,train_labels,val_labels,construction_method,distance_metric,mode = 'val',use_labels = False,nearest_neighbors = None,farthest_neighbors = None):
    "Add validation nodes to the train graph"
    "Dont use labels for the nearest neighbors connections"
    A_val,new_train_inds, new_val_inds = _add_unseen_nodes(A_train,X_val,X_train,construction_method,distance_metric,y = None,use_labels = False,nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors)
    "Construct graph from adjacency matrix"
    X_temp = np.vstack((X_train,X_val))
    G_val = construct_from_np_adjacency(A_val,X_temp)
    data_val = from_networkx(G_val)
    "Get labels"
    y_target_val = torch.cat((train_labels, val_labels), dim=0)
    #y_target_val = torch.tensor(y_target[val_inds], dtype=torch.long)
    data_val.num_classes = len(np.unique(y_target))
    data_val.x = data_val.embedding
    data_val.y = y_target_val
    "Convert data labels to tensors"
    data_val.x = data_val.x.clone().detach() #torch.tensor(data_val.x, dtype=torch.float)
    data_val.num_features = len(data_val.x[0])
    #Try to see if there is any difference by also giving train mask
    if mode == 'val':
        data_val = get_graph_masks(data_val,val_inds = new_val_inds)
    elif mode == 'test':
        data_val = get_graph_masks(data_val, test_inds = new_val_inds)
    
    return data_val,G_val