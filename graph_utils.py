import pandas as pd
import scipy.io 
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gc
import os

PATH_PLOTS = os.path.join(os.getcwd(),'plots')

def explore_graph_features(X: pd.DataFrame,
                           corr_thresholds: list[float],

                           ):

    #Initialize graph features
    num_nodes = []
    num_edges = []
    density = []
    avg_degree = []
    clustering_coeff = []


    for corr_thresh in corr_thresholds:
        # Compute the positive correlation matrix
        corr_mat = np.corrcoef(X, rowvar=True) #X.T.corr()
        pos_corr_mat = np.where(corr_mat > corr_thresh, corr_mat, 0) #corr_mat.where(corr_mat>corr_thresh,0)

        #pos_corr_mat = X.T.corr().where(lambda x: x > corr_thresh, 0)

        # Create a graph from the correlation matrix
        #G = nx.from_pandas_adjacency(pos_corr_mat)
        G = nx.from_numpy_array(pos_corr_mat)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Add the number of nodes and edges to the lists
        num_nodes.append(G.number_of_nodes())
        num_edges.append(G.number_of_edges())
        density.append(nx.density(G))
        avg_degree.append(sum(dict(G.degree()).values()) / G.number_of_nodes())
        clustering_coeff.append(nx.average_clustering(G))
        #degree_distribution = np.array(list(dict(G.degree()).values()))

        del G, pos_corr_mat
        gc.collect()

    # Plot the number of nodes and edges against the correlation thresholds
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.plot(corr_thresholds, num_edges, label='Number of edges')
    plt.xlabel('Correlation threshold')
    plt.ylabel('Number of edges')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(corr_thresholds, density, label='Density')
    plt.xlabel('Correlation threshold')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(corr_thresholds, avg_degree, label='Average Node Degree')
    plt.xlabel('Correlation threshold')
    plt.ylabel('Avg. Node Degree')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(corr_thresholds, avg_degree, label='Clustering Coefficient')
    plt.xlabel('Correlation threshold')
    plt.ylabel('Clustering Coeff.')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_PLOTS,'graph_features_vs_corr_thresh.png'))
    plt.show()

    return None