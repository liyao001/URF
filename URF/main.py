#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File:      URF.py
# @Author:    Li Yao
# @Created:   28/03/2018 7:56 PM
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from Pycluster import kcluster


def build_synthetic_matrix(X):
    """
    Build synthetic matrix
    :param X:
    :return:
    """
    X_flattened = X.flatten()
    indices = np.random.choice(X.shape[0], X.shape[0] * X.shape[1], replace=True)
    synt_mat = X_flattened[indices]
    synt_mat = np.reshape(synt_mat, X.shape)
    return synt_mat


def proximity_matrix(rf, X, normalize=True):
    """
    Calculate proximity matrix
    :param rf:
    :param X:
    :param normalize:
    :return:
    """
    leaves = rf.apply(X)
    n_trees = leaves.shape[1]

    prox_mat = np.zeros((leaves.shape[0], leaves.shape[0]))

    for i in range(n_trees):
        a = leaves[:, i]
        prox_mat += 1 * np.equal.outer(a, a)

    if normalize:
        prox_mat = prox_mat / n_trees

    return prox_mat


def plot_proximity_matrix(mat, cmap=plt.cm.Reds, output=None):
    """
    Plot proximity matrix as a heatmap
    :param mat:
    :param cmap:
    :param output:
    :return:
    """
    plt.figure()
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    if output is not None:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()


def unsupervised_random_forest(X, **kwargs):
    """
    Unsupervised Random Forest
    :param X:
    :param kwargs:
    :return: (random forest, proximity matrix)
    """
    syn_mat = build_synthetic_matrix(X)
    X_merged = np.vstack((X, syn_mat))
    original_label = np.asarray([0] * X.shape[0])
    synthetic_label = np.asarray([1] * X.shape[0])
    y_merged = np.hstack((original_label, synthetic_label))

    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_merged, y_merged)

    prox_mat = proximity_matrix(clf, X)
    return clf, prox_mat


def random_forest_cluster(X, k=2, dissimilarity=True, **kwargs):
    """
    Random Forest Cluster
    :param X:
    :param k:
    :param dissimilarity:
    :param kwargs:
    :return:
    """
    clf, prox_mat = unsupervised_random_forest(X, **kwargs)

    if dissimilarity:
        prox_mat = 1 - prox_mat
    cluster_ids, error, n_found = kcluster(prox_mat, nclusters=k, method="m")

    return clf, prox_mat, cluster_ids


def map_label_with_marker(labels):
    """
    Map label with marker
    :param labels:
    :return:
    """
    markers = ('o', '*', '^', '<', '>', '8', 's', 'p', 'v', 'h', 'H', 'D', 'd', 'P', 'X',
               '.', ',', '1', '2', '3', '4', '+', 'x', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    return dict(zip(list(set(labels)), markers))


def plot_cluster_result(prox_mat, cluster_ids, n_components=2, marker=None, output=None, show_legend=True, pre_computed_sim=True, **kwargs):
    """
    Plot cluster result
    :param prox_mat:
    :param cluster_ids:
    :param n_components:
    :param marker:
    :param output:
    :param show_legend:
    :param kwargs:
    :return:
    """
    if pre_computed_sim:
        mds = MDS(n_components=n_components, dissimilarity="precomputed", **kwargs)
    else:
        mds = MDS(n_components=n_components, dissimilarity="euclidean", **kwargs)
    x_transformed = mds.fit_transform(prox_mat)
    plt.figure()
    if marker is not None:
        markers = map_label_with_marker(cluster_ids)
        color_set = {}
        for k, v in enumerate(list(set(marker))):
            color_set[v] = "C%d" % k
        for i in list(set(marker)):
            for j in list(set(cluster_ids)):
                indecies = np.logical_and(cluster_ids == j, marker == i)
                plt.scatter(x_transformed[indecies, 0], x_transformed[indecies, 1],
                            c=color_set[i], marker=markers[j], label="Y: "+str(i)+"-C: "+str(j))
    else:
        plt.scatter(x_transformed[:, 0], x_transformed[:, 1], c=cluster_ids, cmap=plt.cm.Spectral)

    if show_legend:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    if output is not None:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(len(list(set(y))))

    clf, prox_mat, cluster_ids = random_forest_cluster(X, k=3, max_depth=20, random_state=0)
    plot_cluster_result(prox_mat, cluster_ids, 2, y)
