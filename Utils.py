import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import logging
from logging import handlers
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from PIL import Image


def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return acc


def wasserstein_distance(p, q):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(p.detach().numpy().flatten(), q.detach().numpy().flatten())


def Jaccard_Index(before_recluster, after_recluster):
    A = len(before_recluster)
    B = len(after_recluster)
    Unio = [i for i in before_recluster if i in after_recluster]
    C = len(Unio)
    return C / (A + B - C)


def get_true_label(cellList, df):
    true_lab = []
    for cell in cellList:
        cell_df = df[df.Cell == cell]
        cell_label = cell_df['assigned_cluster'].values
        print(cell_label)
        true_lab.append(cell_label[0])
    return true_lab
