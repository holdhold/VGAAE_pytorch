import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


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


def get_trueLab(cellList, df):
    true_lab = []
    for cell in cellList:
        cell_df = df[df.Cell == cell]
        cell_label = cell_df['assigned_cluster'].values
        true_lab.append(cell_label[0])
    return true_lab


def getMetrics(dataType, dataName):
    df = pd.read_csv('dataset/{}/{}_ground_truth.csv'.format(dataType, dataName))
    true_lab = df['assigned_cluster'].values
    true_lab = np.array(true_lab)
    types = np.unique(true_lab)
    ids = np.arange(0, len(types))
    dict1 = {}
    dict1 = dict(zip(ids, types))
    for id, type in dict1.items():
        for i in range(len(true_lab)):
            if true_lab[i] == type:
                true_lab[i] = id
    pred_labels = np.load('results/{}/cluster.npy'.format(dataName))
    ARI = adjusted_rand_score(true_lab, pred_labels)
    NMI = normalized_mutual_info_score(true_lab, pred_labels)
    CA = cluster_accuracy(true_lab, pred_labels)
    print('{}--ARI--{:.4f},NMI--{:.4f},CA--{:.4f}'.format(dataName, ARI, NMI, CA))
    return round(ARI, 4), round(NMI, 4), round(CA, 4)
