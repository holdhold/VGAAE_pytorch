import numpy as np
import anndata
import pandas as pd
import scanpy as sc
import torch
import faiss
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MinMaxScaler
import random


def read_data(file_path, file_type):
    adata = []
    if file_type == 'csv':
        adata = anndata.read_csv(file_path)
    return adata


def subsample_anndata(adata):
    seed = random.randint(1, 10000)
    sub_adata = sc.pp.subsample(adata, n_obs=500, copy=True, random_state=seed)
    return sub_adata

def preprocess_raw_data(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def prepare_training_data(adata):
    adata_hvg = adata.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=1000, inplace=True, flavor='seurat')
    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    return adata_hvg, X_hvg



def faiss_knn(data_nmupy, k, metric='euclidean'):
    data_nmupy = data_nmupy.astype(np.float32)
    data_nmupy = data_nmupy.copy(order='C')
    data_nmupy = np.ascontiguousarray(data_nmupy, dtype=np.float32)

    if metric == 'euclidean':
        index = faiss.IndexFlatL2(data_nmupy.shape[1])
    elif metric == 'manhattan':
        index = faiss.IndexFlat(data_nmupy, faiss.METRIC_L1)
    elif metric == 'cosine':
        index = faiss.IndexFlat(data_nmupy, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(data_nmupy)

    data_nmupy = np.ascontiguousarray(data_nmupy, dtype=np.float32)
    index.train(data_nmupy)
    assert index.is_trained
    index.add(data_nmupy)
    nprobe = data_nmupy.shape[0]
    index.nprobe = nprobe
    distances, neighbors = index.search(data_nmupy, k)
    return distances, neighbors



def get_edgelist(datasetName, X_hvg, k, type):
    if type == 'Faiss_KNN':
        distances, neighbors = faiss_knn(data_nmupy=X_hvg, k=k)
    cutoff = np.mean(np.nonzero(distances), axis=None)
    print(cutoff)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                distance = distances[i][j]
                if distance < cutoff:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)
    filname = 'process/{}_edgelist.txt'.format(datasetName)
    with open(filname, 'w') as f:
        edegs = [' '.join(e) + '\n' for e in edgelist]
        f.writelines(edegs)
    return distances, neighbors, cutoff, edgelist


def load_separate_graph_edgelist(edgelist_path):
    edgelist = []
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist


def create_graph(edges, X):
    num_nodes = X.shape[0]
    edge_index = np.array(edges).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    scaler_X = torch.from_numpy(MinMaxScaler().fit_transform(X))
    data_obj = Data(edge_index=edge_index, x=scaler_X)
    return data_obj
