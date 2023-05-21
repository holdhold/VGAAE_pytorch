import numpy as np
import anndata
import pandas as pd
import scanpy as sc
import torch
import faiss
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MinMaxScaler
import h5py
import random


# 读取原始表达数据
def read_data(file_path, file_type):
    adata = []
    if file_type == 'csv':
        adata = anndata.read_csv(file_path)
    if file_type == 'normal':
        df_data = pd.read_csv(file_path)
        data = df_data.to_numpy().T
        ori_metrix = data[1:, 2:].astype(float)
        adata = anndata.AnnData(ori_metrix)
    if file_type == 'h5':
        adata = sc.read_hdf(file_path, 'X')
    return adata

# 随机子采样
def subsample_anndata(adata):
    seed = random.randint(1, 10000)
    sub_adata = sc.pp.subsample(adata, n_obs=500, copy=True, random_state=seed)
    return sub_adata


def subasmple_h5py(datasetType, datasetName):
    source_path = 'dataset/{}/{}.h5'.format(datasetType, datasetName)
    # Get length of files and prepare samples
    source_file = h5py.File(source_path, "r")
    dataset = source_file['X']
    indices = np.sort(np.random.choice(dataset.shape[0], 500, replace=False))
    target_path = 'dataset/{}/{}_subsample.h5'.format(datasetType, datasetName)
    target_file = h5py.File(target_path, "w")
    for k in source_file.keys():
        dataset = source_file[k]
        if k is 'X':
            dataset = dataset[indices, :]
        if k is 'Y':
            dataset = dataset[indices]
        dest_dataset = target_file.create_dataset(k, shape=(dataset.shape), dtype=np.float32)
        #
        dest_dataset.write_direct(dataset)
    target_file.close()
    source_file.close()


# 对原始数据进行预处理，先归一化再对数化
def preprocess_raw_data(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


# 准备训练数据
def prepare_training_data(adata):
    adata_hvg = adata.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=1000, inplace=True, flavor='seurat')
    # 获取前两千个高变量基因的细胞数据
    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    return adata_hvg, X_hvg


# 使用faiss库计算样本点之间的距离
def faiss_knn(data_nmupy, k, metric='euclidean'):
    # In Python, the matrices are always represented as numpy arrays.
    # The data type dtype must be float32.
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


# 计算距离
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
    # 将边列表保存为txt文件
    filname = 'process/{}_edgelist.txt'.format(datasetName)
    with open(filname, 'w') as f:
        edegs = [' '.join(e) + '\n' for e in edgelist]
        f.writelines(edegs)
    print(f'一共有{len(edgelist)}条边')
    return distances, neighbors, cutoff, edgelist


# 读取边列表
def load_separate_graph_edgelist(edgelist_path):
    edgelist = []
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist


# 构建邻接图
def create_graph(edges, X):
    num_nodes = X.shape[0]
    edge_index = np.array(edges).astype(int).T
    # torch_geometric.Data需要的参数为tensor型
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    scaler_X = torch.from_numpy(MinMaxScaler().fit_transform(X))
    data_obj = Data(edge_index=edge_index, x=scaler_X)
    return data_obj
