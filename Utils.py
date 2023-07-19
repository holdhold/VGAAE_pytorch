import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import logging
from logging import handlers
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from PIL import Image


# 聚类精确度
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


# wasserstein距离
def wasserstein_distance(p, q):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(p.detach().numpy().flatten(), q.detach().numpy().flatten())


# Jaccard Similarity Index :衡量集合之间的相似度
def Jaccard_Index(before_recluster, after_recluster):
    A = len(before_recluster)
    B = len(after_recluster)
    Unio = [i for i in before_recluster if i in after_recluster]
    C = len(Unio)
    return C / (A + B - C)


# 数据子采样获取相应真实标签，用于评估簇数
def get_true_label(cellList, df):
    true_lab = []
    for cell in cellList:
        cell_df = df[df.Cell == cell]
        cell_label = cell_df['assigned_cluster'].values
        print(cell_label)
        true_lab.append(cell_label[0])
    return true_lab


# 计算聚类评估指标
def get_cluster_metrics(dataType, dataName, pred_labels):
    df = pd.read_csv('dataset/{}/{}_ground_truth.csv'.format(dataType, dataName))
    true_lab = df['cell_label'].values
    true_lab = np.array(true_lab)
    types = np.unique(true_lab)
    ids = np.arange(0, len(types))
    dict1 = dict(zip(ids, types))
    for id, type in dict1.items():
        for i in range(len(true_lab)):
            if true_lab[i] == type:
                true_lab[i] = id
    ARI = adjusted_rand_score(true_lab, pred_labels)
    NMI = normalized_mutual_info_score(true_lab, pred_labels)
    CA = cluster_accuracy(true_lab, pred_labels)
    print('{}--ARI--{:.4f},NMI--{:.4f},CA--{:.4f}'.format(dataName, ARI, NMI, CA))
    return round(ARI, 4), round(NMI, 4), round(CA, 4)


# 日志级别关系映射
level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}


# 日志输出
def _get_logger(filename, level='info'):
    # 创建日志对象
    log = logging.getLogger(filename)
    # 设置日志级别
    log.setLevel(level_relations.get(level))
    # 日志输出格式
    fmt = logging.Formatter('%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    # 输出到文件
    # 日志文件按天进行保存，每天一个日志文件
    file_handler = handlers.TimedRotatingFileHandler(filename=filename, when='D', backupCount=1, encoding='utf-8')
    file_handler.setFormatter(fmt)
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    return log


# 真实标签与预测标签统计指标，用于制作桑基图
def get_label_statistics_data(dataType, dataName, modelName=None):
    true_df = pd.read_csv('Dataset/{}/{}_ground_truth.csv'.format(dataType, dataName))
    true_list = true_df['cell_label'].values
    df = pd.read_csv('VGAAEResults/ClusteringLabels/PredictLabelResult/{}/{}_pred.csv'.format(modelName, dataName))
    pred_list = df['pred_label'].values
    get_cluster_metrics(dataType, dataName, pred_list)
    print(f'model--{modelName},number of cluster is {len(np.unique(pred_list))}')
    write_df = pd.DataFrame({'true_label': true_list, 'pred_label': pred_list})
    fileName = 'VGAAEResults/ClusteringLabels/TrueAndPredict/{}/{}_label_statistics_data_{}.csv'.format(dataType,
                                                                                                        dataName,
                                                                                                        modelName)
    if os.path.exists(fileName):
        read_df = pd.read_csv(fileName)
    else:
        write_df.to_csv(fileName, index=False)
        read_df = pd.read_csv(fileName)
    # 创建结点
    nodes = []
    for i in range(2):
        values = read_df.iloc[:, i].unique()
        for value in values:
            dict = {}
            if i == 0:
                dict['name'] = value
            if i == 1:
                dict['name'] = 'Cluster' + str(value)
            nodes.append(dict)
    # 创建边
    linkes = []
    true_uni = true_df['cell_label'].unique()
    for true_lab in true_uni:
        # 返回DataFrame
        query_df = read_df.query(f'true_label=="{true_lab}"')
        # 返回索引为唯一值，数据为唯一值对应的数量 DataFrame
        pred_count = query_df['pred_label'].value_counts().to_frame(name='count')
        for item in range(pred_count.shape[0]):
            linkes.append({'source': true_lab, 'target': 'Cluster' + str(pred_count.index.values[item]),
                           'value': int(pred_count.values[item][0])})
    return nodes, linkes


# 合并图片 axis=0为纵向拼接；axis=1为横向拼接
def merge_picture(pic_path_list, pic_width, pic_height, axis, fileName, picType):
    img_array = ''
    img = ''
    for i, v in enumerate(pic_path_list):
        if i == 0:
            img = Image.open(v)  # 打开图片
            # 此处将单张图像进行缩放为统一大小，改为自己单张图像的平均尺寸即可
            img = img.resize((pic_width, pic_height), Image.ANTIALIAS)
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img = Image.open(v)
            # 此处将单张图像进行缩放为统一大小，改为自己单张图像的平均尺寸即可
            img = img.resize((pic_width, pic_height), Image.ANTIALIAS)
            img_array2 = np.array(img)
            img_array = np.concatenate((img_array, img_array2), axis=axis)  # 纵向拼接
            img = Image.fromarray(img_array)
    # 保存图片
    img.save(f'results/VisualResult/{fileName}_{picType}.png')


# 得到真实簇与预测簇的对应关系
def get_relation(dataType, dataName, modelName):
    true_df = pd.read_csv('dataset/{}/{}_ground_truth.csv'.format(dataType, dataName))
    true_labels = true_df['assigned_cluster'].values
    pred_labels = pd.read_csv(f'results/{modelName}/{dataName}_pred.csv')
    uni_true_lab = np.unique(true_labels)
    _, linkes = get_label_statistics_data(dataType, dataName, modelName)
    dictList = []
    for cellType in uni_true_lab:
        dict = {}
        dict['source'] = cellType
        list = []
        for link in linkes:
            if link['source'] == cellType:
                target = int(link['target'].replace('Cluster', ''))
                list.append(target)
        dict['pred'] = list
        dictList.append(dict)


# 将预测标签保存至csv文件中
def save_cluster_result(datasetType, datasetName, y_pred, modelName):
    data = pd.read_csv(f"dataset/{datasetType}/{datasetName}.csv")
    cells = data.iloc[:, 0].tolist()
    pred_path = f'other/{modelName}/{datasetName}_pred.csv'
    result = []
    for i in range(len(y_pred)):
        result.append([cells[i], y_pred[i]])
    result = pd.DataFrame(np.array(result), columns=['cell', 'pred_label'])
    result.to_csv(pred_path, index=False, sep=',')
