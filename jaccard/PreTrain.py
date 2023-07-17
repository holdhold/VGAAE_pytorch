from VGAAE_Model import VGATEncoder, VGATDecoder
from CreateGraph import *
from DataProcessing import CMF
import torch
import torch.nn.functional as F
import torch_geometric.transforms as Trans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse
from Utils import get_true_label
import h5py
import pandas as pd
from tqdm import tqdm
import os

os.environ["OMP_NUM_THREADS"] = '1'
import warnings

warnings.filterwarnings('ignore')


def pretrain(model, optimizer, train_data, true_label, device, num_subsample, cur_cluster):
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    res_ari = 0.0000
    for epoch in range(args['max_epoch']):
        model.train()
        z = model.encode(x, edge_index)
        # 重构邻接图损失
        reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        # 总损失
        L_vgaa = args['re_loss'] * reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()
        # 重构样本
        recon_adjency = model.decoder_nn(z)
        decoder_loss = 0.0
        decoder_loss = F.mse_loss(recon_adjency, x) * 10
        loss = args['vgaa_loss'] * L_vgaa + decoder_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z = model.encode(x, edge_index)
            kmeans = KMeans(n_clusters=cur_cluster, n_init=20).fit(z.detach().numpy())
            ari = adjusted_rand_score(true_label, kmeans.labels_)
            nmi = normalized_mutual_info_score(true_label, kmeans.labels_)
            # print(f"epoch {epoch}:nmi {nmi:.4f}, ari {ari:.4f}")
            # 保存预训练最好的模型参数
            if res_ari <= ari:
                res_ari = ari
                torch.save(
                    model.state_dict(),
                    f"./tmpFile/{args['datasetName']}/{args['datasetName']}_subsample{num_subsample}_cluster{cur_cluster}.pkl"
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VGAAC pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datasetType", type=str, default="Chen")
    parser.add_argument("--datasetName", type=str, default="Chen")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_clusters", default=47, type=int)
    parser.add_argument("--num_heads", default=[3, 3, 3, 3], type=int)
    parser.add_argument('--k', type=int, default=10, help='K of neighbors Faiss KNN')
    parser.add_argument('--decoder_nn_dim1', type=int, default=128,
                        help='First hidden dimension for the neural network decoder')
    parser.add_argument('--dropout', type=float, default=[0.2, 0.2], help='Dropout for each layer')
    parser.add_argument('--hidden_dims', type=int, default=[128, 128], help='Output dimension for each hidden layer.')
    parser.add_argument('--latent_dim', type=int, default=50, help='output dimension for node embeddings')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--num_subsample', type=float, default=20, help='Number of subsample')
    parser.add_argument('--re_loss', type=float, default=0.1)
    parser.add_argument('--vgaa_loss', type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(args)

    # csv文件读取
    df = pd.read_csv('../Dataset/{}/{}_ground_truth.csv'.format(args['datasetType'], args['datasetName']))
    true_lab = df['cell_label'].values
    adata = read_data('../Dataset/{}/{}.csv'.format(args['datasetType'], args['datasetName']), file_type='csv')

    for cur_cluster in range(args['num_clusters'] - 37, args['num_clusters'] -6,5):
        print(f'current number of cluster is {cur_cluster},start subsample....')
        sample_bar = tqdm(range(args['num_subsample']), desc='start subample:')
        for i in sample_bar:
            # 子采样500个细胞
            sub_adata = subsample_anndata(adata)
            cellList = sub_adata.obs_names.tolist()
            true_lab = get_true_label(cellList, df)
            np.save(
                './tmpFile/{}/true_lab_subsample{}_cluster{}.npy'.format(args['datasetName'], (i + 1), cur_cluster),
                true_lab)
            # 预处理数据
            sub_adata = preprocess_raw_data(sub_adata)
            print(sub_adata)
            X = sub_adata.X
            # 准备训练数据，当基因数远大于1000时选取前1000个高变量基因
            sub_adata_hvg, sub_X_hvg = prepare_training_data(sub_adata)
            print(sub_adata_hvg)
            # 生成插补矩阵
            sub_X_impute = CMF(sub_X_hvg, 1, 1, 0.0001, 0.0001)
            np.save('./tmpFile/{}/X_impute_subsample{}_cluster{}.npy'.format(args['datasetName'], (i + 1), cur_cluster),
                    sub_X_impute)
            # X_impute = np.load('process/{}.npy'.format(args['datasetName']))
            # 计算距离，生成边列表文件 第一列为结点索引，第二列结点对应的边权值
            distances, neighbors, cutoff, edgelist = get_sub_edgelist(datasetName=args['datasetName'], X_hvg=sub_X_impute,
                                                                  k=args['k'],
                                                                  type='Faiss_KNN', num_subsample=(i + 1),
                                                                  cur_cluster=cur_cluster)
            # 读取边列表文件，获得构图的每一条边
            edges = load_separate_graph_edgelist(
                './tmpFile/{}/edgelist_subsample{}_cluster{}.txt'.format(args['datasetName'], (i + 1), cur_cluster))
            # 构建邻接图
            data_obj = create_graph(edges, sub_X_impute)
            # print(data_obj)
            data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
            # --------------------------------划分数据集------------------------------------#
            test_split = args['test_split']
            val_split = args['val_split']
            try:
                transform = Trans.RandomLinkSplit(num_val=val_split, num_test=test_split,
                                                  is_undirected=True, add_negative_train_samples=True,
                                                  split_labels=True)
                # 训练集、验证集和测试集
                train_data, val_data, test_data = transform(data_obj)
                # print(train_data)
            except IndexError as ie:
                print()
                print('Might need to transpose input with the --transpose_input argument.')

            num_features = data_obj.num_features
            heads = args['num_heads']
            num_heads = {}
            num_heads['first'] = heads[0]
            num_heads['second'] = heads[1]
            num_heads['mean'] = heads[2]
            num_heads['std'] = heads[3]
            hidden_dims = args['hidden_dims']
            latent_dim = args['latent_dim']
            dropout = args['dropout']
            num_clusters = cur_cluster

            encoder = VGATEncoder(
                in_channels=num_features,
                num_heads=num_heads,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                dropout=dropout,
                concat={'first': True, 'second': False},
            )
            model = VGATDecoder(encoder=encoder, decoder_nn_dim1=args['decoder_nn_dim1'])
            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
            pretrain(model=model,
                     optimizer=optimizer,
                     train_data=train_data,
                     true_label=true_lab,
                     device=device,
                     num_subsample=i + 1,
                     cur_cluster=num_clusters
                     )
        sample_bar.close()
