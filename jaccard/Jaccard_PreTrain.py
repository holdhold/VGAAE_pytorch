from VGAAE_Model import VGATEncoder, VGATDecoder
from CreateGraph import *
from DataProcessing import CMF
import torch
import torch.nn.functional as F
import torch_geometric.transforms as Trans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse
from Utils import get_trueLab
import h5py
import pandas as pd
from tqdm import tqdm
import os

os.environ["OMP_NUM_THREADS"] = '1'
import warnings

warnings.filterwarnings('ignore')


def pretrain(model, optimizer, train_data, val_data, true_label, device, num_subsample, cur_cluster):
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    res_ari = 0.0000
    for epoch in range(args['max_epoch']):
        model.train()
        z = model.encode(x, edge_index)
        reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss = reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()
        recon_adjency = model.decoder_nn(z)
        decoder_loss = 0.0
        decoder_loss = F.mse_loss(recon_adjency, x) * 10
        loss += decoder_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z = model.encode(x, edge_index)
            kmeans = KMeans(n_clusters=cur_cluster, n_init=20).fit(z.detach().numpy())
            ari = adjusted_rand_score(true_label, kmeans.labels_)
            nmi = normalized_mutual_info_score(true_label, kmeans.labels_)
            if res_ari <= ari:
                res_ari = ari
                torch.save(
                    model.state_dict(),
                    f"./pretrain/{args['datasetName']}/{args['datasetName']}_subsample{num_subsample}_cluster{cur_cluster}.pkl"
                )

        auroc, ap = pretest(model, val_data, device)


@torch.no_grad()
def pretest(model, data, device):
    model = model.eval()
    z = model.encode(data.x.to(torch.float).to(device), data.edge_index.to(torch.long).to(device))
    auroc, ap = model.test(z, data.pos_edge_label_index.to(torch.long).to(device),
                           data.neg_edge_label_index.to(torch.long).to(device))
    return auroc, ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VGAAC pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datasetType", type=str, default="Baron_Mouse")
    parser.add_argument("--datasetName", type=str, default="Baron_Mouse1")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_clusters", default=13, type=int)
    parser.add_argument("--num_heads", default=[3, 3, 3, 3], type=int)
    parser.add_argument('--k', type=int, default=30, help='K of neighbors Faiss KNN')
    parser.add_argument('--decoder_nn_dim1', type=int, default=128,
                        help='First hidden dimension for the neural network decoder')
    parser.add_argument('--dropout', type=float, default=[0.2, 0.2], help='Dropout for each layer')
    parser.add_argument('--hidden_dims', type=int, default=[128, 128], help='Output dimension for each hidden layer.')
    parser.add_argument('--latent_dim', type=int, default=50, help='output dimension for node embeddings')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--num_subsample', type=float, default=20, help='Number of subsample')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(args)

    # csv文件读取
    df = pd.read_csv('dataset/{}/{}_ground_truth.csv'.format(args['datasetType'], args['datasetName']))
    true_lab = df['assigned_cluster'].values
    adata = read_data('dataset/{}/{}.csv'.format(args['datasetType'], args['datasetName']), file_type='csv')

    # 簇数为(num_cluster-8,num_cluster+8)之间，每个簇设置进行20次随机子采样,左闭右开
    for cur_cluster in range(args['num_clusters'] - 8, args['num_clusters'] + 1):
        print(f'current number of cluster is {cur_cluster},start subsample....')
        sample_bar = tqdm(range(args['num_subsample']), desc='start subample:')
        for i in sample_bar:
            sub_adata = subsample_anndata(adata)
            cellList = sub_adata.obs_names.tolist()
            true_lab = get_trueLab(cellList, df)
            np.save(
                'process/{}/true_lab_subsample{}_cluster{}.npy'.format(args['datasetName'], (i + 1), cur_cluster),
                true_lab)
            sub_adata = preprocess_raw_data(sub_adata)
            print(sub_adata)
            X = sub_adata.X
            sub_adata_hvg, sub_X_hvg = prepare_training_data(sub_adata)
            print(sub_adata_hvg)
            sub_X_impute = CMF(sub_X_hvg, 1, 1, 0.0001, 0.0001)
            np.save('process/{}/subsample{}_cluster{}.npy'.format(args['datasetName'], (i + 1), cur_cluster),
                    sub_X_impute)
            distances, neighbors, cutoff, edgelist = get_edgelist(datasetName=args['datasetName'], X_hvg=sub_X_impute,
                                                                  k=args['k'],
                                                                  type='Faiss_KNN', num_subsample=(i + 1),
                                                                  cur_cluster=cur_cluster)
            edges = load_separate_graph_edgelist(
                'process/{}/subsample{}_cluster{}_edgelist.txt'.format(args['datasetName'], (i + 1), cur_cluster))
            data_obj = create_graph(edges, sub_X_impute)
            data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
            # --------------------------------划分数据集------------------------------------#
            test_split = args['test_split']
            val_split = args['val_split']
            try:
                transform = Trans.RandomLinkSplit(num_val=val_split, num_test=test_split,
                                                  is_undirected=True, add_negative_train_samples=True,
                                                  split_labels=True)
                train_data, val_data, test_data = transform(data_obj)
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
                     val_data=val_data,
                     true_label=true_lab,
                     device=device,
                     num_subsample=i + 1,
                     cur_cluster=num_clusters
                     )
        sample_bar.close()
