import argparse
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from VGAAE_Model import VGATEncoder, VGATDecoder
from CreateGraph import *
from Utils import *
import torch_geometric.transforms as Trans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DEC(nn.Module):
    def __init__(self, model, latent_dims, alpha=1, num_subsample=0, cur_cluster=0):
        super(DEC, self).__init__()
        self.num_subsample = num_subsample
        self.cur_cluster=cur_cluster
        self.alpha = alpha
        self.model = model
        self.model.load_state_dict(
            torch.load(
                './pretrain/{}/{}_subsample{}_cluster{}.pkl'.format(args['datasetName'], args['datasetName'],
                                                                    num_subsample, cur_cluster),
                map_location='cpu'))

        self.cluster_layer = Parameter(torch.Tensor(num_clusters, latent_dims))
        torch.nn.init.xavier_normal(self.cluster_layer.data)

    def forward(self, x, edge_index):
        z = self.model.encode(x, edge_index)
        q = self.get_Q(z)
        return z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train(dec, optimizer, train_data, val_data, device, true_label, num_subsample, num_reclustering, cur_cluster):
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    with torch.no_grad():
        z = dec.model.encode(x, edge_index)

    kmeans = KMeans(n_clusters=cur_cluster, n_init=20)
    y_pred = kmeans.fit_predict(z.data.detach().numpy())
    y_pred_last = np.copy(y_pred)
    ari = adjusted_rand_score(true_label, y_pred)
    nmi = normalized_mutual_info_score(true_label, y_pred)
    dec.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    res_ari = 0.0000
    res_nmi = 0.0000
    for epoch in range(args['max_epoch']):
        dec.train()
        if epoch % args['update_interval'] == 0:
            z, Q = dec(x, edge_index)
            q = Q.detach().data.cpu().numpy().argmax(1)
            y_pred = np.copy(q)
            ari = adjusted_rand_score(true_label, q)
            nmi = normalized_mutual_info_score(true_label, q)
            if res_ari <= ari:
                res_ari = ari
                res_nmi = nmi
                np.save(
                    f"results/{args['datasetName']}/subsample{num_subsample}_cluster{cur_cluster}_recluster{num_reclustering}.npy",
                    q)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and delta_label < 1e-3:
                print('delta_label=={}'.format(delta_label))
                print('Reach tolerance threshold,Stopping training.')
                break

        z, q = dec(x, edge_index)
        p = target_distribution(Q.detach())

        clu_loss = wasserstein_distance(p, q)

        reconstruction_loss = dec.model.recon_loss(z, train_data.pos_edge_label_index)
        loss = args['clu_loss'] * clu_loss + args['re_loss'] * reconstruction_loss + (
                1 / train_data.num_nodes) * dec.model.kl_loss()
        recon_adjency = dec.model.decoder_nn(z)
        decoder_loss = 0.0
        decoder_loss = args['de_loss'] * F.mse_loss(recon_adjency, x)
        loss += decoder_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:03d} -- Total epoch loss: {:.4f} -- NN decoder epoch loss: {:.4f}'.format(epoch, loss,
                                                                                                 decoder_loss))
    print('final result---ARI={:.4f},NMI={:.4f}'.format(res_ari, res_nmi))


@torch.no_grad()
def test(dec, val_data, device):
    dec = dec.eval()
    z = dec.model.encode(val_data.x.to(torch.float).to(device), val_data.edge_index.to(torch.long).to(device))
    auroc, ap = dec.model.test(z, val_data.pos_edge_label_index.to(torch.long).to(device),
                               val_data.neg_edge_label_index.to(torch.long).to(device))
    return auroc, ap


if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='train', description='VGAAC train')
    parse.add_argument("--datasetType", type=str, default="Baron_Mouse")
    parse.add_argument("--datasetName", type=str, default="Baron_Mouse2")
    parse.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers')
    parse.add_argument('--hidden_dims', type=int, default=[128, 128],
                       help='Output dimension for each hidden layer.')
    parse.add_argument('--latent_dim', type=int, default=50, help='output dimension for node embeddings')
    parse.add_argument('--dropout', type=float, default=[0.2, 0.2], help='Dropout for each layer')
    # 可调
    parse.add_argument('--num_heads', type=int, default=[3, 3, 3, 3],
                       help='Number of attention heads for each layer')
    parse.add_argument('--decoder_nn_dim1', type=int, default=128,
                       help='First hidden dimension for the neural network decoder')
    parse.add_argument('--lr', type=int, default=1e-3, help='Learning rate of Adma')
    parse.add_argument('--max_epoch', type=int, default=200, help='Number of training epoch ')
    parse.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parse.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parse.add_argument('--clu_loss', type=float, default=10)
    parse.add_argument('--re_loss', type=float, default=0.1)
    parse.add_argument('--de_loss', type=float, default=10)
    parse.add_argument('--update_interval', default=1, type=int)
    parse.add_argument('--num_subsample', type=float, default=20, help='Number of subsample')
    # 可调
    parse.add_argument('--num_clusters', type=int, default=13, help='Number of clusters')
    args = parse.parse_args()

    # --------------------------------构图------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = vars(args)
    for cur_cluster in tqdm(range(args['num_clusters']-8,args['num_clusters']+1)):
        print(f'current number of cluster is {cur_cluster},start training...')
        for i in range(args['num_subsample']):
            X_impute = np.load(
                'process/{}/subsample{}_cluster{}.npy'.format(args['datasetName'], (i + 1), cur_cluster))
            # 读取边列表文件，获得构图的每一条边
            edges = load_separate_graph_edgelist(
                'process/{}/subsample{}_cluster{}_edgelist.txt'.format(args['datasetName'], (i + 1), cur_cluster))
            true_lab = np.load(
                'process/{}/true_lab_subsample{}_cluster{}.npy'.format(args['datasetName'], (i + 1), cur_cluster))
            # 构建邻接图
            data_obj = create_graph(edges, X_impute)
            # print(data_obj)
            data_obj.num_nodes = X_impute.shape[0]
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
            for j in range(2):
                encoder = VGATEncoder(
                    in_channels=num_features,
                    num_heads=num_heads,
                    hidden_dims=hidden_dims,
                    latent_dim=latent_dim,
                    dropout=dropout,
                    concat={'first': True, 'second': False},
                )
                model = VGATDecoder(encoder=encoder,
                                    decoder_nn_dim1=args['decoder_nn_dim1'])

                dec = DEC(
                          model=model,
                          latent_dims=latent_dim,
                          num_subsample=(i + 1),
                          cur_cluster=num_clusters)

                optimizer = torch.optim.Adam(dec.parameters(), lr=args['lr'])

                train(dec=dec,
                      optimizer=optimizer,
                      train_data=train_data,
                      val_data=val_data,
                      device=device,
                      true_label=true_lab,
                      num_subsample=(i + 1),
                      num_reclustering=(j + 1),
                      cur_cluster=num_clusters)
