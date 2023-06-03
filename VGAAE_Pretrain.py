from VGAAE_Model import VGATEncoder, VGATDecoder
from CreateGraph import *
from DataProcessing import CMF
import torch
import torch.nn.functional as F
import torch_geometric.transforms as Trans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse
import pandas as pd


def pretrain(model, optimizer, train_data, val_data, true_label, device):
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    res_ari = 0.0000
    for epoch in range(args['max_epoch']):
        model.train()
        z = model.encode(x, edge_index)
        reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        L_vgaa = args['re_loss'] * reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()
        recon_adjency = model.decoder_nn(z)
        decoder_loss = 0.0
        decoder_loss = F.mse_loss(recon_adjency, x)
        loss = args['vgaa_loss']*L_vgaa + decoder_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            'Epoch {:03d} -- Decoder epoch loss: {:.4f} -- VGAA epoch loss: {:.4f} -- Total epoch loss: {:.4f} '.format(
                epoch, decoder_loss, L_vgaa, loss
            ))
        with torch.no_grad():
            z = model.encode(x, edge_index)
            kmeans = KMeans(n_clusters=args['num_clusters'], n_init=20).fit(z.detach().numpy())
            ari = adjusted_rand_score(true_label, kmeans.labels_)
            nmi = normalized_mutual_info_score(true_label, kmeans.labels_)
            print(f"epoch {epoch}:nmi {nmi:.4f}, ari {ari:.4f}")
            if res_ari <= ari:
                res_ari = ari
                torch.save(
                    model.state_dict(),
                f"./pretrain/{args['datasetName']}/{args['datasetName']}.pkl"
                )
        auroc, ap = pretest(model, val_data, device)
        print('Validation AUROC {:.4f} -- AP {:.4f}.'.format(auroc, ap))


@torch.no_grad()
def pretest(model, data, device):
    model = model.eval()
    z = model.encode(data.x.to(torch.float).to(device), data.edge_index.to(torch.long).to(device))
    auroc, ap = model.test(z, data.pos_edge_label_index.to(torch.long).to(device),
                           data.neg_edge_label_index.to(torch.long).to(device))
    return auroc, ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VGAAE pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datasetType", type=str, default="Baron_Human")
    parser.add_argument("--datasetName", type=str, default="Baron_Human2")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_clusters", default=7, type=int)
    parser.add_argument("--num_heads", default=[3, 3, 3, 3], type=int)
    parser.add_argument('--k', type=int, default=10, help='K of neighbors Faiss KNN')
    parser.add_argument('--decoder_nn_dim1', type=int, default=128,
                        help='First hidden dimension for the neural network decoder')
    parser.add_argument('--dropout', type=float, default=[0.2, 0.2], help='Dropout for each layer')
    parser.add_argument('--hidden_dims', type=int, default=[128, 128], help='Output dimension for each hidden layer.')
    parser.add_argument('--latent_dim', type=int, default=50, help='output dimension for node embeddings')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--re_loss', type=float, default=0.1)
    parser.add_argument('--vgaa_loss', type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(args)
    df = pd.read_csv('dataset/{}/{}_ground_truth.csv'.format(args['datasetType'], args['datasetName']))
    true_lab = df['assigned_cluster'].values
    adata = read_data('dataset/{}/{}.csv'.format(args['datasetType'], args['datasetName']), file_type='csv')
    adata = preprocess_raw_data(adata)
    X = adata.X
    adata_hvg, X_hvg = prepare_training_data(adata)
    X_impute = CMF(X_hvg, 1, 1, 0.0001, 0.0001)
    np.save('process/{}.npy'.format(args['datasetName']),X_impute)
    # X_impute = np.load('process/{}.npy'.format(args['datasetName']))
    distances, neighbors, cutoff, edgelist = get_edgelist(datasetName=args['datasetName'], X_hvg=X_impute, k=args['k'],
                                                          type='Faiss_KNN')
    edges = load_separate_graph_edgelist('process/{}_edgelist.txt'.format(args['datasetName']))
    data_obj = create_graph(edges, X_impute)
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
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
    num_clusters = args['num_clusters']

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
             device=device)
