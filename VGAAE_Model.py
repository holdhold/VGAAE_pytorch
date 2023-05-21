
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GAE, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

EPS = 1e-15
MAX_LOGVAR = 10


# 编码器
class VGATEncoder(nn.Module):
    def __init__(self, in_channels, num_heads, hidden_dims, latent_dim, dropout, concat):
        super(VGATEncoder, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.hidden_layer1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dims[0],
            heads=self.num_heads['first'],
            dropout=dropout[0],
            # concat=True(concatenated) or False(averaged)
            concat=concat['first']
        )
        in_dim2 = hidden_dims[0] * self.num_heads['first'] if concat['first'] else hidden_dims[0]

        self.hidden_layer2 = GATConv(
            in_channels=in_dim2,
            out_channels=hidden_dims[1],
            heads=self.num_heads['second'],
            dropout=dropout[1],
            concat=concat['second']
        )
        in_dim_final = hidden_dims[-1] * self.num_heads['second'] if concat['second'] else hidden_dims[-1]

        self.mean = GATConv(
            in_channels=in_dim_final,
            out_channels=latent_dim,
            heads=self.num_heads['mean'],
            concat=False,
            dropout=0.2
        )
        self.log_std = GATConv(
            in_channels=in_dim_final,
            out_channels=latent_dim,
            heads=self.num_heads['std'],
            concat=False,
            dropout=0.2
        )

    def forward(self, x, edge_index):
        hidden_out1, atten_w_1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)
        hidden_out1 = F.relu(hidden_out1)
        hidden_out2, atten_w_2 = self.hidden_layer2(hidden_out1, edge_index, return_attention_weights=True)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        z_mean, atten_w_mean = self.mean(hidden_out2, edge_index, return_attention_weights=True)
        z_log_std, atten_w_log_std = self.log_std(hidden_out2, edge_index, return_attention_weights=True)
        return z_mean, z_log_std, [atten_w_1, atten_w_2, atten_w_mean, atten_w_log_std]


class VGATDecoder(GAE):
    def __init__(self, encoder, decoder=None, decoder_nn_dim1=None):
        super(VGATDecoder, self).__init__(encoder, decoder)
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.decoder_nn_dim1 = decoder_nn_dim1
        self.decoder_nn_dim2 = self.encoder.in_channels
        if decoder_nn_dim1:
            self.decoder_nn = Sequential(
                Linear(in_features=self.encoder.latent_dim, out_features=self.decoder_nn_dim1),
                BatchNorm1d(self.decoder_nn_dim1),
                ReLU(),
                Dropout(0.4),
                Linear(in_features=self.decoder_nn_dim1, out_features=self.decoder_nn_dim2)
            )

    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logvar__, attn_w = self.encoder(*args, **kwargs)
        self.__logvar__ = self.__logvar__.clamp(max=MAX_LOGVAR)
        z = self.reparametrize(self.__mu__, self.__logvar__)
        return z

    def kl_loss(self, mu=None, logvar=None):
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss


