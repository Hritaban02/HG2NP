__HOME_DIR__ = "/home/du4/19CS30053/MTP2"

import sys
sys.path.append(f"{__HOME_DIR__}/Model/src")
from lib import *


class VGAEModel(nn.Module):
    def __init__(self, hidden1_dim, hidden2_dim, device):
        super(VGAEModel, self).__init__()
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.device = device

        self.get_latent_vector = tgnn.GCNConv(
            in_channels     = -1,
            out_channels    = self.hidden1_dim,
            add_self_loops  = True,
        )
        
        self.get_mu = tgnn.GCNConv(
            in_channels     = -1,
            out_channels    = self.hidden2_dim,
            add_self_loops  = True,
        )
        
        self.get_log_std = tgnn.GCNConv(
            in_channels     = -1,
            out_channels    = self.hidden2_dim,
            add_self_loops  = True,
        )

    def encoder(self, x, edge_index):
        h = self.get_latent_vector(x, edge_index)
        h = F.relu(h)

        self.mean = self.get_mu(h, edge_index)
        self.log_std = self.get_log_std(h, edge_index)
        
        gaussian_noise = torch.randn(x.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)
        return sampled_z # num_nodes x hidden2_dim

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t())) # num_nodes x num_nodes
        return adj_rec

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_rec = self.decoder(z)
        return adj_rec