__HOME_DIR__ = "/home/du4/19CS30053/MTP2"

import sys
sys.path.append(f"{__HOME_DIR__}/Model/src")
from lib import *


def compute_loss_para(adj):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def vgae_loss(logits, adj, mean, log_std, device):
    weight_tensor, norm = compute_loss_para(adj)
    loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor.to(device))
    kl_divergence = (0.5 / logits.size(0) * (1 + 2 * log_std - mean**2 - torch.exp(log_std) ** 2).sum(1).mean())
    loss -= kl_divergence
    return loss