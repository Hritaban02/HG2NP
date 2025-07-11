import torch


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)
        if torch.any(batch_n_nodes >= p.shape[0]).item():
            print("###########################################################################")
            print("Found a graph with larger number of nodes than what is seen in the dataset.")
            batch_n_nodes = torch.where(batch_n_nodes >= p.shape[0], p.shape[0]-1, batch_n_nodes)
            print("###########################################################################")
        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
