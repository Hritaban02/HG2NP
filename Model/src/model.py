from lib import *
from clean_up import *
from utils import *


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.HetMPNN = HeterogeneousMessagePassingNeuralNetwork(cfg["mpnn"])
        self.mean_aggr = aggr.MeanAggregation()

        self.Linear_Layers = nn.Sequential()
        
        i = 0
        n = len(cfg["linear"]["hidden_layers"])
        self.Linear_Layers.add_module(f"Linear_{i}", nn.Linear(in_features=cfg["linear"]["latent_dim"]*len(cfg["mpnn"]["metadata"][0]), out_features=cfg["linear"]["hidden_layers"][0]))
        i+=1

        while(i<n):
            torch.nn.init.kaiming_normal(self.Linear_Layers[-1].weight, a=cfg["linear"]["negative_slope"], mode='fan_in', nonlinearity='leaky_relu') 
            self.Linear_Layers.add_module(f"LeakyReLU_{i-1}", nn.LeakyReLU(negative_slope=cfg["linear"]["negative_slope"]))
            self.Linear_Layers.add_module(f"Linear_{i}", nn.Linear(in_features=cfg["linear"]["hidden_layers"][i-1], out_features=cfg["linear"]["hidden_layers"][i]))
            i+=1

        torch.nn.init.kaiming_normal(self.Linear_Layers[-1].weight, a=cfg["linear"]["negative_slope"], mode='fan_in', nonlinearity='leaky_relu') 
        self.Linear_Layers.add_module(f"LeakyReLU_{i-1}", nn.LeakyReLU(negative_slope=cfg["linear"]["negative_slope"]))
        self.Linear_Layers.add_module(f"Linear_{i}", nn.Linear(in_features=cfg["linear"]["hidden_layers"][i-1], out_features=1))

        torch.nn.init.xavier_normal(self.Linear_Layers[-1].weight) 
        self.Linear_Layers.add_module(f"Sigmoid", nn.Sigmoid())

    def forward(self, x_dict, edge_index_dict, batch_dict):
        with ClearCache():
            x_dict = self.HetMPNN(x_dict, edge_index_dict)
            graph_repr = torch.cat([self.mean_aggr(x_dict[key], batch_dict[key]) for key in x_dict], dim=1)
            return self.Linear_Layers(graph_repr)
        
class HeterogeneousMessagePassingNeuralNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.HGTConv_Layers = nn.ModuleList()

        i = 0
        n = len(cfg["hidden_layers"])
        self.HGTConv_Layers.append(tgnn.HGTConv(in_channels=-1, 
                                                out_channels=cfg["hidden_layers"][0], 
                                                heads=cfg["heads"], 
                                                metadata=cfg["metadata"])) 
        i+=1

        while(i<n):
            self.HGTConv_Layers.append(tgnn.HGTConv(in_channels=cfg["hidden_layers"][i-1], 
                                                    out_channels=cfg["hidden_layers"][i], 
                                                    heads=cfg["heads"], 
                                                    metadata=cfg["metadata"]))
            i+=1

        self.HGTConv_Layers.append(tgnn.HGTConv(in_channels=cfg["hidden_layers"][i-1], 
                                                out_channels=cfg["latent_dim"], 
                                                heads=cfg["heads"], 
                                                metadata=cfg["metadata"]))
    
        self.LeakyReLU_Layer = nn.LeakyReLU(negative_slope=cfg["negative_slope"])

    def forward(self, x_dict, edge_index_dict):
        with ClearCache():
            for layer in self.HGTConv_Layers:
                with ClearCache():
                    x_dict = layer(x_dict, edge_index_dict)
                    for key in x_dict:
                        x_dict[key] = self.LeakyReLU_Layer(x_dict[key])
            return x_dict

class HomogeneousMessagePassingNeuralNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.GATv2Conv_Layers = nn.ModuleList()

        i = 0
        n = len(cfg["hidden_layers"])
        self.GATv2Conv_Layers.append(tgnn.GATv2Conv(in_channels=-1, 
                                                    out_channels=cfg["hidden_layers"][0], 
                                                    heads=cfg["heads"], 
                                                    concat=False,
                                                    edge_dim=cfg["edge_types"])) 
        i+=1

        while(i<n):
            self.GATv2Conv_Layers.append(tgnn.GATv2Conv(in_channels=cfg["hidden_layers"][i-1], 
                                                        out_channels=cfg["hidden_layers"][i], 
                                                        heads=cfg["heads"], 
                                                        concat=False,
                                                        edge_dim=cfg["edge_types"]))
            i+=1

        self.GATv2Conv_Layers.append(tgnn.GATv2Conv(in_channels=cfg["hidden_layers"][i-1], 
                                                    out_channels=cfg["latent_dim"], 
                                                    heads=cfg["heads"], 
                                                    concat=False,
                                                    edge_dim=cfg["edge_types"]))
    
        self.LeakyReLU_Layer = nn.LeakyReLU(negative_slope=cfg["negative_slope"])

    def forward(self, x, edge_index, edge_attr=None):
        with ClearCache():
            if edge_attr is not None:
                for layer in self.GATv2Conv_Layers:
                    with ClearCache():
                        x = layer(x, edge_index, edge_attr)
                        x = self.LeakyReLU_Layer(x)
            else:
                for layer in self.GATv2Conv_Layers:
                    with ClearCache():
                        x = layer(x, edge_index)
                        x = self.LeakyReLU_Layer(x)
            return x

class Sampler(nn.Module):
    def __init__(self, cfg, types):
        super().__init__()
        self.Linear_Layers = nn.Sequential()
        
        i = 0
        n = len(cfg["hidden_layers"])
        self.Linear_Layers.add_module(f"Linear_{i}", nn.Linear(in_features=cfg["latent_dim"], out_features=cfg["hidden_layers"][0]))
        i+=1

        while(i<n):
            torch.nn.init.kaiming_normal(self.Linear_Layers[-1].weight, a=cfg["negative_slope"], mode='fan_in', nonlinearity='leaky_relu') 
            self.Linear_Layers.add_module(f"LeakyReLU_{i-1}", nn.LeakyReLU(negative_slope=cfg["negative_slope"]))
            self.Linear_Layers.add_module(f"Linear_{i}", nn.Linear(in_features=cfg["hidden_layers"][i-1], out_features=cfg["hidden_layers"][i]))
            i+=1

        torch.nn.init.kaiming_normal(self.Linear_Layers[-1].weight, a=cfg["negative_slope"], mode='fan_in', nonlinearity='leaky_relu') 
    
        self.Linear_Layers.add_module(f"LeakyReLU_{i-1}", nn.LeakyReLU(negative_slope=cfg["negative_slope"]))
        self.Linear_Layers.add_module(f"Linear_{i}", nn.Linear(in_features=cfg["hidden_layers"][i-1], out_features=types))
        torch.nn.init.xavier_normal(self.Linear_Layers[-1].weight) 
        
    def forward(self, x):
        with ClearCache():
            return self.Linear_Layers(x)
        
class HETModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = cfg["device"]

        self.dataset_name = cfg["dataset_name"]

        if self.dataset_name=="DBLP":
            self.category_map = DBLP_index_to_category
            if "conference"==cfg["split_criteria"] or "conference" in cfg["split_criteria"]:
                self.category_map = {k:v for k,v in self.category_map.items() if v != "conference"}
        if self.dataset_name=="IMDB":
            self.category_map = IMDB_index_to_category

        self.HomMPNN = HomogeneousMessagePassingNeuralNetwork(cfg["mpnn"])
        self.samplers = nn.ModuleDict()

        self.sample_pools = {}
        for key in self.category_map:
            self.sample_pools[key]=get_sample_pool(dataset=self.dataset_name, category=self.category_map[key]).to(cfg["device"])
            self.samplers.update({self.category_map[key]:Sampler(cfg=cfg["sampler"], types=len(self.sample_pools[key]))})
    
    def forward(self, batch):
        with ClearCache():
            hetero_graph_list = []
            for data in batch.to_data_list():
                graph_x = self.HomMPNN(data.x, data.edge_index)

                node_type = torch.argmax(data.x, dim=1)

                indexToPosition = {}
                positionToIndex = {}

                d = HeteroData()

                for key in self.category_map:
                    indexToPosition[key] = torch.nonzero(node_type==key)
                    positionToIndex[key] = {x.item(): i for i, x in enumerate(indexToPosition[key])}
                    indexToPosition[key] = indexToPosition[key].reshape((-1))

                    logits = self.samplers[self.category_map[key]](graph_x[indexToPosition[key]])
                    one_hot = F.gumbel_softmax(logits=logits, tau=1.0, hard=True)
                
                    d[self.category_map[key]]['x']=torch.matmul(one_hot, self.sample_pools[key])
                    if d[self.category_map[key]]['x'].shape[0]==0:
                        d[self.category_map[key]]['x']=torch.zeros(1, d[self.category_map[key]]['x'].shape[1]).to(self.device)

                for key1 in self.category_map:
                    for key2 in self.category_map:
                        d[self.category_map[key1], 'to', self.category_map[key2]].edge_index = []

                for edge in data.edge_index.t():
                    i = edge[0].item()
                    j = edge[1].item()

                    if (node_type[i].item() in self.category_map and node_type[j].item() in self.category_map):
                        d[self.category_map[node_type[i].item()], 'to', self.category_map[node_type[j].item()]].edge_index.append(torch.tensor([positionToIndex[node_type[i].item()][i], positionToIndex[node_type[j].item()][j]], dtype=torch.int))

                for key1 in self.category_map:
                    for key2 in self.category_map:
                        if len(d[self.category_map[key1], 'to', self.category_map[key2]].edge_index)!=0:
                            d[self.category_map[key1], 'to', self.category_map[key2]].edge_index = torch.stack(d[self.category_map[key1], 'to', self.category_map[key2]].edge_index, dim=0).t()
                        else:
                            del d[self.category_map[key1], 'to', self.category_map[key2]]

                hetero_graph_list.append(d)
            return Batch.from_data_list(hetero_graph_list)