import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

import re
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

__HOME_DIR__ = "/home/du4/19CS30053/MTP2"
__MAX_NODES__ = 200
__TRAIN_SPLIT__ = 0.8
__TEST_SPLIT__ = 0.2

import sys
sys.path.append(f"{__HOME_DIR__}/Model/src")
from dataset import Heterogeneous_Graph_Dataset

class CategoricalGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None, single_graph=False):
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}
        self.dataset_name = dataset_name
        self.split = split
        self.single_graph = single_graph
        
        if re.match('(?:% s)' % '|'.join(['DBLP_*', 'IMDB_*']), self.dataset_name):
            self.name = self.dataset_name.split("_")
            self.split_criteria=self.name[1]
            if self.name[1]=="":
                self.split_criteria=self.name[2:]
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx[self.split]])

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        self.dataset = Heterogeneous_Graph_Dataset(dataset_name=self.name[0], split_criteria=self.split_criteria)
        self.dataset.get_categorical_graph(on_split_data=True)
        self.num_graphs = len(self.dataset.split_data_category_graph)

        if self.single_graph:
            index_of_graph_with_max_nodes = -1
            num_max_nodes = 0
            for i, g in enumerate(self.dataset.split_data_category_graph):
                if (index_of_graph_with_max_nodes==-1 or g['x'].shape[0]>num_max_nodes) and g['x'].shape[0] <= __MAX_NODES__ :
                    index_of_graph_with_max_nodes=i
                    num_max_nodes = g['x'].shape[0]

            print(index_of_graph_with_max_nodes)

            g_cpu = torch.Generator()
            g_cpu.manual_seed(0)

            test_len = int(round(self.num_graphs * __TEST_SPLIT__))
            train_len = int(round((self.num_graphs - test_len) * __TRAIN_SPLIT__))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]
            test_indices = indices[train_len + val_len:]

            train_data = []
            val_data = []
            test_data = []

            if self.dataset.split_data_category_graph is not None:
                for i, g in enumerate(self.dataset.split_data_category_graph):
                    if g['x'].shape[0] <= __MAX_NODES__:
                        g['edge_attr'] = torch.zeros(g['edge_index'].shape[-1], 2, dtype=torch.float)
                        g['edge_attr'][:, 1] = 1
                        g['num_nodes'] = g['x'].shape[0]
                        g['y'] = torch.zeros([1, 0]).float()

                        if self.pre_filter is not None and not self.pre_filter(g):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(g)

                        if i==index_of_graph_with_max_nodes:
                            train_data.append(g)
                            
                        if i in train_indices or i in val_indices:
                            val_data.append(g)
                        elif i in test_indices:
                            test_data.append(g)
                        else:
                            raise ValueError(f'Index {i} not in any split')

            torch.save(self.collate(train_data), self.processed_paths[self.file_idx['train']])
            torch.save(self.collate(val_data), self.processed_paths[self.file_idx['val']])
            torch.save(self.collate(test_data), self.processed_paths[self.file_idx['test']])
        else:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(0)

            test_len = int(round(self.num_graphs * __TEST_SPLIT__))
            train_len = int(round((self.num_graphs - test_len) * __TRAIN_SPLIT__))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]
            test_indices = indices[train_len + val_len:]

            train_data = []
            val_data = []
            test_data = []

            if self.dataset.split_data_category_graph is not None:
                for i, g in enumerate(self.dataset.split_data_category_graph):
                    if g['x'].shape[0] <= __MAX_NODES__:
                        g['edge_attr'] = torch.zeros(g['edge_index'].shape[-1], 2, dtype=torch.float)
                        g['edge_attr'][:, 1] = 1
                        g['num_nodes'] = g['x'].shape[0]
                        g['y'] = torch.zeros([1, 0]).float()

                        if self.pre_filter is not None and not self.pre_filter(g):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(g)

                        if i in train_indices:
                            train_data.append(g)
                        elif i in val_indices:
                            val_data.append(g)
                        elif i in test_indices:
                            test_data.append(g)
                        else:
                            raise ValueError(f'Index {i} not in any split')

            torch.save(self.collate(train_data), self.processed_paths[self.file_idx['train']])
            torch.save(self.collate(val_data), self.processed_paths[self.file_idx['val']])
            torch.save(self.collate(test_data), self.processed_paths[self.file_idx['test']])


class CategoricalGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200, single_graph=False):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]

        if not single_graph:
            root_path = os.path.join(base_path, self.datadir)
        else:
            root_path = os.path.join(base_path, "single_"+self.datadir)

        datasets = {
            'train': CategoricalGraphDataset(dataset_name=self.cfg.dataset.name, split='train', root=root_path, single_graph=single_graph),
            'val': CategoricalGraphDataset(dataset_name=self.cfg.dataset.name, split='val', root=root_path, single_graph=single_graph),
            'test': CategoricalGraphDataset(dataset_name=self.cfg.dataset.name, split='test', root=root_path, single_graph=single_graph)
            }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class CategoricalDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()         
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)