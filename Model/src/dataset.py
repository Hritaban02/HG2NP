from lib import *
__HOME_DIR__ = "/home/du4/19CS30053/MTP2"
path_prefix=f"{__HOME_DIR__}/Model"


def plot_node_frequency(set_of_graphs:list[Data], filename:str, title:Optional[str]=None):
    plt.rcParams["figure.figsize"] = [8, 8]
    freq = []
    for g in set_of_graphs:
        freq.append(g.num_nodes)
    freq.sort()

    plt.scatter(range(0, len(freq)), freq)
    plt.ylabel('Number of Nodes')
    if title is not None:
        plt.title(f'Node Frequency Plot - {title}')
    else:
        plt.title(f'Node Frequency Plot')
    os.makedirs(os.path.dirname(f'{path_prefix}/images/{filename}'), exist_ok=True)
    plt.savefig(f'{path_prefix}/images/{filename}')
    plt.close()

def plot_node_frequency_with_cap(set_of_graphs:list[Data], filename:str, title:Optional[str]=None, cap:int=200):
    plt.rcParams["figure.figsize"] = [8, 8]
    freq = []
    for g in set_of_graphs:
        freq.append(g.num_nodes)
    freq.sort()

    plt.scatter(range(0, len(freq)), freq)
    plt.axhline(y = 200, color = 'r', linestyle = '-') 
    plt.ylabel('Number of Nodes')
    if title is not None:
        plt.title(f'Node Frequency Plot - {title}')
    else:
        plt.title(f'Node Frequency Plot')
    os.makedirs(os.path.dirname(f'{path_prefix}/images/{filename}'), exist_ok=True)
    plt.savefig(f'{path_prefix}/images/{filename}')
    plt.close()

def visualize_categorical_graph(graph:Data, filename:str):
    plt.rcParams["figure.figsize"] = [min((max(200.0, graph.num_nodes)/200.0)*8, 500.0), min((max(200.0, graph.num_nodes)/200.0)*8, 500.0)]
    G = to_networkx(graph, node_attrs=["x"])
    color_map = []

    for node in G.nodes(data=True):
        if len(node[1]['x'])==3:
            if node[1]['x']==[1.0, 0.0, 0.0]:
                color_map.append('#fc0303')
            elif node[1]['x']==[0.0, 1.0, 0.0]:
                color_map.append('#0307fc')
            elif node[1]['x']==[0.0, 0.0, 1.0]:
                color_map.append('#03fc2c')
        elif len(node[1]['x'])==4:
            if node[1]['x']==[1.0, 0.0, 0.0, 0.0]:
                color_map.append('#fc0303')
            elif node[1]['x']==[0.0, 1.0, 0.0, 0.0]:
                color_map.append('#0307fc')
            elif node[1]['x']==[0.0, 0.0, 1.0, 0.0]:
                color_map.append('#03fc2c')
            elif node[1]['x']==[0.0, 0.0, 0.0, 1.0]:
                color_map.append('#fcfc03')
    
    pos = nx.spring_layout(G, k=0.5, iterations=20)
    nx.draw(G, pos=pos, node_color=color_map, node_size=(max(200.0, graph.num_nodes)/200.0)*100)
    os.makedirs(os.path.dirname(f'{path_prefix}/images/{filename}'), exist_ok=True)
    plt.savefig(f'{path_prefix}/images/{filename}')
    plt.close()

class IMDB_modified(InMemoryDataset):
    r"""A subset of the Internet Movie Database (IMDB), as collected in the
    `"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph
    Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    IMDB is a heterogeneous graph containing three types of entities - movies
    (4,278 nodes), actors (5,257 nodes), and directors (2,081 nodes).
    The movies are divided into three classes (action, comedy, drama) according
    to their genre.
    Movie features correspond to elements of a bag-of-words representation of
    its plot keywords.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    # url = 'https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        # self.load(self.processed_paths[0], data_cls=HeteroData)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npz',
            'labels.npy', 'train_val_test_idx.npz', 'years.npy', 'countries.npy', 'languages.npy'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        # path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.remove(path)

        shutil.copytree(f"{__HOME_DIR__}/MAGNN/data/preprocessed/IMDB_processed", self.raw_dir, dirs_exist_ok=True)

    def process(self) -> None:
        data = HeteroData()

        node_types = ['movie', 'director', 'actor']
        for i, node_type in enumerate(node_types):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        year = np.load(osp.join(self.raw_dir, 'years.npy'), allow_pickle=True)
        language = np.load(osp.join(self.raw_dir, 'languages.npy'), allow_pickle=True)
        country = np.load(osp.join(self.raw_dir, 'countries.npy'), allow_pickle=True)
        
        data['movie'].y = torch.from_numpy(y).to(torch.long)

        self.yle = preprocessing.LabelEncoder()
        data['movie'].year = torch.from_numpy(self.yle.fit_transform(year)).to(torch.long)

        self.lle = preprocessing.LabelEncoder()
        data['movie'].language = torch.from_numpy(self.lle.fit_transform(language)).to(torch.long)

        self.cle = preprocessing.LabelEncoder()
        data['movie'].country = torch.from_numpy(self.cle.fit_transform(country)).to(torch.long)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data['movie'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['movie'][f'{name}_mask'] = mask

        s = {}
        N_m = data['movie'].num_nodes
        N_d = data['director'].num_nodes
        N_a = data['actor'].num_nodes
        s['movie'] = (0, N_m)
        s['director'] = (N_m, N_m + N_d)
        s['actor'] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # self.save(data, self.processed_paths[0])
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class DBLP_modified(InMemoryDataset):
    r"""A subset of the DBLP computer science bibliography website, as
    collected in the `"MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    DBLP is a heterogeneous graph containing four types of entities - authors
    (4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences
    (20 nodes).
    The authors are divided into four research areas (database, data mining,
    artificial intelligence, information retrieval).
    Each author is described by a bag-of-words representation of their paper
    keywords.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Node/Edge Type
          - #nodes/#edges
          - #features
          - #classes
        * - Author
          - 4,057
          - 334
          - 4
        * - Paper
          - 14,328
          - 4,231
          -
        * - Term
          - 7,723
          - 50
          -
        * - Conference
          - 20
          - 0
          -
        * - Author-Paper
          - 196,425
          -
          -
        * - Paper-Term
          - 85,810
          -
          -
        * - Conference-Paper
          - 14,328
          -
          -
    """

    # url = 'https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npy',
            'labels.npy', 'node_types.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.remove(path)

        shutil.copytree(f"{__HOME_DIR__}/MAGNN/data/preprocessed/DBLP_processed", self.raw_dir, dirs_exist_ok=True)

    def process(self):
        data = HeteroData()

        node_types = ['author', 'paper', 'term', 'conference']
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        data['paper'].y = data['paper'].x[:, -1]
        data['paper'].x = data['paper'].x[:, :-1]

        x = np.load(osp.join(self.raw_dir, 'features_2.npy'))
        data['term'].x = torch.from_numpy(x).to(torch.float)

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        data['conference'].num_nodes = int((node_type_idx == 3).sum())

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['author'].y = torch.from_numpy(y).to(torch.long)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['author'][f'{name}_mask'] = mask

        s = {}
        N_a = data['author'].num_nodes
        N_p = data['paper'].num_nodes
        N_t = data['term'].num_nodes
        N_c = data['conference'].num_nodes
        s['author'] = (0, N_a)
        s['paper'] = (N_a, N_a + N_p)
        s['term'] = (N_a + N_p, N_a + N_p + N_t)
        s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class Heterogeneous_Graph_Dataset():
    def __init__(self, dataset_name:str, split_criteria:Optional[Union[str, list[str]]]=None):
        self.dataset_name = dataset_name
        self.split_criteria = split_criteria

        if self.dataset_name=="DBLP":
            if self.split_criteria in ["author", "conference", "type"]:
                # HeteroData(
                #     author={
                #         x=[4057, 334],
                #         y=[4057],
                #         train_mask=[4057],
                #         val_mask=[4057],
                #         test_mask=[4057],
                #     },
                #     paper={ x=[14328, 4231] },
                #     term={ x=[7723, 50] },
                #     conference={ num_nodes=20 },
                #     (author, to, paper)={ edge_index=[2, 19645] },
                #     (paper, to, author)={ edge_index=[2, 19645] },
                #     (paper, to, term)={ edge_index=[2, 85810] },
                #     (paper, to, conference)={ edge_index=[2, 14328] },
                #     (term, to, paper)={ edge_index=[2, 85810] },
                #     (conference, to, paper)={ edge_index=[2, 14328] }
                # )

                # use HAN paper's preprocessed data as the features of authors (https://github.com/Jhy1993/HAN)
                # The authors are divided into four research areas (database, data mining, artificial intelligence, information retrieval). 
                # Each author is described by a bag-of-words representation of their paper keywords.
                
                # use bag-of-words representation of paper titles as the features of papers
                
                # use pretrained GloVe vectors as the features of terms

                self.dataset = DBLP_modified(root=f"{path_prefix}/data/{self.dataset_name}_{self.split_criteria}")

            elif isinstance(self.split_criteria, list):
                if len(set(self.split_criteria))!=len(self.split_criteria):
                    sys.exit("ERROR: Split criteria cannot have repititions.")

                for s in self.split_criteria:
                    if s not in ["author", "conference", "type"]:
                        sys.exit("ERROR: No such split criteria.")

                self.split_criteria.sort()

                title = ""
                for s in self.split_criteria:
                    title+="_"
                    title+=s

                self.dataset = DBLP_modified(root=f"{path_prefix}/data/{self.dataset_name}_{title}")

            elif self.split_criteria is None:
                self.dataset = DBLP_modified(root=f"{path_prefix}/data/{self.dataset_name}")
            else:
                sys.exit("ERROR: No such split criteria.")

        elif self.dataset_name=="IMDB":
            if self.split_criteria in ["movie", "year", "country", "language"]:
                # HeteroData(
                #     movie={
                #         x=[4278, 3066],
                #         y=[4278],
                #         train_mask=[4278],
                #         val_mask=[4278],
                #         test_mask=[4278],
                #     },
                #     director={ x=[2081, 3066] },
                #     actor={ x=[5257, 3066] },
                #     (movie, to, director)={ edge_index=[2, 4278] },
                #     (movie, to, actor)={ edge_index=[2, 12828] },
                #     (director, to, movie)={ edge_index=[2, 4278] },
                #     (actor, to, movie)={ edge_index=[2, 12828] }
                # )

                # extract bag-of-word representations of plot keywords for each movie
                # The movies are divided into three classes (action, comedy, drama) according to their genre. 
                # Movie features correspond to elements of a bag-of-words representation of its plot keywords.

                # assign features to directors and actors as the means of their associated movies' features

                self.dataset = IMDB_modified(root=f"{path_prefix}/data/{self.dataset_name}_{self.split_criteria}")

            elif isinstance(self.split_criteria, list):
                if len(set(self.split_criteria))!=len(self.split_criteria):
                    sys.exit("ERROR: Split criteria cannot have repititions.")

                for s in self.split_criteria:
                    if s not in ["movie", "year", "country", "language"]:
                        sys.exit("ERROR: No such split criteria.")

                self.split_criteria.sort()

                title = ""
                for s in self.split_criteria:
                    title+="_"
                    title+=s

                self.dataset = IMDB_modified(root=f"{path_prefix}/data/{self.dataset_name}_{title}")

            elif split_criteria is None:
                self.dataset = IMDB_modified(root=f"{path_prefix}/data/{self.dataset_name}")
            else:
                sys.exit("ERROR: No such split criteria.")
        else:
            sys.exit("ERROR: Invalid dataset name.")

        if self.split_criteria is not None:
            self.split_graph(self.split_criteria)

    def __split_function__DBLP_author(self, graph:HeteroData):
        split_data = []

        for i in torch.unique(graph['author']['y']):
            data = HeteroData()

            indices = (graph['author']['y'] == i).nonzero()
            
            index2author = torch.reshape(indices, (-1,))
            data['author'].x = graph['author']['x'][index2author]
            data['author'].y = graph['author']['y'][index2author]
            author2index = {k.item(): v for v, k in enumerate(index2author)}

            data['author', 'to', 'paper'].edge_index = []

            for a in index2author:
                indices_ = (graph['author', 'to', 'paper']['edge_index'][0, :] == a).nonzero()
                indices_ = torch.reshape(indices_, (-1,))

                data['author', 'to', 'paper'].edge_index.append(graph['author', 'to', 'paper']['edge_index'][:, indices_])
        
            data['author', 'to', 'paper'].edge_index = torch.cat(data['author', 'to', 'paper'].edge_index, dim=1)


            index2paper = torch.unique(data['author', 'to', 'paper']['edge_index'][1, :])
            data['paper'].x = graph['paper']['x'][index2paper]
            data['paper'].y = graph['paper']['y'][index2paper]
            paper2index = {k.item(): v for v, k in enumerate(index2paper)}

            data['paper', 'to', 'conference'].edge_index = []
            data['paper', 'to', 'term'].edge_index = []

            for p in index2paper:
                indices_ = (graph['paper', 'to', 'conference']['edge_index'][0, :] == p).nonzero()
                indices_ = torch.reshape(indices_, (-1,))
                data['paper', 'to', 'conference'].edge_index.append(graph['paper', 'to', 'conference']['edge_index'][:, indices_])

                indices__ = (graph['paper', 'to', 'term']['edge_index'][0, :] == p).nonzero()
                indices__ = torch.reshape(indices__, (-1,))
                data['paper', 'to', 'term'].edge_index.append(graph['paper', 'to', 'term']['edge_index'][:, indices__])

            data['paper', 'to', 'conference'].edge_index = torch.cat(data['paper', 'to', 'conference'].edge_index, dim=1)
            data['paper', 'to', 'term'].edge_index = torch.cat(data['paper', 'to', 'term'].edge_index, dim=1)

            index2conference = torch.unique(data['paper', 'to', 'conference'].edge_index[1, :])
            data['conference'].num_nodes = index2conference.shape[0]
            conference2index = {k.item(): v for v, k in enumerate(index2conference)}

            index2term = torch.unique(data['paper', 'to', 'term'].edge_index[1, :])
            data['term'].x = graph['term']['x'][index2term]
            term2index = {k.item(): v for v, k in enumerate(index2term)}

            
            for j in range(data['author', 'to', 'paper'].edge_index.shape[1]):
                data['author', 'to', 'paper'].edge_index[0, j]=author2index[data['author', 'to', 'paper'].edge_index[0, j].item()]
                data['author', 'to', 'paper'].edge_index[1, j]=paper2index[data['author', 'to', 'paper'].edge_index[1, j].item()]

            data['paper', 'to', 'author'].edge_index = data['author', 'to', 'paper'].edge_index[[1, 0]]

            for j in range(data['paper', 'to', 'conference'].edge_index.shape[1]):
                data['paper', 'to', 'conference'].edge_index[0, j]=paper2index[data['paper', 'to', 'conference'].edge_index[0, j].item()]
                data['paper', 'to', 'conference'].edge_index[1, j]=conference2index[data['paper', 'to', 'conference'].edge_index[1, j].item()]

            data['conference', 'to', 'paper'].edge_index = data['paper', 'to', 'conference'].edge_index[[1, 0]]

            for j in range(data['paper', 'to', 'term'].edge_index.shape[1]):
                data['paper', 'to', 'term'].edge_index[0, j]=paper2index[data['paper', 'to', 'term'].edge_index[0, j].item()]
                data['paper', 'to', 'term'].edge_index[1, j]=term2index[data['paper', 'to', 'term'].edge_index[1, j].item()]

            data['term', 'to', 'paper'].edge_index = data['paper', 'to', 'term'].edge_index[[1, 0]]

            split_data.append(data)

        return split_data

    def __split_function__DBLP_type(self, graph:HeteroData):
        split_data = []

        for i in torch.unique(graph['paper']['y']):
            data = HeteroData()

            indices = (graph['paper']['y'] == i).nonzero()
            
            index2paper = torch.reshape(indices, (-1,))
            data['paper'].x = graph['paper']['x'][index2paper]
            data['paper'].y = graph['paper']['y'][index2paper]
            paper2index = {k.item(): v for v, k in enumerate(index2paper)}

            data['paper', 'to', 'author'].edge_index = []

            for a in index2paper:
                indices_ = (graph['paper', 'to', 'author']['edge_index'][0, :] == a).nonzero()
                indices_ = torch.reshape(indices_, (-1,))

                data['paper', 'to', 'author'].edge_index.append(graph['paper', 'to', 'author']['edge_index'][:, indices_])
        
            data['paper', 'to', 'author'].edge_index = torch.cat(data['paper', 'to', 'author'].edge_index, dim=1)

            index2author = torch.unique(data['paper', 'to', 'author']['edge_index'][1, :])
            data['author'].x = graph['author']['x'][index2author]
            data['author'].y = graph['author']['y'][index2author]
            author2index = {k.item(): v for v, k in enumerate(index2author)}

            data['paper', 'to', 'conference'].edge_index = []
            data['paper', 'to', 'term'].edge_index = []

            for p in index2paper:
                indices_ = (graph['paper', 'to', 'conference']['edge_index'][0, :] == p).nonzero()
                indices_ = torch.reshape(indices_, (-1,))
                data['paper', 'to', 'conference'].edge_index.append(graph['paper', 'to', 'conference']['edge_index'][:, indices_])

                indices__ = (graph['paper', 'to', 'term']['edge_index'][0, :] == p).nonzero()
                indices__ = torch.reshape(indices__, (-1,))
                data['paper', 'to', 'term'].edge_index.append(graph['paper', 'to', 'term']['edge_index'][:, indices__])

            data['paper', 'to', 'conference'].edge_index = torch.cat(data['paper', 'to', 'conference'].edge_index, dim=1)
            data['paper', 'to', 'term'].edge_index = torch.cat(data['paper', 'to', 'term'].edge_index, dim=1)

            index2conference = torch.unique(data['paper', 'to', 'conference'].edge_index[1, :])
            data['conference'].num_nodes = index2conference.shape[0]
            conference2index = {k.item(): v for v, k in enumerate(index2conference)}

            index2term = torch.unique(data['paper', 'to', 'term'].edge_index[1, :])
            data['term'].x = graph['term']['x'][index2term]
            term2index = {k.item(): v for v, k in enumerate(index2term)}
            
            for j in range(data['paper', 'to', 'author'].edge_index.shape[1]):
                data['paper', 'to', 'author'].edge_index[0, j]=paper2index[data['paper', 'to', 'author'].edge_index[0, j].item()]
                data['paper', 'to', 'author'].edge_index[1, j]=author2index[data['paper', 'to', 'author'].edge_index[1, j].item()]

            data['author', 'to', 'paper'].edge_index = data['paper', 'to', 'author'].edge_index[[1, 0]]

            for j in range(data['paper', 'to', 'conference'].edge_index.shape[1]):
                data['paper', 'to', 'conference'].edge_index[0, j]=paper2index[data['paper', 'to', 'conference'].edge_index[0, j].item()]
                data['paper', 'to', 'conference'].edge_index[1, j]=conference2index[data['paper', 'to', 'conference'].edge_index[1, j].item()]

            data['conference', 'to', 'paper'].edge_index = data['paper', 'to', 'conference'].edge_index[[1, 0]]

            for j in range(data['paper', 'to', 'term'].edge_index.shape[1]):
                data['paper', 'to', 'term'].edge_index[0, j]=paper2index[data['paper', 'to', 'term'].edge_index[0, j].item()]
                data['paper', 'to', 'term'].edge_index[1, j]=term2index[data['paper', 'to', 'term'].edge_index[1, j].item()]

            data['term', 'to', 'paper'].edge_index = data['paper', 'to', 'term'].edge_index[[1, 0]]

            split_data.append(data)

        return split_data

    def __split_function__DBLP_conference(self, graph:HeteroData):
        split_data = []

        for i in range(graph['conference']['num_nodes']):
            data = HeteroData()
            
            indices = (graph['conference', 'to', 'paper']['edge_index'][0, :] == i).nonzero()
            indices = torch.reshape(indices, (-1,))
            
            data['paper', 'to', 'author'].edge_index = []
            data['paper', 'to', 'term'].edge_index = []

            index2paper = torch.unique(graph['conference', 'to', 'paper']['edge_index'][1, indices])
            data['paper'].x = graph['paper']['x'][index2paper]
            data['paper'].y = graph['paper']['y'][index2paper]
            paper2index = {k.item(): v for v, k in enumerate(index2paper)}

            for p in index2paper:
                indices_ = (graph['paper', 'to', 'author']['edge_index'][0, :] == p).nonzero()
                indices_ = torch.reshape(indices_, (-1,))
                data['paper', 'to', 'author'].edge_index.append(graph['paper', 'to', 'author']['edge_index'][:, indices_])

                indices__ = (graph['paper', 'to', 'term']['edge_index'][0, :] == p).nonzero()
                indices__ = torch.reshape(indices__, (-1,))
                data['paper', 'to', 'term'].edge_index.append(graph['paper', 'to', 'term']['edge_index'][:, indices__])

            data['paper', 'to', 'author'].edge_index = torch.cat(data['paper', 'to', 'author'].edge_index, dim=1)
            data['paper', 'to', 'term'].edge_index = torch.cat(data['paper', 'to', 'term'].edge_index, dim=1)


            index2author = torch.unique(data['paper', 'to', 'author'].edge_index[1, :])
            data['author'].x = graph['author']['x'][index2author]
            data['author'].y = graph['author']['y'][index2author]
            author2index = {k.item(): v for v, k in enumerate(index2author)}

            index2term = torch.unique(data['paper', 'to', 'term'].edge_index[1, :])
            data['term'].x = graph['term']['x'][index2term]
            term2index = {k.item(): v for v, k in enumerate(index2term)}

            for j in range(data['paper', 'to', 'author'].edge_index.shape[1]):
                data['paper', 'to', 'author'].edge_index[0, j]=paper2index[data['paper', 'to', 'author'].edge_index[0, j].item()]
                data['paper', 'to', 'author'].edge_index[1, j]=author2index[data['paper', 'to', 'author'].edge_index[1, j].item()]

            data['author', 'to', 'paper'].edge_index = data['paper', 'to', 'author'].edge_index[[1, 0]]

            for j in range(data['paper', 'to', 'term'].edge_index.shape[1]):
                data['paper', 'to', 'term'].edge_index[0, j]=paper2index[data['paper', 'to', 'term'].edge_index[0, j].item()]
                data['paper', 'to', 'term'].edge_index[1, j]=term2index[data['paper', 'to', 'term'].edge_index[1, j].item()]

            data['term', 'to', 'paper'].edge_index = data['paper', 'to', 'term'].edge_index[[1, 0]]

            split_data.append(data)

        return split_data

    def __split_function__IMDB_all(self, graph:HeteroData, split_criteria:str):
        split_data = []

        if split_criteria=="movie":
            split_attr = "y"
        else:
            split_attr = split_criteria

        for i in torch.unique(graph['movie'][split_attr]):
            data = HeteroData()

            indices = (graph['movie'][split_attr] == i).nonzero()
            
            index2movie = torch.reshape(indices, (-1,))
            data['movie'].x = graph['movie']['x'][index2movie]

            data['movie']['y'] = graph['movie']['y'][index2movie]
            data['movie']['year'] = graph['movie']['year'][index2movie]
            data['movie']['language'] = graph['movie']['language'][index2movie]
            data['movie']['country'] = graph['movie']['country'][index2movie]
            movie2index = {k.item(): v for v, k in enumerate(index2movie)}

            data['movie', 'to', 'actor'].edge_index = []
            data['movie', 'to', 'director'].edge_index = []

            for p in index2movie:
                indices_ = (graph['movie', 'to', 'actor']['edge_index'][0, :] == p).nonzero()
                indices_ = torch.reshape(indices_, (-1,))
                data['movie', 'to', 'actor'].edge_index.append(graph['movie', 'to', 'actor']['edge_index'][:, indices_])

                indices__ = (graph['movie', 'to', 'director']['edge_index'][0, :] == p).nonzero()
                indices__ = torch.reshape(indices__, (-1,))
                data['movie', 'to', 'director'].edge_index.append(graph['movie', 'to', 'director']['edge_index'][:, indices__])

            data['movie', 'to', 'actor'].edge_index = torch.cat(data['movie', 'to', 'actor'].edge_index, dim=1)
            data['movie', 'to', 'director'].edge_index = torch.cat(data['movie', 'to', 'director'].edge_index, dim=1)

            index2actor = torch.unique(data['movie', 'to', 'actor'].edge_index[1, :])
            data['actor'].x = graph['actor']['x'][index2actor]
            actor2index = {k.item(): v for v, k in enumerate(index2actor)}

            index2director = torch.unique(data['movie', 'to', 'director'].edge_index[1, :])
            data['director'].x = graph['director']['x'][index2director]
            director2index = {k.item(): v for v, k in enumerate(index2director)}


            for j in range(data['movie', 'to', 'actor'].edge_index.shape[1]):
                data['movie', 'to', 'actor'].edge_index[0, j]=movie2index[data['movie', 'to', 'actor'].edge_index[0, j].item()]
                data['movie', 'to', 'actor'].edge_index[1, j]=actor2index[data['movie', 'to', 'actor'].edge_index[1, j].item()]

            data['actor', 'to', 'movie'].edge_index = data['movie', 'to', 'actor'].edge_index[[1, 0]]

            for j in range(data['movie', 'to', 'director'].edge_index.shape[1]):
                data['movie', 'to', 'director'].edge_index[0, j]=movie2index[data['movie', 'to', 'director'].edge_index[0, j].item()]
                data['movie', 'to', 'director'].edge_index[1, j]=director2index[data['movie', 'to', 'director'].edge_index[1, j].item()]

            data['director', 'to', 'movie'].edge_index = data['movie', 'to', 'director'].edge_index[[1, 0]]

            split_data.append(data)

        return split_data

    def split_graph(self, split_criteria:Union[str, list[str]]):
        if hasattr(Heterogeneous_Graph_Dataset, 'split_data_category_graph'):
            self.split_data_category_graph = None
         
        self.split_data = []

        if self.dataset_name=="DBLP":
            if split_criteria=="author":
                # Author is connected to Paper
                # Paper is connected to Conference and Term

                self.split_data = self.__split_function__DBLP_author(graph=self.dataset.data)

            elif split_criteria=="conference":
                # Conference is connected to Paper
                # Paper is connected to Author and Term

                self.split_data = self.__split_function__DBLP_conference(graph=self.dataset.data)

            elif split_criteria=="type":
                # Paper is connected to Author, Term and Conference
 
                self.split_data = self.__split_function__DBLP_type(graph=self.dataset.data)

            elif isinstance(self.split_criteria, list):
                self.split_data = [self.dataset.data]

                new_split_data = []
                
                for s in self.split_criteria:
                    if s=="conference":
                        continue
                    for g in self.split_data:
                        if s=="author":
                            new_split_data.extend(self.__split_function__DBLP_author(graph=g))
                        elif s=="type":
                            new_split_data.extend(self.__split_function__DBLP_type(graph=g))
                        
                    self.split_data=new_split_data
                    new_split_data=[]

                if "conference" in self.split_criteria:
                    for g in self.split_data:
                        new_split_data.extend(self.__split_function__DBLP_conference(graph=g))
                    
                    self.split_data=new_split_data
                    new_split_data=[]

        elif self.dataset_name=="IMDB":
            if split_criteria in ["movie", "year", "country", "language"]:
                # Movie is connected to Actor and Director

                self.split_data = self.__split_function__IMDB_all(graph=self.dataset.data, split_criteria=split_criteria)

            elif isinstance(self.split_criteria, list):
                self.split_data = [self.dataset.data]

                new_split_data = []
                
                for s in self.split_criteria:
                    for g in self.split_data:
                        new_split_data.extend(self.__split_function__IMDB_all(graph=g, split_criteria=s))

                    self.split_data=new_split_data
                    new_split_data=[]

    def get_categorical_graph(self, on_split_data=False):
        if self.dataset_name=="DBLP":
            category_map = {
                'author':torch.tensor([1.0, 0.0, 0.0, 0.0]), 
                'paper':torch.tensor([0.0, 1.0, 0.0, 0.0]), 
                'term':torch.tensor([0.0, 0.0, 1.0, 0.0]), 
                'conference':torch.tensor([0.0, 0.0, 0.0, 1.0])
            }

            if not on_split_data:
                self.dataset_category_graph = Data()
                
                self.dataset_category_graph.x = torch.cat((
                    category_map['author'].repeat(self.dataset.data['author']['x'].shape[0], 1),
                    category_map['paper'].repeat(self.dataset.data['paper']['x'].shape[0], 1),
                    category_map['term'].repeat(self.dataset.data['term']['x'].shape[0], 1),
                    category_map['conference'].repeat(self.dataset.data['conference']['num_nodes'], 1)
                ), dim=0)
                
                self.dataset_category_graph.edge_index = torch.cat((
                    torch.add(self.dataset.data['author', 'to', 'paper']['edge_index'], 
                              torch.tensor([0, 
                                            self.dataset.data['author']['x'].shape[0]], dtype=int).repeat(self.dataset.data['author', 'to', 'paper']['edge_index'].shape[1], 1).t()),
                    torch.add(self.dataset.data['paper', 'to', 'term']['edge_index'], 
                              torch.tensor([self.dataset.data['author']['x'].shape[0], 
                                            self.dataset.data['author']['x'].shape[0]+self.dataset.data['paper']['x'].shape[0]], dtype=int).repeat(self.dataset.data['paper', 'to', 'term']['edge_index'].shape[1], 1).t()),
                    torch.add(self.dataset.data['paper', 'to', 'conference']['edge_index'], 
                              torch.tensor([self.dataset.data['author']['x'].shape[0], 
                                            self.dataset.data['author']['x'].shape[0]+self.dataset.data['paper']['x'].shape[0]+self.dataset.data['term']['x'].shape[0]], dtype=int).repeat(self.dataset.data['paper', 'to', 'conference']['edge_index'].shape[1], 1).t()),
                ), dim=1)
                
                self.dataset_category_graph.edge_index = torch.cat((self.dataset_category_graph.edge_index,
                                                                    self.dataset_category_graph.edge_index[[1, 0]]), dim=1).int()

                return self.dataset_category_graph
            else:
                if self.split_criteria is None:
                    sys.exit("ERROR: Splitting not yet performed. Please split the data first.")
                elif self.split_criteria=="author" or self.split_criteria=="type" or (isinstance(self.split_criteria, list) and "conference" not in self.split_criteria):
                    self.split_data_category_graph = []

                    for G in self.split_data:
                        data = Data()

                        data.x = torch.cat((
                            category_map['author'].repeat(G['author']['x'].shape[0], 1),
                            category_map['paper'].repeat(G['paper']['x'].shape[0], 1),
                            category_map['term'].repeat(G['term']['x'].shape[0], 1),
                            category_map['conference'].repeat(G['conference']['num_nodes'], 1)
                        ), dim=0)
                        
                        data.edge_index = torch.cat((
                            torch.add(G['author', 'to', 'paper']['edge_index'], 
                                      torch.tensor([0, 
                                                    G['author']['x'].shape[0]], dtype=int).repeat(G['author', 'to', 'paper']['edge_index'].shape[1], 1).t()),
                            torch.add(G['paper', 'to', 'term']['edge_index'], 
                                      torch.tensor([G['author']['x'].shape[0], 
                                                    G['author']['x'].shape[0]+G['paper']['x'].shape[0]], dtype=int).repeat(G['paper', 'to', 'term']['edge_index'].shape[1], 1).t()),
                            torch.add(G['paper', 'to', 'conference']['edge_index'], 
                                      torch.tensor([G['author']['x'].shape[0], 
                                                    G['author']['x'].shape[0]+G['paper']['x'].shape[0]+G['term']['x'].shape[0]], dtype=int).repeat(G['paper', 'to', 'conference']['edge_index'].shape[1], 1).t()),
                        ), dim=1)
                        
                        data.edge_index = torch.cat((data.edge_index, 
                                                     data.edge_index[[1, 0]]), dim=1).int()

                        self.split_data_category_graph.append(data)

                elif self.split_criteria=="conference" or (isinstance(self.split_criteria, list) and "conference" in self.split_criteria):
                    self.split_data_category_graph = []

                    for G in self.split_data:
                        data = Data()

                        data.x = torch.cat((
                            category_map['author'].repeat(G['author']['x'].shape[0], 1),
                            category_map['paper'].repeat(G['paper']['x'].shape[0], 1),
                            category_map['term'].repeat(G['term']['x'].shape[0], 1),
                        ), dim=0)
                        
                        data.edge_index = torch.cat((
                            torch.add(G['author', 'to', 'paper']['edge_index'], 
                                      torch.tensor([0, 
                                                    G['author']['x'].shape[0]], dtype=int).repeat(G['author', 'to', 'paper']['edge_index'].shape[1], 1).t()),
                            torch.add(G['paper', 'to', 'term']['edge_index'], 
                                      torch.tensor([G['author']['x'].shape[0], 
                                                    G['author']['x'].shape[0]+G['paper']['x'].shape[0]], dtype=int).repeat(G['paper', 'to', 'term']['edge_index'].shape[1], 1).t()),
                        ), dim=1)
                        
                        data.edge_index = torch.cat((data.edge_index, 
                                                     data.edge_index[[1, 0]]), dim=1).int()

                        self.split_data_category_graph.append(data)

        elif self.dataset_name=="IMDB":
            category_map = {
                'movie':torch.tensor([1.0, 0.0, 0.0]), 
                'actor':torch.tensor([0.0, 1.0, 0.0]), 
                'director':torch.tensor([0.0, 0.0, 1.0]), 
            }

            if not on_split_data:
                self.dataset_category_graph = Data()
                
                self.dataset_category_graph.x = torch.cat((
                    category_map['movie'].repeat(self.dataset.data['movie']['x'].shape[0], 1),
                    category_map['actor'].repeat(self.dataset.data['actor']['x'].shape[0], 1),
                    category_map['director'].repeat(self.dataset.data['director']['x'].shape[0], 1),
                ), dim=0)
                
                self.dataset_category_graph.edge_index = torch.cat((
                    torch.add(self.dataset.data['movie', 'to', 'actor']['edge_index'], 
                              torch.tensor([0, 
                                            self.dataset.data['movie']['x'].shape[0]], dtype=int).repeat(self.dataset.data['movie', 'to', 'actor']['edge_index'].shape[1], 1).t()),
                    torch.add(self.dataset.data['movie', 'to', 'director']['edge_index'], 
                              torch.tensor([0, 
                                            self.dataset.data['movie']['x'].shape[0]+self.dataset.data['actor']['x'].shape[0]], dtype=int).repeat(self.dataset.data['movie', 'to', 'director']['edge_index'].shape[1], 1).t()),
                ), dim=1)
                
                self.dataset_category_graph.edge_index = torch.cat((self.dataset_category_graph.edge_index,
                                                                    self.dataset_category_graph.edge_index[[1, 0]]), dim=1).int()

                return self.dataset_category_graph
            else:
                if self.split_criteria is None:
                    sys.exit("ERROR: Splitting not yet performed. Please split the data first.")
                elif self.split_criteria in ["movie", "year", "country", "language"] or isinstance(self.split_criteria, list):
                    self.split_data_category_graph = []

                    for G in self.split_data:
                        data = Data()

                        data.x = torch.cat((
                            category_map['movie'].repeat(G['movie']['x'].shape[0], 1),
                            category_map['actor'].repeat(G['actor']['x'].shape[0], 1),
                            category_map['director'].repeat(G['director']['x'].shape[0], 1),
                        ), dim=0)
                        
                        data.edge_index = torch.cat((
                            torch.add(G['movie', 'to', 'actor']['edge_index'], 
                                      torch.tensor([0, 
                                                    G['movie']['x'].shape[0]], dtype=int).repeat(G['movie', 'to', 'actor']['edge_index'].shape[1], 1).t()),
                            torch.add(G['movie', 'to', 'director']['edge_index'], 
                                      torch.tensor([0, 
                                                    G['movie']['x'].shape[0]+G['actor']['x'].shape[0]], dtype=int).repeat(G['movie', 'to', 'director']['edge_index'].shape[1], 1).t()),
                        ), dim=1)
                        
                        data.edge_index = torch.cat((data.edge_index, 
                                                     data.edge_index[[1, 0]]), dim=1).int()

                        self.split_data_category_graph.append(data)