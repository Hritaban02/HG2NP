from lib import *
from dataset import *

__HOME_DIR__ = "/home/du4/19CS30053/MTP2"
path_prefix=f"{__HOME_DIR__}/Model"


split_criteria_dictionary = {
    "IMDB": ["movie", "year", "language", "country"],
    "DBLP": ["author", "conference", "type"]
}

DBLP_index_to_category = {
    0:'author', 
    1:'paper', 
    2:'term', 
    3:'conference'
}

IMDB_index_to_category = {
    0:'movie', 
    1:'actor', 
    2:'director'
}

DBLP_category_to_vectors = {
    'author':torch.tensor([1.0, 0.0, 0.0, 0.0]), 
    'paper':torch.tensor([0.0, 1.0, 0.0, 0.0]), 
    'term':torch.tensor([0.0, 0.0, 1.0, 0.0]), 
    'conference':torch.tensor([0.0, 0.0, 0.0, 1.0])
}

IMDB_category_to_vectors = {
    'movie':torch.tensor([1.0, 0.0, 0.0]), 
    'actor':torch.tensor([0.0, 1.0, 0.0]), 
    'director':torch.tensor([0.0, 0.0, 1.0]), 
}

def get_combinations(lst, start_len=1): # creating a user-defined method
    combination = [] # empty list 
    for r in range(start_len, len(lst) + 1):
        # to generate combination
        combination.extend(itertools.combinations(lst, r))

    combination = [[item for item in pair] for pair in combination]
    return combination

def get_sample_pool(dataset:str, category:str):
    if dataset=="DBLP":
        if category!="conference":
            return DBLP_modified(root=f"{path_prefix}/data/{dataset}").data[category].x
        else:
            return torch.arange(DBLP_modified(root=f"{path_prefix}/data/{dataset}").data[category].num_nodes)
    if dataset=="IMDB":
        return IMDB_modified(root=f"{path_prefix}/data/{dataset}").data[category].x

def preprocess_generated_samples_text_file(path:str, dataset:str):
    if not os.path.isfile(path=path):
        print(f"ERROR: {path} does not exists.")
    
    with open(path, 'r') as f:
        lines = [line.rstrip() for line in f]
        i = 0
        n = len(lines)

        data_list = []

        while(i<n):
            d = Data()

            # Extract number of nodes
            N = int(lines[i][2:])
            i+=2

            # Extract node types
            if dataset=="DBLP":
                d.x = torch.stack([DBLP_category_to_vectors[DBLP_index_to_category[int(x)]] for x in lines[i].split()], dim=0)
            if dataset=="IMDB":
                d.x = torch.stack([IMDB_category_to_vectors[IMDB_index_to_category[int(x)]] for x in lines[i].split()], dim=0)
            i+=2

            # Extract edges
            edge_index = []
            for j in range(0, N):
                for k, x in enumerate(lines[i+j].split()):
                    if x=='1':
                        edge_index.append(torch.tensor([j, k], dtype=torch.int))
            if len(edge_index)==0:
                i+=N
                i+=1
                continue
            d.edge_index = torch.stack(edge_index, dim=0)
            d.edge_index = torch.t(d.edge_index)
            i+=N

            data_list.append(d)
            i+=1
    
    return data_list

def get_graph_metadata(dataset_name:str, split_criteria:Optional[Union[str, list[str]]]=None):
    metadata = ()
    if dataset_name=="IMDB":
        metadata+=(list(IMDB_category_to_vectors.keys()),)
        edge_list = []
        for key1 in IMDB_category_to_vectors:
            for key2 in IMDB_category_to_vectors:
                edge_list.append((key1, 'to', key2))
        metadata+=(edge_list,)
    if dataset_name=="DBLP":
        if "conference"==split_criteria or "conference" in split_criteria:
            category_list = list(DBLP_category_to_vectors.keys())
            category_list.remove("conference")
            metadata+=(category_list,)
            edge_list = []
            for key1 in category_list:
                for key2 in category_list:
                    edge_list.append((key1, 'to', key2))
            metadata+=(edge_list,)
        else:
            metadata+=(list(DBLP_category_to_vectors.keys()),)
            edge_list = []
            for key1 in DBLP_category_to_vectors:
                for key2 in DBLP_category_to_vectors:
                    edge_list.append((key1, 'to', key2))
            metadata+=(edge_list,)
    return metadata

def validate_split_criteria(dataset_name:str, split_criteria:Optional[Union[str, list[str]]]):
    if isinstance(split_criteria, str):
        return split_criteria in split_criteria_dictionary[dataset_name]
    else:
        valid_s = copy.deepcopy(split_criteria_dictionary[dataset_name])
        for s in split_criteria:
            if s not in valid_s:
                return False
            valid_s.remove(s)
        return True
  
def compute_d_emd(dataset, real_graph, fake_graph):
    total_nodes_in_real_graph = 0
    for category in real_graph.x_dict:
        total_nodes_in_real_graph += real_graph[category]["x"].shape[0]

    total_nodes_in_fake_graph = 0
    for category in fake_graph.x_dict:
        total_nodes_in_fake_graph += fake_graph[category]["x"].shape[0]

    d_emd_total1 = [0]*2
    d_emd_total2 = [0]*2
    for category in real_graph.x_dict:
        x_r = real_graph[category]["x"].detach().numpy()
        x_g = fake_graph[category]["x"].detach().numpy()

        a, b = np.full(x_r.shape[0], 1.0/x_r.shape[0]), np.full(x_g.shape[0], 1.0/x_g.shape[0])

        if dataset=="DBLP" and category=="term":
            M1 = ot.dist(x_r, x_g, 'euclidean')
            M2 = ot.dist(x_r, x_g, 'jaccard')

            d_emd_sqe = ot.emd2(a, b, M1)
            d_emd_jac = ot.emd2(a, b, M2)
            
            d_emd_total1[0] += (float(x_r.shape[0])/total_nodes_in_real_graph)*d_emd_sqe
            d_emd_total1[1] += (float(x_r.shape[0])/total_nodes_in_real_graph)*d_emd_jac

            d_emd_total2[0] += (float(x_g.shape[0])/total_nodes_in_fake_graph)*d_emd_sqe
            d_emd_total2[1] += (float(x_g.shape[0])/total_nodes_in_fake_graph)*d_emd_jac
        else:
            M = ot.dist(x_r, x_g, 'jaccard')
        
            d_emd_jac = ot.emd2(a, b, M)

            d_emd_total1[0] += (float(x_r.shape[0])/total_nodes_in_real_graph)*d_emd_jac
            d_emd_total1[1] += (float(x_r.shape[0])/total_nodes_in_real_graph)*d_emd_jac

            d_emd_total2[0] += (float(x_g.shape[0])/total_nodes_in_fake_graph)*d_emd_jac
            d_emd_total2[1] += (float(x_g.shape[0])/total_nodes_in_fake_graph)*d_emd_jac

    return d_emd_total1, d_emd_total2

def triangle_count(G):
    return int(np.sum(list(nx.triangles(G).values())) / 3)

def LCC(G):
    return len(max(nx.connected_components(G), key=len))

def powerlaw_coeff(G):
    degrees = np.array([val for (_, val) in G.degree()])
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1), verbose=False).power_law.alpha

def avg_statistics(graph_list, statistic):
    functions = {
        "clustering": nx.average_clustering,
        "triangle": triangle_count,
        "lcc": LCC,
        "powerlaw": powerlaw_coeff,
        "assortativity": nx.degree_assortativity_coefficient,
    }
    avg = 0
    num = len(graph_list)
    for G in graph_list:
        val = functions[statistic](G)
        if np.isnan(val) or math.isinf(val):
            num-=1
        else:
            avg += val
    avg = float(avg)/num
    return avg

def get_random_het_from_hom(data, dataset_name, split_criteria):
    node_type = torch.argmax(data.x, dim=1)
    if dataset_name=="DBLP":
        category_map = DBLP_index_to_category
        if "conference"==split_criteria or "conference" in split_criteria:
            category_map = {k:v for k,v in category_map.items() if v != "conference"}
    if dataset_name=="IMDB":
        category_map = IMDB_index_to_category

    sample_pools = {}
    for key in category_map:
        sample_pools[key]=get_sample_pool(dataset=dataset_name, category=category_map[key])

    indexToPosition = {}
    positionToIndex = {}

    d = HeteroData()

    for key in category_map:
        indexToPosition[key] = torch.nonzero(node_type==key)
        positionToIndex[key] = {x.item(): i for i, x in enumerate(indexToPosition[key])}
        indexToPosition[key] = indexToPosition[key].reshape((-1))
    
        d[category_map[key]]['x']=sample_pools[key][torch.randint(sample_pools[key].shape[0], (indexToPosition[key].shape[0],))] 
        if d[category_map[key]]['x'].shape[0]==0:
            d[category_map[key]]['x']=torch.zeros(1, d[category_map[key]]['x'].shape[1])

    return d

def get_type_degree_distribution(graph):
    num_types = len(graph.nodes(data=True)[0]['x'])
    node_degree_mat = np.zeros((graph.number_of_nodes(), num_types), dtype=int)
    for e in graph.edges:
        node_degree_mat[e[0]][np.argmax(graph.nodes(data=True)[e[1]]['x'])]+=1
        node_degree_mat[e[1]][np.argmax(graph.nodes(data=True)[e[0]]['x'])]+=1
    return node_degree_mat

sys.path.append(f"{__HOME_DIR__}/DiGress")
from src.analysis.dist_helper import compute_mmd, gaussian_emd, gaussian_tv

def type_degree_distribution_stats(num_types, graph_ref_list_mats, graph_pred_list_mats, compute_emd=False):
    # in case an empty graph is generated

    graph_pred_list_mats_remove_empty = [
        mat for mat in graph_pred_list_mats if not mat.shape[0] == 0
    ]

    total_mmd = 0

    for k in range(num_types):
        sample_ref = []
        sample_pred = []
        for i in range(len(graph_ref_list_mats)):
            degree_temp = np.array(np.histogram(graph_ref_list_mats[i][:,k], bins=np.arange(0, graph_ref_list_mats[i].shape[0]+2, 1, dtype=int))[0])
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_mats_remove_empty)):
            degree_temp = np.array(np.histogram(graph_pred_list_mats_remove_empty[i][:,k], bins=np.arange(0, graph_pred_list_mats_remove_empty[i].shape[0]+2, 1, dtype=int))[0])
            sample_pred.append(degree_temp)

        if compute_emd:
            mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
        else:
            mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

        total_mmd+=mmd_dist

    return total_mmd/num_types