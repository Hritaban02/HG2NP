__HOME_DIR__ = "/home/du4/19CS30053/MTP2"
__MAX_NODES__ = 200

import sys
sys.path.append(f"{__HOME_DIR__}/Model/src")

from src.lib import *
from src.dataset import Heterogeneous_Graph_Dataset, visualize_categorical_graph, plot_node_frequency, plot_node_frequency_with_cap
from src.utils import get_combinations, split_criteria_dictionary


for dataset_name in split_criteria_dictionary:
    print(f"Loading Dataset {dataset_name} with no split ...", flush=True)
    D = Heterogeneous_Graph_Dataset(dataset_name=dataset_name)
    print("Done.", flush=True)

    print(f"Get Categorical Graph for whole dataset {dataset_name} ...", flush=True)
    D.get_categorical_graph()
    print("Done.", flush=True)
    
    # Images
    print(f"Visualizing the whole dataset {dataset_name} ...", flush=True)
    visualize_categorical_graph(D.dataset_category_graph, f"{dataset_name}/full.png")
    print("Done.", flush=True)

    for split_criteria in split_criteria_dictionary[dataset_name] + get_combinations(split_criteria_dictionary[dataset_name], start_len=2):
        print(f"Loading Dataset {dataset_name} with {split_criteria} split ...", flush=True)
        D = Heterogeneous_Graph_Dataset(dataset_name=dataset_name, split_criteria=split_criteria)
        print("Done.", flush=True)

        print(f"Get Categorical Graph for whole dataset {dataset_name} with {split_criteria} split ...", flush=True)
        D.get_categorical_graph(on_split_data=True)
        print("Done.", flush=True)

        title=""
        if isinstance(split_criteria, list):
            for s in split_criteria:
                title+="_"
                title+=s
        else:
            title=split_criteria
        
        if D.split_data_category_graph is not None:
            count = 0
            for i, g in enumerate(D.split_data_category_graph):
                # Images 
                print(f"Visualizing the dataset {dataset_name} with {split_criteria} split ...", flush=True)
                if(g.num_nodes <= __MAX_NODES__):    
                    visualize_categorical_graph(g, f"{dataset_name}/{title}/sub_graphs_with_200_cap/{i}.png")
                    count+=1
                else:    
                    visualize_categorical_graph(g, f"{dataset_name}/{title}/sub_graphs/{i}.png")
                print("Done.", flush=True)

            print(f"Dataset : {dataset_name}, Split Criteria : {title}, Number Of Graphs : {count}")

            # Frequency Plots
            print(f"Plotting Node Frequency of the dataset {dataset_name} with {split_criteria} split ...", flush=True)
            plot_node_frequency(D.split_data_category_graph, f"{dataset_name}/{title}/Node_Frequency_Plot.png", f"{dataset_name}_{title}")
            print("Done.", flush=True)

            # Frequency Plots
            print(f"Plotting Node Frequency (200 Cap) of the dataset {dataset_name} with {split_criteria} split ...", flush=True)
            plot_node_frequency_with_cap(D.split_data_category_graph, f"{dataset_name}/{title}/Node_Frequency_Plot_with_200_Cap.png", f"{dataset_name}_{title}", cap=200)
            print("Done.", flush=True)