'''
Code taken from https://github.com/dmlc/dgl/blob/master/examples/pytorch/vgae/model.py and modified.
'''

__HOME_DIR__ = "/home/du4/19CS30053/MTP2"
__MAX_NODES__ = 200

import sys
sys.path.append(f"{__HOME_DIR__}/VGAE")
from src.model import VGAEModel
from src.utils import vgae_loss

sys.path.append(f"{__HOME_DIR__}/Model/src")
from lib import *
from dataset import Heterogeneous_Graph_Dataset
from utils import validate_split_criteria


wandb.login()
datetime_string = datetime.now().strftime("%Y_%m_%d___%H_%M_%S")

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--split_criteria', type=str, required=True)

args = parser.parse_args()

device = args.device
dataset_name = args.dataset_name[1:-1]                      # Remove quotations
split_criteria = (args.split_criteria[1:-1]).split(',')     # Remove quotations and make list
if len(split_criteria)==1:
    split_criteria=split_criteria[0]

# Sanity Check
if dataset_name not in ["DBLP", "IMDB"]:
  sys.exit(f"ERROR: dataset_name {dataset_name} is not available")

if not validate_split_criteria(dataset_name=dataset_name, split_criteria=split_criteria):
    sys.exit(f"ERROR: split_criteria {split_criteria} is not valid")

# Set Device
device = (device if torch.cuda.is_available() else "cpu")
print(f"Device being used : {device}")

dataset = Heterogeneous_Graph_Dataset(dataset_name=dataset_name, split_criteria=split_criteria)
dataset.get_categorical_graph(on_split_data=True)

# All homogeneous graphs
real_data_list = dataset.split_data_category_graph

new_real_data_list = []
for graph in real_data_list:
    if graph.num_nodes <= __MAX_NODES__:
        graph.edge_index = graph.edge_index.long()
        new_real_data_list.append(graph)

#####################################################################
index_of_graph_with_max_nodes = -1
num_max_nodes = 0
for i, g in enumerate(new_real_data_list):
    if (index_of_graph_with_max_nodes==-1 or g['x'].shape[0]>num_max_nodes) and g['x'].shape[0] <= __MAX_NODES__ :
        index_of_graph_with_max_nodes=i
        num_max_nodes = g['x'].shape[0]
#####################################################################

train_data_list, test_data_list = random_split(dataset=new_real_data_list, lengths=[0.8, 0.2])
train_data_list, val_data_list = random_split(dataset=train_data_list, lengths=[0.9, 0.1])

#####################################################################
train_data_list = [new_real_data_list[index_of_graph_with_max_nodes]]
#####################################################################

train_loader = DataLoader(train_data_list, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=128, shuffle=True)

vgae_model = VGAEModel(32, 16, device)
vgae_model.to(device)

optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.0002, weight_decay=0.1)

dataset_title = dataset_name
if isinstance(split_criteria, str):
    dataset_title+="_"
    dataset_title+=split_criteria
if isinstance(split_criteria, list):
    dataset_title+="_"
    for s in split_criteria:
        dataset_title+="_"
        dataset_title+=s

#####################################################################
dataset_title = "single_" + dataset_title
#####################################################################

if not os.path.exists(f"./outputs/{dataset_title}/{datetime_string}/"):
    os.makedirs(f"./outputs/{dataset_title}/{datetime_string}/")

with open(f"./outputs/{dataset_title}/{datetime_string}/info.txt", 'w') as f:
    print("========================", file=f)
    print(vgae_model, file=f)

wandb.init(
    # Set the project where this run will be logged
    project="VGAE_Graph_Generation_V",
    group=f"{dataset_title}",

    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"{dataset_title}/{datetime_string}",
    dir=f"./outputs/{dataset_title}/{datetime_string}/",

    # Track hyperparameters and run metadata
    config={
        "device":device,
        "dataset_name":dataset_name,
        "split_criteria":split_criteria
    }
)

num_epochs = 1000

epoch=0
step_train_number=0
step_val_number = 0

while epoch<num_epochs:
    vgae_model.train()

    epoch_train_loss = 0
    for real in tqdm(train_loader):
        real = real.to(device)
        logits = vgae_model(real.x, real.edge_index)

        adj = (to_dense_adj(real.edge_index, max_num_nodes=real.num_nodes)[0]>0.0).float()
        loss = vgae_loss(logits, adj, vgae_model.mean, vgae_model.log_std, device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss+=loss.item()

        wandb.log({ "train_step/epoch":         epoch,
                    "train_step/step_number":   step_train_number,
                    "train_step/loss":          loss.item()})

        step_train_number+=1

    epoch_train_loss/=len(train_loader)
    wandb.log({ "train_epoch/epoch":    epoch,
                "train_epoch/loss":     epoch_train_loss})
    
    if (epoch+1)%(int(num_epochs/2.0))==0:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': vgae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 
            f"./outputs/{dataset_title}/{datetime_string}/vgae_model_{epoch}.pt")
    
    if epoch%5==0:
        vgae_model.eval()

        epoch_val_loss = 0
        for real in tqdm(val_loader):
            real = real.to(device)
            logits = vgae_model(real.x, real.edge_index)

            adj = (to_dense_adj(real.edge_index, max_num_nodes=real.num_nodes)[0]>0.0).float()
            loss = vgae_loss(logits, adj, vgae_model.mean, vgae_model.log_std, device)

            epoch_val_loss+=loss.item()

            wandb.log({ "val_step/epoch":       epoch,
                        "val_step/step_number": step_val_number,
                        "val_step/loss":        loss.item()})

            step_val_number+=1
    
    epoch_val_loss/=len(val_loader)
    wandb.log({ "val_epoch/epoch":  epoch,
                "val_epoch/loss":   epoch_val_loss})
    
    epoch+=1

torch.save(
    {
        'epoch': epoch,
        'model_state_dict': vgae_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 
    f"./outputs/{dataset_title}/{datetime_string}/vgae_model.pt")

vgae_model.eval()

test_loss = 0
for real in tqdm(test_loader):
    real = real.to(device)
    logits = vgae_model(real.x, real.edge_index)

    adj = (to_dense_adj(real.edge_index, max_num_nodes=real.num_nodes)[0]>0.0).float()
    loss = vgae_loss(logits, adj, vgae_model.mean, vgae_model.log_std, device)

    test_loss+=loss.item()
      
test_loss/=len(test_loader)
wandb.log({ "test/loss": test_loss})