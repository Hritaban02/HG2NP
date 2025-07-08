__HOME_DIR__ = "/home/du4/19CS30053/MTP2"
__MAX_NODES__ = 200

import sys
sys.path.append(f"{__HOME_DIR__}/Model/src")

from src.clean_up import ClearCache

from src.lib import *
from src.utils import preprocess_generated_samples_text_file, get_graph_metadata, validate_split_criteria
from src.model import HETModel, Discriminator
from src.dataset import Heterogeneous_Graph_Dataset


wandb.login()
datetime_string = datetime.now().strftime("%Y_%m_%d___%H_%M_%S")

# datetime_string = datetime_string + "---"

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--split_criteria', type=str, required=True)
parser.add_argument('--configuration_number', type=str, required=True)
parser.add_argument('--generated_samples_file', type=str, required=True)
parser.add_argument('--optimizer', type=str, required=True)

args = parser.parse_args()

device = args.device
dataset_name = args.dataset_name[1:-1]                      # Remove quotations
split_criteria = (args.split_criteria[1:-1]).split(',')     # Remove quotations and make list
if len(split_criteria)==1:
    split_criteria=split_criteria[0]

configuration_number = args.configuration_number
generated_samples_file = args.generated_samples_file[1:-1]  # Remove quotations
optimizer = args.optimizer

# Sanity Check
if dataset_name not in ["DBLP", "IMDB"]:
  sys.exit(f"ERROR: dataset_name {dataset_name} is not available")

if not validate_split_criteria(dataset_name=dataset_name, split_criteria=split_criteria):
    sys.exit(f"ERROR: split_criteria {split_criteria} is not valid")

config_file = f"./config/train_config_{configuration_number}.json"
if not os.path.isfile(config_file):
    sys.exit(f"ERROR: config_file {config_file} is not available")

if not os.path.isfile(generated_samples_file):
    sys.exit(f"ERROR: generated_samples_file {generated_samples_file} is not available")

if optimizer not in ["Adam", "AdamW", "NAdam", "RAdam"]:
    sys.exit(f"ERROR: optimizer {optimizer} is not available")

# Set Device
device = (device if torch.cuda.is_available() else "cpu")
print(f"Device being used : {device}")

dataset_title = dataset_name
if isinstance(split_criteria, str):
    dataset_title+="_"
    dataset_title+=split_criteria
if isinstance(split_criteria, list):
    dataset_title+="_"
    for s in split_criteria:
        dataset_title+="_"
        dataset_title+=s

# Load and Preprocess config file
cfg = json.load(open(config_file))

# Generated Graphs
# Load Generated Data List

# For DiGress
gen_data_list = preprocess_generated_samples_text_file(path=generated_samples_file, dataset=dataset_name)

# For VGAE
# VGAE_Generated_Graphs = {
#     "IMDB_year"                             :"/home/du4/19CS30053/MTP2/VGAE/outputs/IMDB_year/2024_04_11___14_45_39/fake_hom_graphs.pt",
#     "IMDB__country_year"                    :"/home/du4/19CS30053/MTP2/VGAE/outputs/IMDB__country_year/2024_04_11___14_43_44/fake_hom_graphs.pt",
#     "IMDB__country_language_movie_year"     :"/home/du4/19CS30053/MTP2/VGAE/outputs/IMDB__country_language_movie_year/2024_04_11___14_36_24/fake_hom_graphs.pt",
#     "IMDB__country_language_movie"          :"/home/du4/19CS30053/MTP2/VGAE/outputs/IMDB__country_language_movie/2024_04_11___14_37_43/fake_hom_graphs.pt",
#     "DBLP__author_conference_type"          :"/home/du4/19CS30053/MTP2/VGAE/outputs/DBLP__author_conference_type/2024_04_11___14_53_17/fake_hom_graphs.pt",
#     "DBLP__author_conference"               :"/home/du4/19CS30053/MTP2/VGAE/outputs/DBLP__author_conference/2024_04_11___14_51_52/fake_hom_graphs.pt",
# }
# gen_data_list_ = torch.load(VGAE_Generated_Graphs[dataset_title])
# gen_data_list = []
# for g in gen_data_list_:
#     if g.num_nodes <= __MAX_NODES__:
#         gen_data_list.append(g)

train_gen_data_list, test_gen_data_list = random_split(dataset=gen_data_list, lengths=[0.8, 0.2])
train_gen_data_list, val_gen_data_list = random_split(dataset=train_gen_data_list, lengths=[0.9, 0.1])

train_gen_loader = DataLoader(train_gen_data_list, batch_size=cfg["batch_size"], shuffle=True)
val_gen_loader = DataLoader(val_gen_data_list, batch_size=cfg["batch_size"], shuffle=True)
test_gen_loader = DataLoader(test_gen_data_list, batch_size=cfg["batch_size"], shuffle=True)

# Real Graphs
real_data_list = Heterogeneous_Graph_Dataset(dataset_name=dataset_name, split_criteria=split_criteria).split_data
new_real_data_list = []
for graph in real_data_list:
    if graph.num_nodes <= __MAX_NODES__:
        new_real_data_list.append(graph)
real_loader = DataLoader(new_real_data_list, batch_size=cfg["batch_size"], shuffle=True)

cfg["common"]["device"] = device

cfg["generator"]["dataset_name"]=dataset_name
cfg["generator"]["split_criteria"]=split_criteria
cfg["generator"]["mpnn"]["edge_types"]=1
cfg["generator"].update(cfg["common"])
cfg["generator"]["mpnn"].update(cfg["common"])
cfg["generator"]["sampler"].update(cfg["common"])

cfg["discriminator"]["mpnn"]["metadata"]=get_graph_metadata(dataset_name=dataset_name, split_criteria=split_criteria)
cfg["discriminator"].update(cfg["common"])
cfg["discriminator"]["mpnn"].update(cfg["common"])
cfg["discriminator"]["linear"].update(cfg["common"])

generator = HETModel(cfg["generator"])
discriminator = Discriminator(cfg["discriminator"])

# Use Binary Cross Entropy Loss as the Loss Criterion
criterion = nn.BCELoss()

generator.to(device)
discriminator.to(device)

if optimizer=="Adam":
    optimizer_for_generator = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer_for_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
elif optimizer=="AdamW":
    optimizer_for_generator = torch.optim.AdamW(generator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=True,)
    optimizer_for_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=True,)
elif optimizer=="NAdam":
    optimizer_for_generator = torch.optim.NAdam(generator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer_for_discriminator = torch.optim.NAdam(discriminator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
elif optimizer=="RAdam":
    optimizer_for_generator = torch.optim.RAdam(generator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer_for_discriminator = torch.optim.RAdam(discriminator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
else:
    sys.exit(f"ERROR: optimizer {optimizer} is not available")

if not os.path.exists(f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/"):
    os.makedirs(f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/")

with open(f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/info.txt", 'w') as f:
    pprint.pprint(cfg, stream=f)
    print("========================", file=f)
    print(generator, file=f)
    print("========================", file=f)
    print(discriminator, file=f)

wandb.init(
    # Set the project where this run will be logged
    project="Heterogeneous_Graph_Generation_V",
    group=f"{optimizer}_{dataset_title}",

    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"{dataset_title}/{datetime_string}",
    dir=f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/",

    # Track hyperparameters and run metadata
    config={
        **cfg,
        "device":device,
        "config_file":config_file,
        "dataset_name":dataset_name,
        "split_criteria":split_criteria,
        "generated_samples_file":generated_samples_file
    }
)

epoch=0
step_real_graph_number=0
step_generated_graph_number=0
step_val_number = 0

while epoch<cfg["num_epochs"]:
    generator.train()
    discriminator.train()

    epoch_loss_of_D_on_fake=0
    epoch_loss_of_D_on_real=0
    epoch_loss_of_G=0

    for real in tqdm(real_loader):
        with ClearCache():
            # Pass a real graph to the discriminator
            real = real.to(device)
            y = discriminator(real.x_dict, real.edge_index_dict, real.batch_dict)
            loss_D_real = criterion(y, torch.ones(y.shape[0], 1).to(device))    # Discriminator should declare 1 for real graphs.
            
            optimizer_for_discriminator.zero_grad()
            loss_D_real.backward()
            optimizer_for_discriminator.step()
            epoch_loss_of_D_on_real+=loss_D_real.item()
            
            wandb.log({ "train_step/epoch":       epoch,
                        "train_step/step_number": step_real_graph_number,
                        "train_step/loss_D_real": loss_D_real.item()})

            step_real_graph_number+=1

    for category_map in tqdm(train_gen_loader):
        with ClearCache():
            # Pass a fake graph to the discriminator
            category_map = category_map.to(device)
            fake = generator(category_map).to(device)
            y = discriminator(fake.x_dict, fake.edge_index_dict, fake.batch_dict)
            loss_D_fake = criterion(y, torch.zeros(y.shape[0], 1).to(device))   # Discriminator should declare 0 for fake graphs.
        
            optimizer_for_discriminator.zero_grad()
            loss_D_fake.backward(retain_graph=True)
            optimizer_for_discriminator.step()
            epoch_loss_of_D_on_fake+=loss_D_fake.item()

            # Pass a ground truth graph to the discriminator
            y = discriminator(fake.x_dict, fake.edge_index_dict, fake.batch_dict)
            loss_G = criterion(y, torch.ones(y.shape[0], 1).to(device))         # According to the generator, discriminator should declare 1 for real graphs.

            optimizer_for_generator.zero_grad()
            loss_G.backward()
            optimizer_for_generator.step()
            epoch_loss_of_G+=loss_G.item()

            wandb.log({ "train_step/epoch":       epoch,
                        "train_step/step_number": step_generated_graph_number,
                        "train_step/loss_D_fake": loss_D_fake.item(),
                        "train_step/loss_G":      loss_G.item()})
            step_generated_graph_number+=1

    epoch_loss_of_D_on_fake/=len(train_gen_loader)
    epoch_loss_of_D_on_real/=len(real_loader)
    epoch_loss_of_G/=len(train_gen_loader)

    wandb.log({ "train_epoch/epoch":       epoch,
                "train_epoch/loss_D_fake": epoch_loss_of_D_on_fake,
                "train_epoch/loss_D_real": epoch_loss_of_D_on_real,
                "train_epoch/loss_G":      epoch_loss_of_G})

    if (epoch+1)%(int(cfg["num_epochs"]/2.0))==0:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_for_discriminator.state_dict(),
            }, 
            f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/discriminator_{epoch}.pt")
    
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_for_generator.state_dict(),
            }, 
            f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/generator_{epoch}.pt")
        
    if epoch%5==0:
        generator.eval()
        discriminator.eval()

        epoch_val_loss_of_D_on_fake=0
        for category_map in tqdm(val_gen_loader):
            with ClearCache():
                # Pass a fake graph to the discriminator
                category_map = category_map.to(device)
                fake = generator(category_map).to(device)
                y = discriminator(fake.x_dict, fake.edge_index_dict, fake.batch_dict)
                loss_D_fake = criterion(y, torch.zeros(y.shape[0], 1).to(device))

                epoch_val_loss_of_D_on_fake+=loss_D_fake.item()

                wandb.log({ "val_step/epoch":       epoch,
                            "val_step/step_number": step_val_number,
                            "val_step/loss_D_fake": loss_D_fake.item()})

                step_val_number+=1

        epoch_val_loss_of_D_on_fake/=len(val_gen_loader)
        wandb.log({ "val_epoch/epoch":       epoch,
                    "val_epoch/loss_D_fake": epoch_val_loss_of_D_on_fake})

    epoch+=1

torch.save(
    {
        'epoch': epoch,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer_for_discriminator.state_dict(),
    }, 
    f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/discriminator.pt")

torch.save(
    {
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_for_generator.state_dict(),
    }, 
    f"./outputs/{optimizer}/{dataset_title}/{datetime_string}/generator.pt")

generator.eval()
discriminator.eval()

test_loss_of_D_on_fake=0
for category_map in tqdm(test_gen_loader):
    with ClearCache():
        # Pass a fake graph to the discriminator
        category_map = category_map.to(device)
        fake = generator(category_map).to(device)
        y = discriminator(fake.x_dict, fake.edge_index_dict, fake.batch_dict)
        loss_D_fake = criterion(y, torch.zeros(y.shape[0], 1).to(device))

        test_loss_of_D_on_fake+=loss_D_fake.item()

test_loss_of_D_on_fake/=len(test_gen_loader)
wandb.log({ "test/loss_D_fake": test_loss_of_D_on_fake})
    