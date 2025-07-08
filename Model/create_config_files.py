from src.lib import *

# Custom Produce Train Config Files
config_file = f"./config/templates/train_config_C.json"
cfg = json.load(open(config_file))

i = 5
# for negative_slope in [0.01, 0.1, 0.5, 0.9]:
#     cfg["common"]["negative_slope"] = negative_slope

#     for wg in [1.0e-12, 1.0e-6, 1.0e-1]:
#         cfg["weight_decay"] = wg

for lr in [0.02, 0.0002]:
    cfg["lr"] = lr
    
    with open(f"./config/train_config_{i}.json", 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    
    # print(f"SUCCESS: Created Configuration File with learning rate {lr}, negative slope {negative_slope}, and weight decay {wg}", flush=True)
    i+=1
