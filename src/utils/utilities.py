import yaml
import torch
import gc

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def get_config(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config