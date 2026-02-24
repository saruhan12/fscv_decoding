import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

def get_all_paths(path):
    path = Path(path)
    weight_paths = [p.parent for p in path.rglob("weights.npy")]

    return weight_paths
    

class ActivationData(Dataset):
    def __init__(self, activation_p, labels_p):
        self.activation = activation_p
        self.label = labels_p
    def __len__(self):
        return len(self.activation)
    
    def __getitem__(self, index):
        return torch.tensor(self.activation[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.float32)
    
def get_activation_label_pair(paths):
    act_arr = []
    lbl_arr = []
    for k in paths:
        act_arr.append(np.load(k+"weights.npy"))
        lbl_arr.append(np.load(k+"labels.npy").reshape(-1,150,4))
    return  np.concatenate(act_arr, axis=0),  np.concatenate(lbl_arr, axis=0)

def load_activation_data(paths, batch_size=8, shuffle=True, num_workers=0):
    act_arr, lbl_arr = get_activation_label_pair(paths)

    data = ActivationData(act_arr, lbl_arr)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader