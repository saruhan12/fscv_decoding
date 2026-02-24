import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
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
        #Assuming Posix paths which were obtained from get_all_paths()
        act_arr.append(np.load(k / "weights.npy"))
        lbl_arr.append(np.load(k / "labels.npy").reshape(-1,150,4))
    return  np.concatenate(act_arr, axis=0),  np.concatenate(lbl_arr, axis=0)

def load_activation_data(paths, batch_size=8, shuffle=True, num_workers=0):
    act_arr, lbl_arr = get_activation_label_pair(paths)
    X_train, X_test, y_train, y_test = train_test_split(act_arr, lbl_arr, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    data_train = ActivationData(X_train, y_train)
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    data_val = ActivationData(X_val, y_val)
    val_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    data_test= ActivationData(X_test, y_test)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader