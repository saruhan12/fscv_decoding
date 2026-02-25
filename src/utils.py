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
    print("Expert activation set shape: ", act_arr.shape)
    print("Labels(concentrations) shape: ", lbl_arr.shape)
    unique, counts = np.unique(lbl_arr[:, 0, :3], axis=0, return_counts=True)
    print("Duplicated conditions:")
    for u, c in zip(unique, counts):
        if c > 1:
            print(f"  DA={u[0]}, 5HT={u[1]}, NE={u[2]} — appears {c} times")

            
    lbl_arr = lbl_arr[:,:,:3]

    n_concentrations = act_arr.shape[0]

    indices = np.arange(n_concentrations)

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    X_train = act_arr[train_idx].reshape(-1, 1000, 80)
    y_train = lbl_arr[train_idx].reshape(-1, 3)

    X_test = act_arr[test_idx].reshape(-1, 1000, 80)
    y_test = lbl_arr[test_idx].reshape(-1, 3)

    X_val = act_arr[val_idx].reshape(-1, 1000, 80)
    y_val = lbl_arr[val_idx].reshape(-1, 3)
    
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)


    X_train_norm = (X_train - x_mean)/(x_std + 1e-8)
    X_test_norm = (X_test - x_mean)/(x_std + 1e-8)
    X_val_norm = (X_val - x_mean)/(x_std + 1e-8)

    print(f"y_train mean: {y_mean}; std:{y_std}")

    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    y_val_norm   = (y_val   - y_mean) / (y_std + 1e-8)  
    y_test_norm  = (y_test  - y_mean) / (y_std + 1e-8)
    
    data_train = ActivationData(X_train_norm, y_train_norm)
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    data_val = ActivationData(X_val_norm, y_val_norm)
    val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    data_test= ActivationData(X_test_norm, y_test_norm)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, y_mean, y_std