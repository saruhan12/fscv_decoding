import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


MONO_DOM_LIST = {0:'DA', 1:'5HT', 2:'NE', 3:'EQ'}

def get_all_paths(path_folder, collapsed=True, steps=None, volta=False):
    path = Path(path_folder)
    if steps:
        if volta:
            search = path.rglob("voltammograms.npy")
        else:
            search = path.rglob(f"weights_collapsed_{steps}steps.npy")if collapsed else path.rglob(f"weights_{steps}steps.npy")
    else:
        if volta:
            search = path.rglob("voltammograms.npy")
        else:
            search = path.rglob("weights_collapsed.npy")if collapsed else path.rglob("weights.npy")
    weight_paths = [p.parent for p in search]

    return weight_paths
    

class ActivationData(Dataset):
    def __init__(self, activation_p, labels_p, concentration_p):
        self.activation = activation_p
        self.label = labels_p
        self.concentration = concentration_p
    def __len__(self):
        return len(self.activation)
    
    def __getitem__(self, index):
        return torch.tensor(self.activation[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.float32), torch.tensor(self.concentration[index], dtype=torch.float32) 
    
def get_activation_label_pair(path_folder, collapsed = True, volta=False):
    act_arr = []
    lbl_arr = []
    
    paths = get_all_paths(path_folder=path_folder,collapsed=collapsed,volta=volta)

    if collapsed:
        print('Getting pairs collapsed.')
    else:
        print('Getting pairs not collapsed.')

    for k in paths:
        shape_get = np.load(k / "voltammograms.npy")
        N, T, sweep = shape_get.shape
        if volta:
            act = np.load(k / "voltammograms.npy")
            lbl = np.load(k / "labels.npy")
            if collapsed:
                act = act.mean(axis=2,keepdims=True)
                lbl = lbl.reshape(N,sweep,-1)[:,0,:]
            else:
                act = act.transpose(0,2,1).reshape(-1,1000,1)
                lbl = lbl.reshape(N*sweep,-1)
        else:

            act = np.load(k / "weights_collapsed.npy") if collapsed else np.load(k / "weights.npy")
            lbl = np.load(k / "labels.npy")
                
            # Handle both (N, sweeps, 1000, n_exp) and (N, 1000, n_exp)
            
            if act.ndim == 4:
                _, _, T, n_exp = act.shape
                act = act.reshape(N * sweep, T, n_exp)
                lbl = lbl.reshape(N*sweep,-1)
            elif act.ndim == 3:
                lbl = lbl.reshape(N,sweep,-1)[:,0,:]  # (N, 4)
            
        
        act_arr.append(act)
        lbl_arr.append(lbl)

    print(f'First activation shape{act_arr[0].shape}')
    print(f'First label shape{lbl_arr[0].shape}')

    return  np.concatenate(act_arr, axis=0),  np.concatenate(lbl_arr, axis=0)



def load_activation_data(path_folder, batch_size=8, shuffle=True, num_workers=0, collapsed=True,volta=False, ret_np = False):
    act_arr, lbl_arr = get_activation_label_pair(path_folder=path_folder, collapsed=collapsed,volta=volta)
    
    print("Expert activation set shape:", act_arr.shape)
    print("Labels shape:", lbl_arr.shape)
    
    # Drop pH, keep DA/5HT/NE 
    print("First label row (all 4 cols):", lbl_arr[0])

    lbl_arr = lbl_arr[:, :3]  # (N, 3)
    
        #check max
        #if at least 2 max then assign EQ(3)
    lbl_cat = []
    for k in lbl_arr:
        max = np.max(k)
        tie = np.sum(k==max) >=2
        lbl_cat.append(3 if tie else np.argmax(k))

    lbl_cat = np.array(lbl_cat)
        

    n_samples = act_arr.shape[0]
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.1, random_state=42)

    X_train, y_train, y_lbl_train = act_arr[train_idx], lbl_arr[train_idx], lbl_cat[train_idx] 
    X_val,   y_val, y_lbl_val   = act_arr[val_idx],   lbl_arr[val_idx], lbl_cat[val_idx]
    X_test,  y_test, y_lbl_test  = act_arr[test_idx],  lbl_arr[test_idx], lbl_cat[test_idx]

    # Normalize X
    x_mean = X_train.mean(axis=0)
    x_std  = X_train.std(axis=0)
    X_train_norm = (X_train - x_mean) / (x_std + 1e-8)
    X_val_norm   = (X_val   - x_mean) / (x_std + 1e-8)
    X_test_norm  = (X_test  - x_mean) / (x_std + 1e-8)

    # Normalize y
    
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0)
    print(f"y_mean: {y_mean}  y_std: {y_std}")
        
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    y_val_norm   = (y_val   - y_mean) / (y_std + 1e-8)
    y_test_norm  = (y_test  - y_mean) / (y_std + 1e-8)

    if ret_np:
        return (X_train_norm, y_lbl_train, y_train_norm), (X_val_norm, y_lbl_val, y_val_norm), (X_test_norm, y_lbl_test, y_test_norm), y_mean, y_std, MONO_DOM_LIST 
    else: 
        train_dataloader = DataLoader(ActivationData(X_train_norm, y_lbl_train, y_train_norm),
                                  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_dataloader   = DataLoader(ActivationData(X_val_norm, y_lbl_val,   y_val_norm),
                                        batch_size=batch_size, shuffle=False,   num_workers=num_workers)
        test_dataloader  = DataLoader(ActivationData(X_test_norm, y_lbl_test,  y_test_norm),
                                        batch_size=batch_size, shuffle=False,   num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader, y_mean, y_std, MONO_DOM_LIST
    