import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

MONO_DOM_LIST = {0:'DA', 1:'5HT', 2:'NE', 3:'EQ'}

def get_all_paths(path_folder, collapsed=True, steps=None, volta=False, model_dim='3d'):
    path = Path(path_folder)
    model_last = '_'+model_dim if model_dim else ''
    if steps:
        if volta:
            search = path.rglob("voltammograms.npy")
        else:
            search = path.rglob(f"weights_collapsed_{steps}steps{model_last}.npy")if collapsed else path.rglob(f"weights_{steps}steps{model_last}.npy")
    else:
        if volta:
            search = path.rglob("voltammograms.npy")
        else:
            search = path.rglob(f"weights_collapsed{model_last}.npy")if collapsed else path.rglob(f"weights{model_last}.npy")
    weight_paths = [p.parent for p in search]
    
    print(f"Found {len(weight_paths)} paths")
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
    
def get_activation_label_pair(path_folder, collapsed = True, volta=False,model_dim='3d'):
    act_arr, lbl_arr, probe_arr = [], [], []
    model_last = '_'+model_dim if model_dim else ''
    paths = get_all_paths(path_folder=path_folder,collapsed=collapsed,volta=volta,model_dim=model_dim)

    if collapsed:
        print('Getting pairs collapsed.')
    else:
        print('Getting pairs not collapsed.')

    if volta:
        print('Getting raw voltammograms')

    for k in paths:
        shape_get = np.load(k / "voltammograms.npy")
        N, T, sweep = shape_get.shape

        parts = Path(k).parts
        root_idx = next(i for i, p in enumerate(parts) if 'data_1d_vxlbl' in p)
        experiment = parts[root_idx + 1]  # e.g. AFOR, BFvsALC, ALCIS_155

        import re
        session_folder = str(k)
        match = re.search(r'(ALC\w*|BFA\w*)_INM\d+_\w+', session_folder)
        probe_name = match.group(0) if match else str(k)

        probe_id = f"{experiment}__{probe_name}"

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

            act = np.load(k / f"weights_collapsed{model_last}.npy") if collapsed else np.load(k / f"weights{model_last}.npy")
            lbl = np.load(k / "labels.npy")
                
            # Handle both (N, sweeps, 1000, n_exp) and (N, 1000, n_exp)
            
            if act.ndim == 4:
                _, _, T, n_exp = act.shape
                act = act.reshape(N * sweep, T, n_exp)
                lbl = lbl.reshape(N*sweep,-1)
            elif act.ndim == 3:
                lbl = lbl.reshape(N,sweep,-1)[:,0,:]  # (N, 4)
            
        n=len(act)
        probe_arr.append(np.full(n,probe_id))
        act_arr.append(act)
        lbl_arr.append(lbl)

    print(f'First activation shape{act_arr[0].shape}')
    print(f'First label shape{lbl_arr[0].shape}')

    return  np.concatenate(act_arr, axis=0),  np.concatenate(lbl_arr, axis=0), np.concatenate(probe_arr)



def load_activation_data(path_folder, test_probe=None, batch_size=8, shuffle=True, num_workers=0, collapsed=True,volta=False, ret_np = False, classification=True,model_dim='3d'):
    #Given a directory that contains the activations, it returns the time series(either expert activations or raw voltammograms)
    # labels for each(concentration and dominant monoamine), in train/test/val split.
    act_arr, lbl_arr, probe_arr = get_activation_label_pair(path_folder=path_folder, collapsed=collapsed,volta=volta,model_dim=model_dim)
    
    print("Expert activation set shape:", act_arr.shape)
    print("Labels shape:", lbl_arr.shape)
    
    # Drop pH, keep DA/5HT/NE 
    #print("First label row (all 4 cols):", lbl_arr[0])

    
    
        #check max
        #if at least 2 max then assign EQ(3)
    probe_valid_counts = {}
    for p, k in zip(probe_arr, lbl_arr):
        max_val = np.max(k)
        tie = np.sum(k == max_val) >= 2
        is_valid = not (tie or np.argmax(k) == 3)
        probe_valid_counts[p] = probe_valid_counts.get(p, 0) + (1 if is_valid else 0)

    probes_to_keep = np.array([p for p, count in probe_valid_counts.items() if count > 0])
    removed_probes = [p for p, count in probe_valid_counts.items() if count == 0]
    print(f"Removed probes (all EQ/pH): {removed_probes}")

    # Step 2: remove empty probes from all arrays
    probe_keep_mask = np.isin(probe_arr, probes_to_keep)
    act_arr   = act_arr[probe_keep_mask]
    lbl_arr   = lbl_arr[probe_keep_mask]
    probe_arr = probe_arr[probe_keep_mask]

    # Step 3: apply valid_mask (remove EQ and pH samples)
    lbl_cat = []
    valid_mask = []
    for k in lbl_arr:
        max_val = np.max(k)
        tie = np.sum(k == max_val) >= 2
        if tie or np.argmax(k) == 3:
            valid_mask.append(False if classification else True)
            lbl_cat.append(-1)
        else:
            lbl_cat.append(np.argmax(k))
            valid_mask.append(True)

    valid_mask = np.array(valid_mask)
    lbl_cat = np.array(lbl_cat)

    act_arr   = act_arr[valid_mask]
    lbl_arr   = lbl_arr[valid_mask]
    lbl_cat   = lbl_cat[valid_mask]
    probe_arr = probe_arr[valid_mask]

    # Add this after valid_mask is applied, for debugging
    print("\nPer-probe sample counts after filtering:")
    for p in np.unique(probe_arr):
        n = np.sum(probe_arr == p)
        print(f"  {p}: {n} samples")
    print(f"Total valid samples: {len(probe_arr)}")

    # Step 4: probe selection
    unique_probes = np.unique(probe_arr)
    print(f"Available probe IDs:\n{unique_probes}")

    if test_probe is None:
        rng = np.random.default_rng(42)
        test_probe = rng.choice(unique_probes)
        print(f"No test_probe specified — randomly selected: {test_probe}")
    else:
        assert test_probe in unique_probes, \
            f"test_probe '{test_probe}' not found. Available: {unique_probes}"

    print(f"Test probe:  {test_probe}")
    #print(f"Train probes: {[p for p in unique_probes if p != test_probe]}")
    
    train_mask = probe_arr != test_probe
    test_mask  = probe_arr == test_probe



    X_train, y_train, y_lbl_train = act_arr[train_mask], lbl_arr[train_mask], lbl_cat[train_mask]
    X_test,  y_test,  y_lbl_test  = act_arr[test_mask],  lbl_arr[test_mask],  lbl_cat[test_mask]

    # Normalize X (fit on train only)
    x_mean = X_train.mean(axis=0)
    x_std  = X_train.std(axis=0)
    X_train_norm = (X_train - x_mean) / (x_std + 1e-8)
    X_test_norm  = (X_test  - x_mean) / (x_std + 1e-8)

    # Normalize y (fit on train only)
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0)
    print(f"y_mean: {y_mean}  y_std: {y_std}")

    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    y_test_norm  = (y_test  - y_mean) / (y_std + 1e-8)

    if ret_np:
        return (X_train_norm, y_lbl_train, y_train_norm), \
               (X_test_norm,  y_lbl_test,  y_test_norm), \
               y_mean, y_std, MONO_DOM_LIST, test_probe
    else:
        train_dl = DataLoader(ActivationData(X_train_norm, y_lbl_train, y_train_norm),
                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dl  = DataLoader(ActivationData(X_test_norm,  y_lbl_test,  y_test_norm),
                              batch_size=batch_size, shuffle=False,   num_workers=num_workers)
        return train_dl, test_dl, y_mean, y_std, MONO_DOM_LIST, test_probe      
    