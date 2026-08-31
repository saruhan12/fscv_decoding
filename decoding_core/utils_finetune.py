
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

NEUROMODUL_LIST = {0: 'DA', 1: '5HT', 2: 'NE', 3: 'pH'}


# --------------------------------------------------------------------------
# path discovery
# --------------------------------------------------------------------------
def get_volta_paths(path_folder):
    """Find every folder containing a voltammograms.npy."""
    path = Path(path_folder)
    paths = [p.parent for p in path.rglob("voltammograms.npy")]
    print(f"Found {len(paths)} voltammogram folders")
    return paths


def _probe_id_from_path(folder):
    """{experiment}__{probe_name} — keeps duplicate probe folders distinct."""
    parts = Path(folder).parts
    root_idx = next(i for i, p in enumerate(parts) if 'data_1d_vxlbl' in p)
    experiment = parts[root_idx + 1]
    match = re.search(r'(ALC\w*|BFA\w*)_INM\d+_\w+', str(folder))
    probe_name = match.group(0) if match else str(folder)
    return f"{experiment}__{probe_name}"


# --------------------------------------------------------------------------
# raw loading: returns per-sweep voltammograms, concentration labels, probe ids
# --------------------------------------------------------------------------
def load_volta_label_pairs(path_folder):
    """
    Returns:
      volta:  (n_samples, T, 1)        raw voltammogram per sweep
      labels: (n_samples, n_neuromodul) concentration vector per sweep
      probes: (n_samples,)              probe id per sweep
    """
    volta_arr, lbl_arr, probe_arr = [], [], []

    for folder in get_volta_paths(path_folder):
        volta = np.load(folder / "voltammograms.npy")   # (N, T, sweep)
        lbl = np.load(folder / "labels.npy")
        N, T, sweep = volta.shape

        # per-sweep: (N, T, sweep) -> (N*sweep, T, 1)
        volta = volta.transpose(0, 2, 1).reshape(-1, T, 1)
        lbl = lbl.reshape(N * sweep, -1)

        probe_id = _probe_id_from_path(folder)
        probe_arr.append(np.full(len(volta), probe_id))
        volta_arr.append(volta)
        lbl_arr.append(lbl)

    volta = np.concatenate(volta_arr, axis=0)
    labels = np.concatenate(lbl_arr, axis=0)
    probes = np.concatenate(probe_arr, axis=0)

    print(f"Total sweeps: {len(volta)}  | voltammogram shape: {volta.shape[1:]}")
    return volta, labels, probes


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class VoltaRegressionData(Dataset):
    """Raw voltammogram -> z-scored concentration target."""
    def __init__(self, volta, concentration):
        self.volta = volta              # (n, T, 1)
        self.conc = concentration       # (n, n_neuromodul)

    def __len__(self):
        return len(self.volta)

    def __getitem__(self, i):
        v = torch.tensor(self.volta[i], dtype=torch.float32)   # (T, 1)
        c = torch.tensor(self.conc[i], dtype=torch.float32)    # (n_neuromodul,)
        return v, c


def collate_seq_first(batch):
    """Stack to seq-first context expected by DynaMix: (T, batch, feature_dim)."""
    contexts = torch.stack([b[0] for b in batch], dim=0)   # (B, T, 1)
    targets = torch.stack([b[1] for b in batch], dim=0)    # (B, n_neuromodul)
    contexts = contexts.permute(1, 0, 2).contiguous()      # (T, B, 1)
    return contexts, targets


# --------------------------------------------------------------------------
# main entry: probe-based split + loaders
# --------------------------------------------------------------------------
def load_finetuning_data(path_folder,
                         test_probe=None,
                         probe_id=None,
                         batch_size=8,
                         num_workers=0,
                         shuffle=True,
                         ret_np=False):
    """
    Two modes:
      - probe_id set : single-probe within-probe 80/20 random split.
      - probe_id None: LOPO. test_probe -> test set, all others -> train.
                       If test_probe is None, one is chosen at random (seed 42).

    Concentration targets are z-scored using train-set statistics only.
    Returns y_mean / y_std so predictions can be mapped back to nM.
    """
    volta, labels, probes = load_volta_label_pairs(path_folder)
    unique_probes = np.unique(probes)
    print(f"Available probes ({len(unique_probes)}): {unique_probes}")

    # ---- split ----
    if probe_id is not None:
        assert probe_id in unique_probes, \
            f"probe_id '{probe_id}' not found. Available: {unique_probes}"
        print(f"[Single-probe] probe_id = {probe_id}")
        mask = probes == probe_id
        Xp, yp = volta[mask], labels[mask]
        idx_tr, idx_te = train_test_split(
            np.arange(len(Xp)), test_size=0.2, random_state=42)
        X_train, X_test = Xp[idx_tr], Xp[idx_te]
        y_train, y_test = yp[idx_tr], yp[idx_te]
        active_test_probe = probe_id
    else:
        if test_probe is None:
            test_probe = np.random.default_rng(42).choice(unique_probes)
            print(f"No test_probe given — randomly selected: {test_probe}")
        assert test_probe in unique_probes, \
            f"test_probe '{test_probe}' not found. Available: {unique_probes}"
        print(f"[LOPO] test probe = {test_probe}")
        tr_mask = probes != test_probe
        te_mask = probes == test_probe
        X_train, y_train = volta[tr_mask], labels[tr_mask]
        X_test, y_test = volta[te_mask], labels[te_mask]
        active_test_probe = test_probe

    print(f"Train sweeps: {len(X_train)}  | Test sweeps: {len(X_test)}")

    # ---- z-score concentration targets (fit on train only) ----
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    print(f"y_mean: {y_mean}  y_std: {y_std}")
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    y_test_norm = (y_test - y_mean) / (y_std + 1e-8)

    # NOTE: voltammograms X are NOT normalized here — DynaMix's DataPreprocessor
    # handles standardization/embedding internally as the pretrained gating expects.

    if ret_np:
        return (X_train, y_train_norm), (X_test, y_test_norm), \
               y_mean, y_std, NEUROMODUL_LIST, active_test_probe

    train_dl = DataLoader(VoltaRegressionData(X_train, y_train_norm),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=collate_seq_first)
    test_dl = DataLoader(VoltaRegressionData(X_test, y_test_norm),
                         batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_seq_first)

    return train_dl, test_dl, y_mean, y_std, NEUROMODUL_LIST, active_test_probe