import torch
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.stattools import acf
from scipy import stats
from scipy.signal import find_peaks

from .preprocessing import estimate_pos_tau,estimate_tdm_tau,snap_tau_to_period,make_phis,apply_embedding
from .preprocessing import make_sines as _make_sines_TB
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

MONO_DOM_LIST = {0:'DA', 1:'5HT', 2:'NE', 3:'EQ'}

MODEL_LABELS = ["fine_tune",
                "dyninception_mix",
                "3d",
                "6d"]



# from .your_data_module import get_activation_label_pair   # already in scope


# --------------------------------------------------------------------------- #
# backwards-compatible thin wrappers
# --------------------------------------------------------------------------- #
def estimate_tau(data_np, max_lag=None, min_lag=None):
    """
    tau := argmax_{tau > min_lag} ACF(tau).

    Kept for backwards compatibility. Accepts (T, N) or (T, B, N); the original
    version crashed on an undefined `data` and only handled (T, N).
    """
    return estimate_pos_tau(data_np, min_lag=min_lag, max_lag=max_lag)


def make_sines(T, B, tau, phis, dtype=np.float32, t0=0):
    """
    sin(2*pi*t/tau + phi_i) broadcast over the batch -> (T, B, len(phis)).

    Returns ONLY the sine channels (the caller concatenates), which is what the
    `embed()` call site always assumed. The old version returned data+sines and
    had a different signature, so the two disagreed.
    """
    return _make_sines_TB(T, B, tau, phis, dtype=dtype, t0=t0)

"""
Patched `spit_compliant_data`: the delay-embedding trim is absorbed by tiling
one extra sweep and cropping back, so the saved arrays keep round lengths
(sweep_len / rep*sweep_len / (rep+1)*sweep_len) regardless of tau.

Replaces the version in spit_compliant_data_fixed.py. Imports are unchanged.
"""



def _crop_tail(A, target_len, name=""):
    """Keep the LAST target_len timesteps. The signal is periodic with the
    sweep length, so this lands on a sweep boundary."""
    T = A.shape[0]
    if T == target_len:
        return A
    assert T >= target_len, f"{name}: have {T} steps, need {target_len}"
    return A[T - target_len:]


def spit_compliant_data(path_folder, out_path, probe_id_substr, volta=True,
                        test_frac=0.2, rep=9, seed=42, model_last='3d',
                        n_dim=3, sweep_len=1000,
                        embedding="pos_embedding", tau=None, phi_mode="random",
                        tau_min=None, tau_max=None, snap_tau=True,
                        save_labels=True, keep_round_lengths=True, verbose=True):
    """
    Within-probe, leave-MIXTURE-out split with a consistent embedding across
    context / train / test.

    keep_round_lengths : if True (default), an embedding that consumes history
        (delay embeddings) is given extra tiled sweeps as burn-in and the result
        is cropped back, so data/context/test keep lengths
        sweep_len / rep*sweep_len / (rep+1)*sweep_len. Set False to get the old
        behaviour (arrays shortened by n_extra*tau).
    """
    acts, lbls, probes, paths = get_activation_label_pair(  # noqa: F821
        path_folder, collapsed=False, volta=volta, model_last=model_last)

    matches = [p for p in np.unique(probes) if probe_id_substr in p]
    assert len(matches) == 1, f"probe match not unique: {matches}"
    m = probes == matches[0]
    X, Y = acts[m], lbls[m]
    print(f"{matches[0]}: {len(X)} sweeps")

    # --- group rows by mixture combo ---
    Yr = np.round(Y, 3)
    _, combo_id = np.unique(Yr, axis=0, return_inverse=True)
    combos = np.unique(combo_id)
    print(f"{len(combos)} unique mixtures, ~{len(X) / len(combos):.0f} sweeps each")

    # --- split the COMBOS, not the rows ---
    rng = np.random.default_rng(seed)
    perm = rng.permutation(combos)
    n_test = int(len(perm) * test_frac)
    test_c = set(perm[:n_test].tolist())
    train_c = set(perm[n_test:].tolist())
    assert not (test_c & train_c)

    tr = np.array([c in train_c for c in combo_id])
    te = np.array([c in test_c for c in combo_id])
    print(f"combos -> train {len(train_c)}  test {len(test_c)}")
    print(f"rows   -> train {tr.sum()}  test {te.sum()}")

    # --- reshape to (T, B, 1): ONE sweep, not yet tiled ---
    D = np.swapaxes(X[tr].astype(np.float32).reshape(-1, sweep_len, 1), 0, 1)
    E = np.swapaxes(X[te].astype(np.float32).reshape(-1, sweep_len, 1), 0, 1)

    # --- standardize on TRAIN only, BEFORE embedding ---
    mu  = D.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(D.std(axis=(0, 1), keepdims=True), 1e-6)
    D = (D - mu) / std
    E = (E - mu) / std

    # --- embedding parameters, estimated ONCE on the untiled train sweeps ----
    n_extra = n_dim - D.shape[2]
    params = {"method": embedding}
    trim = 0

    if embedding == "pos_embedding":
        if tau is None:
            tau = estimate_pos_tau(
                D,
                min_lag=tau_min if tau_min is not None else max(2, sweep_len // 10),
                max_lag=tau_max, reduce="mean_acf", seed=seed)
        tau = int(tau)
        if snap_tau:
            snapped = snap_tau_to_period(tau, sweep_len)
            if verbose and snapped != tau:
                print(f"tau {tau} -> {snapped} (snapped to a divisor of {sweep_len})")
            tau = snapped
        phis = make_phis(n_extra, mode=phi_mode, rng=rng)
        params.update({"tau": tau, "phis": phis, "t0": 0})
        print(f"tau = {tau}   phis = {np.round(phis, 4)}")

    elif embedding == "delay_embedding":
        if tau is None:
            tau = estimate_tdm_tau(D, channel=-1, reduce="median", seed=seed)
        tau = int(tau)
        trim = n_extra * tau
        params.update({"tau": tau, "source_channel": -1})
        print(f"delay tau = {tau}  (needs {trim} steps of history)")

    elif embedding == "delay_embedding_random":
        params.update({"source_channel": 0})
        trim = sweep_len          # conservative burn-in; exact value set below

    # --- burn-in: extra tiled sweeps to absorb the trim ---------------------
    pad = int(np.ceil(trim / sweep_len)) if (trim and keep_round_lengths) else 0
    if pad and verbose:
        print(f"burn-in: +{pad} sweep(s) so lengths stay "
              f"{sweep_len}/{rep * sweep_len}/{(rep + 1) * sweep_len}")

    X_data    = np.tile(D, (1 + pad, 1, 1))
    X_context = np.tile(D, (rep + pad, 1, 1))
    X_test    = np.tile(E, (rep + 1 + pad, 1, 1))

    # --- apply the SAME embedding to every split ----------------------------
    X_data,    params = apply_embedding(X_data,    n_dim, embedding, params=params)
    X_context, _      = apply_embedding(X_context, n_dim, embedding, params=params)
    X_test,    _      = apply_embedding(X_test,    n_dim, embedding, params=params)

    # --- crop back to round lengths -----------------------------------------
    if pad:
        X_data    = _crop_tail(X_data,    sweep_len,              "data")
        X_context = _crop_tail(X_context, rep * sweep_len,        "context")
        X_test    = _crop_tail(X_test,    (rep + 1) * sweep_len,  "test")

    print(f"Context shape: {X_context.shape}")
    print(f"Data shape:    {X_data.shape}")
    print(f"Test shape:    {X_test.shape}")
    for name, A in [("data", X_data), ("context", X_context), ("test", X_test)]:
        assert np.isfinite(A).all(), f"{name} has non-finite values"
        assert A.shape[2] == n_dim, f"{name} has {A.shape[2]} channels, expected {n_dim}"
        print(f"  {name}: absmax {np.abs(A).max():.3f}  "
              f"per-channel std {np.round(A.std(axis=(0, 1), dtype=np.float64), 3)}")

    if keep_round_lengths:
        assert X_data.shape[0] == sweep_len
        assert X_context.shape[0] == rep * sweep_len
        assert X_test.shape[0] == (rep + 1) * sweep_len

    np.save(out_path + "data.npy", X_data)
    np.save(out_path + "context.npy", X_context)
    np.save(out_path + "test.npy", X_test)

    if save_labels:
        np.save(out_path + "labels_train.npy", Y[tr].astype(np.float32))
        np.save(out_path + "labels_test.npy",  Y[te].astype(np.float32))
        np.save(out_path + "combo_id_train.npy", combo_id[tr])
        np.save(out_path + "combo_id_test.npy",  combo_id[te])

    meta = {
        "probe": matches[0],
        "embedding": embedding,
        "tau": int(params["tau"]) if params.get("tau") is not None else None,
        "phis": np.asarray(params.get("phis", [])).tolist(),
        "trim": int(params.get("trim", 0)),
        "burn_in_sweeps": pad,
        "taus": [int(t) for t in params.get("taus", [])],
        "snap_tau": bool(snap_tau),
        "phi_mode": phi_mode,
        "mu": float(mu.squeeze()),
        "std": float(std.squeeze()),
        "rep": rep,
        "sweep_len": sweep_len,
        "n_dim": n_dim,
        "seed": seed,
        "test_frac": test_frac,
        "volta": volta,
        "model_name": model_last,
        "shapes": {"data": list(X_data.shape),
                   "context": list(X_context.shape),
                   "test": list(X_test.shape)},
    }
    with open(out_path + "embedding_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return X_data, X_context, X_test, meta


spit_complaint_data = spit_compliant_data

"""

def spit_compliant_data(path_folder, out_path, probe_id_substr, volta=True,
                        test_frac=0.2, rep=9, seed=42, model_last='3d',
                        n_dim=3, sweep_len=1000,
                        embedding="pos_embedding", tau=None, phi_mode="random",
                        tau_min=None, tau_max=None, snap_tau=True,
                        save_labels=True, verbose=True):
    
    Within-probe, leave-MIXTURE-out split. Whole concentration combos are
    assigned entirely to train / test, so identical replicate sweeps never
    cross splits.

    Channels 1..n_dim-1 carry the embedding (eq. 6 sines by default, or delay
    coordinates), estimated ONCE on the standardized training sweeps and then
    re-applied verbatim to context / train / test.

    Parameters
    ----------
    embedding : "pos_embedding" | "zero_embedding" | "delay_embedding"
                | "delay_embedding_random"
    tau       : fixed tau; if None it is estimated from the untiled train sweeps
    phi_mode  : "random" (draws once, saved to metadata) or "linspace"
    snap_tau  : snap tau to a divisor of sweep_len so the sine channel stays
                phase-coherent with the tiling (positional embedding only)
    
    acts, lbls, probes, paths = get_activation_label_pair(  # noqa: F821
        path_folder, collapsed=False, volta=volta, model_last=model_last)

    matches = [p for p in np.unique(probes) if probe_id_substr in p]
    assert len(matches) == 1, f"probe match not unique: {matches}"
    m = probes == matches[0]
    X, Y = acts[m], lbls[m]
    print(f"{matches[0]}: {len(X)} sweeps")

    # --- group rows by mixture combo (stable integer key) ---
    Yr = np.round(Y, 3)
    _, combo_id = np.unique(Yr, axis=0, return_inverse=True)
    combos = np.unique(combo_id)
    print(f"{len(combos)} unique mixtures, ~{len(X) / len(combos):.0f} sweeps each")

    # --- split the COMBOS, not the rows ---
    rng = np.random.default_rng(seed)
    perm = rng.permutation(combos)
    n_test = int(len(perm) * test_frac)
    test_c = set(perm[:n_test].tolist())
    train_c = set(perm[n_test:].tolist())
    assert not (test_c & train_c)

    tr = np.array([c in train_c for c in combo_id])
    te = np.array([c in test_c for c in combo_id])
    print(f"combos -> train {len(train_c)}  test {len(test_c)}")
    print(f"rows   -> train {tr.sum()}  test {te.sum()}")

    # --- reshape to (T, B, 1) ---
    X_data = np.swapaxes(X[tr].astype(np.float32).reshape(-1, sweep_len, 1), 0, 1)
    X_test = np.swapaxes(X[te].astype(np.float32).reshape(-1, sweep_len, 1), 0, 1)

    # tile the sweep along time
    X_context = np.tile(X_data, (rep, 1, 1))          # (rep*sweep_len, B_tr, 1)
    X_test    = np.tile(X_test, (rep + 1, 1, 1))      # ((rep+1)*sweep_len, B_te, 1)

    # --- standardize on TRAIN only, BEFORE embedding ---
    mu  = X_data.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(X_data.std(axis=(0, 1), keepdims=True), 1e-6)

    X_data    = (X_data    - mu) / std
    X_context = (X_context - mu) / std
    X_test    = (X_test    - mu) / std

    # --- embedding parameters: estimated ONCE, on the untiled train sweeps ----
    # Estimating on X_context would just recover the tiling period, since
    # np.tile makes the context exactly periodic with sweep_len.
    n_extra = n_dim - X_data.shape[2]
    params = {"method": embedding}

    if embedding == "pos_embedding":
        if tau is None:
            tau = estimate_pos_tau(
                X_data,
                min_lag=tau_min if tau_min is not None else max(2, sweep_len // 10),
                max_lag=tau_max,
                reduce="mean_acf",
                seed=seed,
            )
        tau = int(tau)
        if snap_tau:
            snapped = snap_tau_to_period(tau, sweep_len)
            if verbose and snapped != tau:
                print(f"tau {tau} -> {snapped} (snapped to a divisor of {sweep_len})")
            tau = snapped
        phis = make_phis(n_extra, mode=phi_mode, rng=rng)
        params.update({"tau": tau, "phis": phis, "t0": 0})
        print(f"tau = {tau}   phis = {np.round(phis, 4)}")

    elif embedding.startswith("delay_embedding"):
        if embedding == "delay_embedding":
            if tau is None:
                tau = estimate_tdm_tau(X_data, channel=-1, reduce="median", seed=seed)
            params.update({"tau": int(tau), "source_channel": -1})
            print(f"delay tau = {int(tau)}  (trim = {n_extra * int(tau)} steps)")
        else:
            params.update({"source_channel": 0})

    # --- apply the SAME embedding to every split ----------------------------
    X_data,    params = apply_embedding(X_data,    n_dim, embedding, params=params)
    X_context, _      = apply_embedding(X_context, n_dim, embedding, params=params)
    X_test,    _      = apply_embedding(X_test,    n_dim, embedding, params=params)

    print(f"Context shape: {X_context.shape}")
    print(f"Data shape:    {X_data.shape}")
    print(f"Test shape:    {X_test.shape}")
    for name, A in [("data", X_data), ("context", X_context), ("test", X_test)]:
        assert np.isfinite(A).all(), f"{name} has non-finite values"
        assert A.shape[2] == n_dim, f"{name} has {A.shape[2]} channels, expected {n_dim}"
        print(f"  {name}: absmax {np.abs(A).max():.3f}  "
              f"per-channel std {np.round(A.std(axis=(0, 1)), 3)}")

    # sanity check: data channel of the context must match the tiled train data
    reps_in_context = X_context.shape[0] // X_data.shape[0]
    if embedding == "pos_embedding" and reps_in_context >= 1:
        assert np.allclose(X_context[:X_data.shape[0], :, 0], X_data[:, :, 0]), \
            "context/data mismatch on the observed channel"

    np.save(out_path + "data.npy", X_data)
    np.save(out_path + "context.npy", X_context)
    np.save(out_path + "test.npy", X_test)

    # labels are per sweep (B), so the time-axis trim of the delay embedding
    # does not touch them; saving them keeps this split reusable downstream.
    if save_labels:
        np.save(out_path + "labels_train.npy", Y[tr].astype(np.float32))
        np.save(out_path + "labels_test.npy",  Y[te].astype(np.float32))
        np.save(out_path + "combo_id_train.npy", combo_id[tr])
        np.save(out_path + "combo_id_test.npy",  combo_id[te])

    meta = {
        "probe": matches[0],
        "embedding": embedding,
        "tau": int(params["tau"]) if params.get("tau") is not None else None,
        "phis": np.asarray(params.get("phis", [])).tolist(),
        "trim": int(params.get("trim", 0)),
        "taus": [int(t) for t in params.get("taus", [])],
        "snap_tau": bool(snap_tau),
        "phi_mode": phi_mode,
        "mu": float(mu.squeeze()),
        "std": float(std.squeeze()),
        "rep": rep,
        "sweep_len": sweep_len,
        "n_dim": n_dim,
        "seed": seed,
        "test_frac": test_frac,
        "volta": volta,
        "model_name": model_last,
        "shapes": {"data": list(X_data.shape),
                   "context": list(X_context.shape),
                   "test": list(X_test.shape)},
    }
    with open(out_path + "embedding_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return X_data, X_context, X_test, meta

"""



def get_all_paths(path_folder, collapsed, volta, model_last, combined):
    path = Path(path_folder)
    
    if combined:
        search = path.rglob(f"combined_{model_last}.npy") 
    else:
        if volta:
            search = path.rglob("voltammograms.npy")

        else:
            search = path.rglob(f"weights_collapsed_{model_last}.npy") if collapsed else path.rglob(f"weights_{model_last}.npy")
    weight_paths = [p.parent for p in search]
    assert weight_paths, (
        f"no files matched under {path} "
        f"(combined={combined}, volta={volta}, collapsed={collapsed}, model_last={model_last})")
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
    
def get_activation_label_pair(path_folder, collapsed, volta,model_last,combined):
    act_arr, lbl_arr, probe_arr = [], [], []

    paths = get_all_paths(path_folder=path_folder,collapsed=collapsed,volta=volta,model_last=model_last,combined=combined)


    if collapsed:
        print('Getting pairs collapsed.')
    else:
        print('Getting pairs not collapsed.')

    if volta:
        print('Getting raw voltammograms')
    else:
        print(f'Getting dynamix_{model_last} activations')

    for k in paths:
        N, T, sweep = np.load(k / "voltammograms.npy", mmap_mode='r').shape

        parts = Path(k).parts
        root_idx = next(i for i, p in enumerate(parts) if 'data_1d_vxlbl' in p)
        experiment = parts[root_idx + 1]  # e.g. AFOR, BFvsALC, ALCIS_155

        import re
        session_folder = str(k)
        match = re.search(r'(ALC\w*|BFA\w*)_INM\d+_\w+', session_folder)
        probe_name = match.group(0) if match else str(k)

        probe_id = f"{experiment}__{probe_name}"
            #print(f'First activation shape{act_arr[0].shape}')
            #print(f'First label shape{lbl_arr[0].shape}')

            
        
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
            if combined: 
                act = np.load(k / f"combined_{model_last}.npy") if collapsed else np.load(k / f"combined_{model_last}.npy")
                lbl = np.load(k / "labels.npy")
                                            
            else:
                act = np.load(k / f"weights_collapsed_{model_last}.npy") if collapsed else np.load(k / f"weights_{model_last}.npy")
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

            #print(f'First activation shape{act_arr[0].shape}')
            #print(f'First label shape{lbl_arr[0].shape}')
    
    '''if combined:
        return  act_arr,  lbl_arr, probe_arr, paths
    else:
        return  np.concatenate(act_arr, axis=0),  np.concatenate(lbl_arr, axis=0), np.concatenate(probe_arr), paths'''
    return (np.concatenate(act_arr, axis=0),
            np.concatenate(lbl_arr, axis=0),
            np.concatenate(probe_arr), paths)

def load_activation_data(path_folder=None, test_probe=None, probe_id=None,
                         preloaded=None,
                         batch_size=512, val_split=0.1, shuffle=True, num_workers=0,
                         collapsed=False, volta=False, ret_np=False,
                         classification=True, model_last='3d', std_floor=1e-3, combined = True):
    if preloaded is not None:
        act_arr, lbl_arr, probe_arr = preloaded

    else:
        act_arr, lbl_arr, probe_arr, _ = get_activation_label_pair(
            path_folder=path_folder, collapsed=collapsed, volta=volta, model_last=model_last,combined=combined)         

    print("Activation shape:", act_arr.shape)
    print("Labels shape:", lbl_arr.shape)

    # --- vectorized validity + categorical labels (col 3 = pH/blank -> EQ) ---
    max_val = lbl_arr.max(axis=1)
    tie     = (lbl_arr == max_val[:, None]).sum(axis=1) >= 2
    argmax  = lbl_arr.argmax(axis=1)
    is_eq_or_ph = tie | (argmax == 3)
    lbl_cat_full = np.where(is_eq_or_ph, -1, argmax).astype(np.int64)

    # regression keeps every row; classification drops EQ/pH-dominant rows
    row_valid = np.ones(len(lbl_arr), bool) if not classification else ~is_eq_or_ph

    removed_probes = sorted(set(np.unique(probe_arr)) - set(np.unique(probe_arr[row_valid])))
    if removed_probes:
        print(f"Removed probes (no valid samples): {removed_probes}")

    # single re-index, and only if something is actually dropped (skips a ~5 GB copy)
    if not row_valid.all():
        act_arr   = act_arr[row_valid]
        lbl_arr   = lbl_arr[row_valid]
        lbl_cat   = lbl_cat_full[row_valid]
        probe_arr = probe_arr[row_valid]
    else:
        lbl_cat = lbl_cat_full

    print("\nPer-probe sample counts after filtering:")
    for p in np.unique(probe_arr):
        print(f"  {p}: {np.sum(probe_arr == p)} samples")
    print(f"Total valid samples: {len(probe_arr)}")

    unique_probes = np.unique(probe_arr)
    print(f"Available probe IDs:\n{unique_probes}")

    def _split_train_val(idx_pool, strat_labels):
        """Random train/val split, stratified by class when feasible."""
        if classification and len(np.unique(strat_labels)) > 1:
            return train_test_split(idx_pool, test_size=val_split,
                                    random_state=42, stratify=strat_labels)
        return train_test_split(idx_pool, test_size=val_split, random_state=42)

    # --- SPLIT ---
    if probe_id is not None:
        assert probe_id in unique_probes, \
            f"probe_id '{probe_id}' not found. Available: {unique_probes}"
        print(f"[Single-probe mode] probe_id = {probe_id}")
        probe_mask = probe_arr == probe_id
        X_p, y_p, c_p = act_arr[probe_mask], lbl_arr[probe_mask], lbl_cat[probe_mask]

        if classification and len(np.unique(c_p)) > 1:
            idx_trv, idx_te = train_test_split(np.arange(len(X_p)), test_size=0.2,
                                               random_state=42, stratify=c_p)
        else:
            idx_trv, idx_te = train_test_split(np.arange(len(X_p)), test_size=0.2,
                                               random_state=42)
        idx_tr_rel, idx_val_rel = _split_train_val(np.arange(len(idx_trv)), c_p[idx_trv])
        idx_tr, idx_val = idx_trv[idx_tr_rel], idx_trv[idx_val_rel]

        X_train, y_train, y_lbl_train = X_p[idx_tr],  y_p[idx_tr],  c_p[idx_tr]
        X_val,   y_val,   y_lbl_val   = X_p[idx_val], y_p[idx_val], c_p[idx_val]
        X_test,  y_test,  y_lbl_test  = X_p[idx_te],  y_p[idx_te],  c_p[idx_te]
        active_test_probe = probe_id

    else:
        if test_probe is None:
            rng = np.random.default_rng(42)
            test_probe = rng.choice(unique_probes)
            print(f"No test_probe specified — randomly selected: {test_probe}")

        test_list = np.atleast_1d(test_probe)
        for tp in test_list:
            assert tp in unique_probes, f"test probe '{tp}' not found. Available: {unique_probes}"
        print(f"[LOPO] test probes: {list(test_list)}")
        test_mask  = np.isin(probe_arr, test_list)
        train_mask = ~test_mask

        X_trv, y_trv, c_trv = act_arr[train_mask], lbl_arr[train_mask], lbl_cat[train_mask]
        idx_tr, idx_val = _split_train_val(np.arange(len(X_trv)), c_trv)

        X_train, y_train, y_lbl_train = X_trv[idx_tr],  y_trv[idx_tr],  c_trv[idx_tr]
        X_val,   y_val,   y_lbl_val   = X_trv[idx_val], y_trv[idx_val], c_trv[idx_val]
        X_test,  y_test,  y_lbl_test  = act_arr[test_mask], lbl_arr[test_mask], lbl_cat[test_mask]
        active_test_probe = test_probe

    print(f"Split sizes — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # cast to float32 before arithmetic
    X_train = X_train.astype(np.float32, copy=False)
    X_val   = X_val.astype(np.float32, copy=False)
    X_test  = X_test.astype(np.float32, copy=False)

    # --- input normalization: z-score, fit on TRAIN only, with std floor ---
    x_mean = X_train.mean(axis=0)
    x_std  = X_train.std(axis=0)
    n_floored = int((x_std < std_floor).sum())
    print(f"features below std floor ({std_floor}): {n_floored} / {x_std.size}")
    x_std = np.maximum(x_std, std_floor)          # dead features stay suppressed, not amplified
    X_train_norm = ((X_train - x_mean) / x_std).astype(np.float32)
    X_val_norm   = ((X_val   - x_mean) / x_std).astype(np.float32)
    X_test_norm  = ((X_test  - x_mean) / x_std).astype(np.float32)

    # --- target normalization: SHIFTED z-score, fit on TRAIN only ---
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0)
    z_train = (y_train - y_mean) / (y_std + 1e-8)
    z_val   = (y_val   - y_mean) / (y_std + 1e-8)
    z_test  = (y_test  - y_mean) / (y_std + 1e-8)
    y_shift = -z_train.min(axis=0)
    y_train_norm = (z_train + y_shift).astype(np.float32)
    y_val_norm   = (z_val   + y_shift).astype(np.float32)
    y_test_norm  = (z_test  + y_shift).astype(np.float32)
    print(f"y_mean: {y_mean}  y_std: {y_std}  y_shift: {y_shift}")

    if ret_np:
        return (X_train_norm, y_lbl_train, y_train_norm), \
               (X_val_norm,   y_lbl_val,   y_val_norm), \
               (X_test_norm,  y_lbl_test,  y_test_norm), \
               y_mean, y_std, y_shift, MONO_DOM_LIST, active_test_probe

    train_dl = DataLoader(ActivationData(X_train_norm, y_lbl_train, y_train_norm),
                          batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dl   = DataLoader(ActivationData(X_val_norm,   y_lbl_val,   y_val_norm),
                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(ActivationData(X_test_norm,  y_lbl_test,  y_test_norm),
                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl, y_mean, y_std, y_shift, MONO_DOM_LIST, active_test_probe



"""
def combine_act(path_folder=None, collapsed=False, volta=True, model_last='3d'):
    act_arr_f, _, _, paths = get_activation_label_pair(
            path_folder=path_folder, collapsed=collapsed, volta=volta, model_last=model_last)
    act_arr_s, _, _, _ = get_activation_label_pair(
            path_folder=path_folder, collapsed=collapsed, volta= not volta, model_last=model_last)
    
    name = model_last
    print(f"First got index 0 act shape {act_arr_f[0].shape}")
    print(f"Got act len {len(act_arr_f)}")

    #act_arr_dyna_split = np.split(act_arr_s, act_arr_s.shape[-1], axis=3) if volta else np.split(act_arr_f, act_arr_f.shape[-1], axis=3)  
        
    print("Got activation + volta pairs")

    non_volta = act_arr_s if volta else act_arr_f
    volta_arr = act_arr_f if volta else act_arr_s

    print(f"Non volta shape at index 0: {non_volta[0].shape}")
    act_arr_dyna_split = [np.split(act, act.shape[-1], axis=3) for act in non_volta]
    
    #act_arr_dyna_split.append(act_arr_f) if volta else act_arr_dyna_split.append(act_arr_s)

    #act_arr = np.concatenate(act_arr_dyna_split, axis=2)

    final = []
    for i, arr in enumerate(act_arr_dyna_split):
        k = arr
        k.append(volta_arr[i])
        final.append(np.concatenate(k, axis=3))
    
    print("Combined")
    
    for i, k in enumerate(paths):
        out = k / f"combined_{name}.npy"
        if Path(out).is_file():
            print(f"Skipped {out}")
            continue
        else:
            np.save(out, final[i])
            print(f"saved to {out}, with shape {final[i].shape}")

    return act_arr_dyna_split, volta_arr, paths
"""

def combine_act(path_folder=None, model_last='3d', overwrite=False):
    """
    Per-path channel concat: expert weights + raw voltammogram.
    Saves (N, sweep, 1000, n_exp + 1) -> combined_{model_last}.npy
    """
    paths = get_all_paths(path_folder, collapsed=False, volta=False, model_last=model_last, combined=False)

    for k in paths:
        out = k / f"combined_{model_last}.npy"
        if out.is_file() and not overwrite:
            print(f"skip (exists): {out}")
            continue

        w = np.load(k / f"weights_{model_last}.npy")    # (N, sweep, 1000, n_exp)
        v = np.load(k / "voltammograms.npy")            # (N, 1000, sweep)

        assert w.ndim == 4, f"expected 4D weights, got {w.shape} @ {k}"
        v = v.transpose(0, 2, 1)[..., None]             # (N, sweep, 1000, 1)
        assert w.shape[:3] == v.shape[:3], f"mismatch {w.shape} vs {v.shape} @ {k}"

        c = np.concatenate([w, v.astype(w.dtype)], axis=3)
        np.save(out, c)
        print(f"saved {out}  {c.shape}")

    return paths