from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import sys
from tqdm import tqdm
import pickle
from pathlib import Path

sys.path.append('..')

import src_v1.utils as utils 

weights_paths = "/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/data_1d_vxlbl/ALCIS_155_macroREF"

probes = ['ALCIS_155_macroREF__ALC1_INM001_01bW04R01M',
 'ALCIS_155_macroREF__ALC2_INM001_02bW02R01M',
 'ALCIS_155_macroREF__ALC3_INM001_03bW02R01M',
 'ALCIS_155_macroREF__ALC3_INM001_03bW06R02M',
 'ALCIS_155_macroREF__ALC4_INM001_04bW02R01M',
 'ALCIS_155_macroREF__ALC4_INM001_04bW06R02M',
 'ALCIS_155_macroREF__ALC4_INM001_04bW08R03M']

T = 1000

TARGET_IDX = [0, 1, 2, 3]   # DA, 5HT, NE, pH
TARGET_NAMES = ['DA', '5HT', 'NE', 'pH']


probe_tags = {
    0: 'alc1',
    1: 'alc2',
    2: 'alc3_t1',
    3: 'alc3_t2',
    4: 'alc4_t1',
    5: 'alc4_t2',
    6: 'alc4_t3',
}

  # adjust as needed

def save_per_probe(out, path):
    np.savez(
        path,
        r2=out['r2'],
        r2_per_tgt=out['r2_per_tgt'],
        rmse_real=out['rmse_real'],
        #This needs bugfixing, broadcasting error, no idea why
        #y_pred_folds=np.array(out['y_pred_folds'], dtype=object),
        y_true_folds=np.array(out['y_true_folds'], dtype=object),
        y_means=out['y_means'],
        y_stds=out['y_stds'],
        probe_id=str(out['probe_id']),
    )
    
def linreg_regressor(train, test, y_mean, y_std, alpha=1.0):
    """
    Per time step t, fit a multi-output Ridge on (X[:, t, :], y_norm).
    Predictions are de-normalized back to real units (nM for DA/5HT/NE, pH for pH).

    Args
    ----
    train : (X_tr (N_tr, T, F), _, y_tr_norm (N_tr, 4))
    test  : (X_te (N_te, T, F), _, y_te_norm (N_te, 4))
    y_mean, y_std : (4,) arrays used to invert the normalization

    Returns dict with:
      r2          : (T,)            uniform-average R² across targets, on real-unit y
      r2_per_tgt  : (T, 4)          per-target R² on real-unit y
      rmse_real   : (T, 4)          per-target RMSE in real units
      y_pred_real : (T, N_te, 4)    predictions in real units, all time steps
      y_true_real : (N_te, 4)       ground truth in real units (same across t)
    """
    X_tr_full, _, y_tr_norm = train
    X_te_full, _, y_te_norm = test

    y_tr = y_tr_norm                    # (N_tr, 4) normalized
    y_te = y_te_norm                    # (N_te, 4) normalized

    N_te = X_te_full.shape[0]
    n_targets = y_tr.shape[1]

    # de-normalize ground truth once
    y_true_real = y_te * (y_std + 1e-8) + y_mean   # (N_te, 4)

    r2_t         = np.zeros(T)
    r2_per_tgt_t = np.zeros((T, n_targets))
    rmse_real_t  = np.zeros((T, n_targets))
    y_pred_real  = np.zeros((T, N_te, n_targets))

    for t in tqdm(range(T), unit='Time Step'):
        X_tr = X_tr_full[:, t, :]
        X_te = X_te_full[:, t, :]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        reg = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
        reg.fit(X_tr_s, y_tr)

        y_pred_norm = reg.predict(X_te_s)                          # (N_te, 4) normalized
        y_pred = y_pred_norm * (y_std + 1e-8) + y_mean             # (N_te, 4) real units

        y_pred_real[t]   = y_pred
        r2_t[t]          = r2_score(y_true_real, y_pred, multioutput='uniform_average')
        r2_per_tgt_t[t]  = r2_score(y_true_real, y_pred, multioutput='raw_values')
        rmse_real_t[t]   = np.sqrt(mean_squared_error(y_true_real, y_pred, multioutput='raw_values'))

    return {
        'r2':          r2_t,
        'r2_per_tgt':  r2_per_tgt_t,
        'rmse_real':   rmse_real_t,
        'y_pred_real': y_pred_real,
        'y_true_real': y_true_real,
    }

def logreg_classifier(train, test):
    
    acc_over_t = []
    for t in tqdm(range(T),unit='Time Step'):
        X_train = train[0][:,t,:]
        X_test = test[0][:,t,:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = LogisticRegression(max_iter=10000, C=1.0, class_weight='balanced',solver='saga')
        reg.fit(X_train_s, train[1])

        acc_over_t.append(reg.score(X_test_s, test[1]))

    return acc_over_t


def k_fold_decoding(probe_list,model_dim='3d',collapsed=True, volta=False):
    all_acc_curves = []
    for test_probe in probe_list:
        train, test, _, _, _, _ = utils.load_activation_data(weights_paths,
                    test_probe=test_probe,
                    ret_np=True,
                    model_dim=model_dim,
                    collapsed=collapsed,
                    volta=volta
                    )
        acc_t = logreg_classifier(train, test)
        all_acc_curves.append(acc_t)

    return np.array(all_acc_curves)
def per_probe_regression(probe_id, model_last='3d', collapsed=False, volta=False,
                         alpha=1.0, n_splits=5, random_state=42, combined = False):
    """
    Within-probe k-fold regression. Sweep-level random splits (leakage present
    when collapsed=False — to be patched later).

    Returns:
      r2:           (n_splits, T)            uniform-avg R² over targets, real units
      r2_per_tgt:   (n_splits, T, n_targets) per-target R², real units
      rmse_real:    (n_splits, T, n_targets) per-target RMSE, real units
      y_pred_folds: list of (T, N_te_fold, n_targets) — predictions per fold, real units
      y_true_folds: list of (N_te_fold, n_targets)    — ground truth per fold, real units
      y_means:      (n_splits, n_targets)
      y_stds:       (n_splits, n_targets)
      probe_id:     str
    """
    # Single load — recover raw y via inverse of the loader's normalization.
    
    train, _, test, y_mean_loader, y_std_loader, _, _, _= utils.load_activation_data(
            weights_paths,
            probe_id=probe_id,
            ret_np=True,
            model_last=model_last,
            collapsed=collapsed,
            volta=volta,
            classification=False,
            combined=combined
        )

    X_full = np.concatenate([train[0], test[0]], axis=0)
    y_full_norm = np.concatenate([train[2], test[2]], axis=0)
    y_full_raw  = y_full_norm * (y_std_loader + 1e-8) + y_mean_loader

    # Note on X: the loader normalized X using stats from its internal 80/20 train
    # split, not from each k-fold train set. For a strictly clean k-fold we'd
    # re-normalize X per fold using only that fold's training data. Within a single
    # probe the X distribution is fairly stable across folds, so this is a small
    # leak — fine for now, fix when the loader exposes x_mean/x_std.

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_targets = y_full_raw.shape[1]

    r2_all         = np.zeros((n_splits, T))
    r2_per_tgt_all = np.zeros((n_splits, T, n_targets))
    rmse_all       = np.zeros((n_splits, T, n_targets))
    y_pred_folds   = []
    y_true_folds   = []
    y_means        = np.zeros((n_splits, n_targets))
    y_stds         = np.zeros((n_splits, n_targets))

    for fold, (idx_tr, idx_te) in enumerate(kf.split(X_full)):
        X_tr, X_te = X_full[idx_tr], X_full[idx_te]
        y_tr_raw, y_te_raw = y_full_raw[idx_tr], y_full_raw[idx_te]

        # per-fold y normalization (fit on train only)
        y_mean = y_tr_raw.mean(axis=0)
        y_std  = y_tr_raw.std(axis=0)
        y_tr_norm = (y_tr_raw - y_mean) / (y_std + 1e-8)
        y_te_norm = (y_te_raw - y_mean) / (y_std + 1e-8)
        y_means[fold] = y_mean
        y_stds[fold]  = y_std

        out = linreg_regressor(
            (X_tr, None, y_tr_norm),
            (X_te, None, y_te_norm),
            y_mean=y_mean, y_std=y_std,
            alpha=alpha,
        )
        r2_all[fold]         = out['r2']
        r2_per_tgt_all[fold] = out['r2_per_tgt']
        rmse_all[fold]       = out['rmse_real']
        y_pred_folds.append(out['y_pred_real'])
        y_true_folds.append(out['y_true_real'])

        print(f"  Fold {fold+1}/{n_splits}: peak R² = {out['r2'].max():.3f} at t={out['r2'].argmax()}")

    return {
        'r2':           r2_all,
        'r2_per_tgt':   r2_per_tgt_all,
        'rmse_real':    rmse_all,
        'y_pred_folds': y_pred_folds,
        'y_true_folds': y_true_folds,
        'y_means':      y_means,
        'y_stds':       y_stds,
        'probe_id':     probe_id,
    }


#acc_over_t_collapsed_over_electrode = k_fold_decoding(probe_list=probes, model_dim=None)
#np.save('k_fold_acc_over_t_collapsed_over_electrode_6d.npy',acc_over_t_collapsed_over_electrode)


def full_reg(model_last = '3d', collapsed=False, alpha = 1.0, volta=False, combined=True):
    folder_tag = "combined" if combined else ""

    out_dir = Path(f"./kfold_{folder_tag}") 

    for i, tag in probe_tags.items():
        if combined:
            out = per_probe_regression(probes[i], model_last=model_last, collapsed=collapsed,
                                    volta=volta, alpha=alpha, combined=combined)
            
            save_per_probe(out, out_dir / f'kfold_reg_over_t_fullCombined_{tag}.npz')

        else:
            print(f"\n=== {tag} ({probes[i]}) ===")

            print(" [fullAct, 3d]")
            out_act = per_probe_regression(probes[i], model_last='3d', collapsed=False,
                                        volta=False, alpha=1.0)
            save_per_probe(out_act, out_dir / f'kfold_reg_over_t_fullAct_{tag}.npz')

            print(" [fullVolta]")
            out_raw = per_probe_regression(probes[i], model_last=None, collapsed=False,
                                        volta=True, alpha=1.0)
            save_per_probe(out_raw, out_dir / f'kfold_reg_over_t_fullVolta_{tag}.npz')

def collapsed_reg():
    for i, tag in probe_tags.items():
        print(f"\n=== {tag} ({probes[i]}) ===")

        print(" [fullAct, 3d]")
        out_act = per_probe_regression(probes[i], model_dim='3d', collapsed=True,
                                    volta=False, alpha=1.0)
        save_per_probe(out_act, out_dir / f'kfold_reg_over_t_collapsedAct_{tag}.npz')


if __name__ == '__main__':
    #collapsed_reg()
    #full_reg(combined=True)
    
    out = per_probe_regression(probe_id=probes[1], model_last='fine_tune_2_epoch50', combined=False)

    save_per_probe(out=out, path="/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/dyna_test/fscv_decoding/time_wise/alc2_finetune_2_epoch50/kfold_reg_over_t_fullActivationsFineTune2E50_alc2.npz")