import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dynamix.model.forecaster import DynaMixForecaster 
from utils import get_activation_label_pair, load_activation_data
import copy
import numpy as np
import os
from scipy.stats import pearsonr

PROBE_DEFAULT = 'ALCIS_155_macroREF__ALC1_INM001_01bW04R01M'
CHANNELS = ['DA', '5HT', 'NE', 'pH']


class TemporalRegressionHead(nn.Module):
    """
    w_exp sequence (B, T, n_exp) -> 1D conv over time -> global avg pool -> concentrations.
    Conv is shift-equivariant: this is the property that made InceptionTime win on w_exp.
    """
    def __init__(self, n_exp=10, n_neuromodul=4, ch=32, kernel_size=7, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_exp, ch, kernel_size=kernel_size, padding=pad), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=pad), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(ch, n_neuromodul),
            nn.Softplus(),                          # non-negative concentrations
        )

    def forward(self, w_exp_seq):                   # (B, T, n_exp)
        x = w_exp_seq.transpose(1, 2)               # (B, n_exp, T) -- conv1d wants channels-first
        x = self.conv(x)                            # (B, ch, T)
        x = x.mean(dim=-1)                          # global avg pool over time -> (B, ch)
        return self.fc(x)                           # (B, n_neuromodul)


class CoupledDynamixRegressor(nn.Module):
    """
    Couples DynaMix forecasting with a temporal-conv regression head.
    Trainable:  gating_network.mlp_layer1/2 weights + TemporalRegressionHead.
    Frozen:     experts, conv, D, sigma, softmax_temp1/2, B, mlp biases.
    Gradient from the regression loss reaches ONLY the gating MLP weights
    (z is detached every forecast step) and the head.
    """
    def __init__(self, dynamix_model, horizon=1000, n_neuromodul=4,
                 head_ch=32, head_kernel=7, head_dropout=0.2,
                 preprocessing_method='pos_embedding', standardize=True):
        super().__init__()
        self.dynamix = dynamix_model
        self.horizon = horizon
        self.n_exp = dynamix_model.Experts
        self.preprocessing_method = preprocessing_method
        self.standardize = standardize

        # forecaster reused ONLY for its preprocessing helpers (no grad needed there)
        self._forecaster = DynaMixForecaster(dynamix_model)

        # --- freeze everything in DynaMix ---
        for p in self.dynamix.parameters():
            p.requires_grad = False

        # --- unfreeze ONLY the two gating MLP weight matrices ---
        self.gating_params = []
        for layer in (self.dynamix.gating_network.mlp_layer1,
                      self.dynamix.gating_network.mlp_layer2):
            layer.weight.requires_grad = True
            self.gating_params.append(layer.weight)
            # biases were built with requires_grad=False — leave them frozen

        # --- temporal-conv regression head ---
        self.head = TemporalRegressionHead(
            n_exp=self.n_exp, n_neuromodul=n_neuromodul,
            ch=head_ch, kernel_size=head_kernel, dropout=head_dropout)

    def trainable_parameters(self):
        return self.gating_params + list(self.head.parameters())

    # ---- front end: reuse forecaster preprocessing, all under no_grad ----
    def _prepare_context(self, context, device):
        from dynamix.model.preprocessing import DataPreprocessor   # adjust import
        with torch.no_grad():
            context, initial_x, meta = self._forecaster._reshape_for_model(
                context, None, device)
            pre = DataPreprocessor(standardize=self.standardize,
                                   power_transform=False, detrending=False,
                                   preprocessing_method=self.preprocessing_method)
            context_embedded, initial_condition = pre.preprocess(
                context, self.dynamix.N, initial_x)
            z = self._forecaster._init_latent_state(initial_condition)
            precomputed_cnn = self.dynamix.precompute_cnn(context_embedded)
        return context_embedded, z, precomputed_cnn

    # ---- gradient-enabled forecast loop ----
    def _gradient_forecast(self, context_embedded, z, precomputed_cnn):
        w_seq = []
        for _ in range(self.horizon):
            z_in = z.detach()                     # state enters as frozen input
            z_next, w_exp = self.dynamix(
                z_in, context_embedded,
                precomputed_cnn=precomputed_cnn, return_w_exp=True)
            w_seq.append(w_exp.transpose(0, 1))   # (batch, n_exp)
            z = z_next
        return torch.stack(w_seq, dim=1)          # (batch, horizon, n_exp)

    def forward(self, context):
        # context: (seq_length, batch, feature_dim) or (seq_length, feature_dim)
        device = next(self.head.parameters()).device
        context_embedded, z, precomputed_cnn = self._prepare_context(context, device)
        w_exp_seq = self._gradient_forecast(context_embedded, z, precomputed_cnn)
        return self.head(w_exp_seq)               # (B, n_neuromodul)

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _grad_health_check(model):
    """One-time sanity print: confirm only the gating MLP weights + head get gradients."""
    print("\n" + "=" * 64)
    print("GRADIENT ISOLATION CHECK")
    print("=" * 64)
    trainable, frozen_sample = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(name)
        elif len(frozen_sample) < 3:
            frozen_sample.append(name)
    print(f"Trainable params ({len(trainable)}):")
    for n in trainable:
        print(f"    + {n}")
    print(f"Frozen (sample): {frozen_sample} ... [+{ ... }]")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,}  ({n_train/n_total:.2%})")
    print("=" * 64 + "\n")


def _post_backward_check(model):
    """After the first backward: gating MLP must have grad, experts must not."""
    g = model.dynamix.gating_network
    mlp1_grad = g.mlp_layer1.weight.grad
    expert0 = next(model.dynamix.experts[0].parameters())
    print("  [first-step grad check]")
    print(f"    mlp_layer1.weight.grad : "
          f"{'OK norm=%.3e' % mlp1_grad.norm() if mlp1_grad is not None else 'None  <-- PROBLEM'}")
    print(f"    mlp_layer2.weight.grad : "
          f"{'OK norm=%.3e' % g.mlp_layer2.weight.grad.norm() if g.mlp_layer2.weight.grad is not None else 'None  <-- PROBLEM'}")
    print(f"    expert[0] param.grad   : "
          f"{'None  (OK, frozen)' if expert0.grad is None else 'NON-NONE  <-- LEAK'}")
    print(f"    conv.weight.grad       : "
          f"{'None  (OK, frozen)' if g.conv.weight.grad is None else 'NON-NONE  <-- LEAK'}\n")


def _gating_drift(model, ref):
    """How far the gating MLP weights have moved from their pretrained values."""
    g = model.dynamix.gating_network
    d1 = (g.mlp_layer1.weight.detach() - ref['mlp1']).norm().item()
    d2 = (g.mlp_layer2.weight.detach() - ref['mlp2']).norm().item()
    return d1, d2


def train_coupled(model, train_loader, val_loader, loss_fn, optimizer,
                  device='cpu', num_epochs=100, n_neuromodul=4,
                  channel_names=('DA', '5HT', 'NE', 'pH')):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    model.to(device)

    _grad_health_check(model)

    # snapshot pretrained gating weights to measure drift later
    g = model.dynamix.gating_network
    ref = {'mlp1': g.mlp_layer1.weight.detach().clone(),
           'mlp2': g.mlp_layer2.weight.detach().clone()}

    print(f"Starting finetuning: {num_epochs} epochs | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val sweeps | "
          f"horizon={model.horizon} | device={device}\n")

    train_losses, val_losses = [], []
    first_backward_done = False

    for epoch in range(num_epochs):
        model.train()
        run_loss, n_seen = 0.0, 0
        per_ch_abs = torch.zeros(n_neuromodul)

        for b, (context, targets) in enumerate(train_loader):
            context, targets = context.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(context)
            l = loss_fn(out, targets)
            l.backward()

            if not first_backward_done:        # one-time leak check
                _post_backward_check(model)
                first_backward_done = True

            optimizer.step()

            bs = targets.size(0)
            run_loss += l.item() * bs
            n_seen += bs
            per_ch_abs += (out.detach().cpu() - targets.cpu()).abs().sum(dim=0)

            if b == 0 or (b + 1) % 2000 == 0:     # intra-epoch heartbeat
                print(f"  ep {epoch+1:3d} | batch {b+1:3d}/{len(train_loader)} "
                      f"| batch loss {l.item():.4f}")

        ep_train = run_loss / n_seen
        train_losses.append(ep_train)
        per_ch_train = per_ch_abs / n_seen      # mean abs err per channel (z-scored units)

        # ---- validation ----
        model.eval()
        v_loss, v_seen = 0.0, 0
        v_ch_abs = torch.zeros(n_neuromodul)
        with torch.no_grad():
            for context, targets in val_loader:
                context, targets = context.to(device), targets.to(device)
                out = model(context)
                bs = targets.size(0)
                v_loss += loss_fn(out, targets).item() * bs
                v_seen += bs
                v_ch_abs += (out.cpu() - targets.cpu()).abs().sum(dim=0)
        ep_val = v_loss / v_seen
        val_losses.append(ep_val)
        per_ch_val = v_ch_abs / v_seen

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(ep_val)
        new_lr = optimizer.param_groups[0]['lr']

        # ---- epoch summary ----
        d1, d2 = _gating_drift(model, ref)
        mark = '  *best*' if ep_val == min(val_losses) else ''
        print(f"[epoch {epoch+1:3d}/{num_epochs}] "
              f"train {ep_train:.4f} | val {ep_val:.4f}{mark}")
        print(f"    val MAE/ch (z): " +
              "  ".join(f"{c} {e:.3f}" for c, e in zip(channel_names, per_ch_val)))
        print(f"    gating drift  : mlp1 {d1:.4e}  mlp2 {d2:.4e}  | lr {new_lr:.2e}")
        if new_lr < prev_lr:
            print(f"    >> lr reduced {prev_lr:.2e} -> {new_lr:.2e}")
        print()

    print(f"Done. best val loss = {min(val_losses):.4f} "
          f"at epoch {val_losses.index(min(val_losses))+1}")
    return train_losses, val_losses

class Projector(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.project = nn.Linear(in_ch,1)
    def forward(self, x):
        return self.project(x).squeeze(-1)

class Regressor(nn.Module):
    def __init__(self, input_dim, n_neuromodul,hidden_dim1=312, hidden_dim2=256, hidden_dim3=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim,hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1,hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.GELU(),
            nn.Linear(hidden_dim3,n_neuromodul),
            nn.Softplus()
        )

    def forward(self, x):
        x = x.reshape(x.size(0),-1)

        return self.head(x)

def train_regressor(model, train_data_loader, val_loader, loss, optimizer,
                    device="cpu", num_epochs=50,
                    sched_patience=5, stop_patience=15, min_delta=0.0,
                    min_lr=1e-5, save_path=None, verbose_every=1):
    """
    Regression training with LR-on-plateau, early stopping, best-model save/restore.
    Loaders yield (activation, class_label, concentration); target = concentration.
    Returns the model with BEST (lowest-val) weights restored, not the last epoch's.
    """
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=sched_patience, min_lr=min_lr)

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, epochs_no_improve = 0, 0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        run_loss = 0.0
        for inputs, _lbl, targets in train_data_loader:        # 3-tuple unpack
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            l = loss(out, targets)
            l.backward()
            optimizer.step()
            run_loss += l.item() * inputs.size(0)
        ep_train_loss = run_loss / len(train_data_loader.dataset)
        train_losses.append(ep_train_loss)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _lbl, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += loss(model(inputs), targets).item() * inputs.size(0)
        ep_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(ep_val_loss)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(ep_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # ---- track best + early stop ----
        if ep_val_loss < best_val - min_delta:
            best_val, best_epoch, epochs_no_improve = ep_val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save({'model_state': best_state, 'epoch': best_epoch,
                            'val_loss': best_val, 'optimizer_state': optimizer.state_dict()},
                           save_path)
        else:
            epochs_no_improve += 1

        if verbose_every and (epoch + 1) % verbose_every == 0:
            mark = '  *best*' if best_epoch == epoch + 1 else ''
            print(f"Epoch {epoch+1:3d}/{num_epochs} | train {ep_train_loss:.4f} | "
                  f"val {ep_val_loss:.4f}{mark}"
                  + (f"  | lr {prev_lr:.1e}->{new_lr:.1e}" if new_lr < prev_lr else ""))

        if epochs_no_improve >= stop_patience:
            print(f"Early stop @ epoch {epoch+1} | best val {best_val:.4f} @ epoch {best_epoch}")
            break

    model.load_state_dict(best_state)            # caller gets the best model
    print(f"Restored best model: val {best_val:.4f} @ epoch {best_epoch}")
    return train_losses, val_losses


def test_regressor(model, test_data_loader, loss, y_mean, y_std, y_shift, device='cpu'):
    model.to(device)
    model.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, _lbl, targets in test_data_loader:        # 3-tuple unpack (still needed)
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            test_loss += loss(out, targets).item() * inputs.size(0)
            all_preds.append(out.cpu())
            all_targets.append(targets.cpu())

    avg_loss = test_loss / len(test_data_loader.dataset)
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = (all_preds - all_targets).abs().mean(dim=0)         # z-space MAE

    y_mean  = torch.tensor(y_mean,  dtype=torch.float32)
    y_std   = torch.tensor(y_std,   dtype=torch.float32)
    y_shift = torch.tensor(y_shift, dtype=torch.float32)

    # forward was:  y_norm = (y - mean)/std + shift
    # so revert is: y      = (y_norm - shift)*std + mean
    all_preds_nM   = (all_preds   - y_shift) * y_std + y_mean
    all_targets_nM = (all_targets - y_shift) * y_std + y_mean
    mae_nM = (all_preds_nM - all_targets_nM).abs().mean(dim=0)

    return avg_loss, mae, mae_nM, all_preds_nM, all_targets_nM

     


def run_lopo_cv(path_folder, volta, n_neuromodul=4,
                batch_size=512, num_epochs=50, lr=1e-3,
                device='cuda', model_dim='3d', seed=42, tag=None):
    """
    Leave-one-probe-out CV (K = number of probes). One probe held out per fold;
    train+val drawn from the rest. Returns per-fold nM-space MAE and the pooled
    test predictions (every probe predicted exactly once -> full coverage).
    """
    tag = tag or ('volta' if volta else 'act')
    torch.manual_seed(seed)

    # ---- load + concatenate ONCE (re-used across all folds) ----
    act_arr, lbl_arr, probe_arr = get_activation_label_pair(
        path_folder=path_folder, collapsed=False, volta=volta, model_dim=model_dim)
    preloaded = (act_arr, lbl_arr, probe_arr)

    input_dim = int(np.prod(act_arr.shape[1:]))        # 1000 (volta) or 10000 (act)
    probes = np.unique(probe_arr)
    print(f"\n[{tag}] input_dim={input_dim} | {len(probes)} probes | "
          f"{act_arr.nbytes/1e9:.1f} GB | folds=LOPO(K={len(probes)})")

    fold_mae, pooled_preds, pooled_tgts, pooled_probe = [], [], [], []

    for k, tp in enumerate(probes):
        print(f"\n===== [{tag}] Fold {k+1}/{len(probes)} | test probe: {tp} =====")

        train_dl, val_dl, test_dl, y_mean, y_std, y_shift, _, _ = load_activation_data(
            preloaded=preloaded, test_probe=tp, batch_size=batch_size,
            collapsed=False, volta=volta, classification=False, model_dim=model_dim)

        # fresh model / optimizer / loss every fold (no leakage across folds)
        model     = Regressor(input_dim=input_dim, n_neuromodul=n_neuromodul)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn   = nn.MSELoss()

        train_regressor(model, train_dl, val_dl, loss_fn, optimizer,
                        device=device, num_epochs=num_epochs,
                        save_path=f'best_{tag}_fold{k}_{tp}.pt')

        _, mae_z, mae_nM, preds_nM, tgts_nM = test_regressor(
            model, test_dl, loss_fn, y_mean, y_std, y_shift, device=device)

        print(f"  test MAE  " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, mae_nM)))

        fold_mae.append(mae_nM)
        pooled_preds.append(preds_nM)
        pooled_tgts.append(tgts_nM)
        pooled_probe.append(np.full(len(preds_nM), tp))

        del model, optimizer, train_dl, val_dl, test_dl
        torch.cuda.empty_cache()

    fold_mae     = torch.stack(fold_mae)               # (K, 4)
    pooled_preds = torch.cat(pooled_preds)             # (N_total, 4)
    pooled_tgts  = torch.cat(pooled_tgts)
    pooled_probe = np.concatenate(pooled_probe)

    print(f"\n========== [{tag}] LOPO summary (units: nM for DA/5HT/NE, pH for pH) ==========")
    print("per-fold MAE:")
    for k, tp in enumerate(probes):
        print(f"  {tp:28s} " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, fold_mae[k])))
    print("-" * 70)
    print("  mean  " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, fold_mae.mean(0))))
    print("  std   " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, fold_mae.std(0))))

    results = {'tag': tag, 'probes': probes, 'fold_mae': fold_mae,
               'preds_nM': pooled_preds, 'tgts_nM': pooled_tgts, 'probe_of_row': pooled_probe}
    torch.save(results, f'lopo_{tag}.pt')
    return results



def fit_within_probe(path_folder, volta, probe=PROBE_DEFAULT, n_neuromodul=4,
                     batch_size=128, num_epochs=80, lr=1e-3,
                     device='cuda', model_name='3d', seed=42, tag=None,
                     out_root='within_probe',combined=False):
    """
    Within-probe MLP (80/20 test, val carved from train). Mirrors run_lopo_cv:
    fresh model, best-model save, results dict to disk. Outputs land in
    out_root/<probe_short>/<tag>/ .
    """
    tag = tag or ('volta' if volta else 'act')
    torch.manual_seed(seed)

    probe_short = probe.split('__')[-1]                  # strip shared prefix for folder name
    out_dir = os.path.join(out_root, probe_short, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[{tag}] within-probe {probe_short} -> {out_dir}")

    train_dl, val_dl, test_dl, y_mean, y_std, y_shift, _, _ = load_activation_data(
        path_folder=path_folder, volta=volta, batch_size=batch_size, model_last=model_name,combined=combined)

    xb, _, _ = next(iter(train_dl))
    input_dim = int(np.prod(xb.shape[1:]))               # 1000 volta / 10000 act
    print(f"  input_dim={input_dim} | train {len(train_dl.dataset)} "
          f"val {len(val_dl.dataset)} test {len(test_dl.dataset)}")

    model     = Regressor(input_dim=input_dim, n_neuromodul=n_neuromodul)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    save_path = os.path.join(out_dir, 'best_model.pt')
    train_losses, val_losses = train_regressor(
        model, train_dl, val_dl, loss_fn, optimizer, device=device,
        num_epochs=num_epochs, save_path=save_path)

    _, mae_z, mae_nM, preds_nM, tgts_nM = test_regressor(
        model, test_dl, loss_fn, y_mean, y_std, y_shift, device=device)

    # slope + R2_corr per monoamine (same format as InceptionTime panels)
    print(f"  {'analyte':8}{'slope':>9}{'R2_corr':>10}{'MAE_nM':>10}")
    metrics = {}
    for j, c in enumerate(CHANNELS):
        x, y = tgts_nM.numpy()[:, j], preds_nM.numpy()[:, j]
        slope, intc = np.polyfit(x, y, 1)
        r2c = pearsonr(x, y)[0] ** 2 if np.std(x) > 1e-9 else float('nan')
        metrics[c] = {'slope': float(slope), 'intercept': float(intc),
                      'r2_corr': float(r2c), 'mae_nM': float(mae_nM[j])}
        print(f"  {c:8}{slope:9.3f}{r2c:10.3f}{mae_nM[j]:10.2f}")

    results = {'tag': tag, 'probe': probe, 'input_dim': input_dim,
               'preds_nM': preds_nM, 'tgts_nM': tgts_nM,
               'mae_nM': mae_nM, 'metrics': metrics,
               'train_losses': train_losses, 'val_losses': val_losses,
               'y_mean': y_mean, 'y_std': y_std, 'y_shift': y_shift}
    torch.save(results, os.path.join(out_dir, 'results.pt'))
    print(f"  saved -> {out_dir}/results.pt  +  best_model.pt")
    return results



