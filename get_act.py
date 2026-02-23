import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('../DynaMix-python')

from src.model.forecaster import DynaMixForecaster
from src.utilities.utilities import load_hf_model

# ===== CONFIG =====
DATA_ROOT = "/Users/saru/Local_Work/neuromodul/fscv/tests_code/data_1d_vxlbl"
CL = 9000   # context length fed to DynaMix

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ===== LOAD MODEL ONCE =====
print("Loading model...")
model = load_hf_model("dynamix-6d-alrnn-v1.0")
model.eval()
model = model.to(device)
forecaster = DynaMixForecaster(model)
print("Model ready.\n")

# ===== INFERENCE LOOP =====
volt_paths = sorted(Path(DATA_ROOT).rglob("voltammograms.npy"))
print(f"Found {len(volt_paths)} voltammogram file(s).\n")

for volt_path in volt_paths:
    print(f"Processing: {volt_path}")
    V = np.load(str(volt_path))  # shape (N_trials, T=1000, S=sweeps)
    N_trials, T, S = V.shape

    all_weights = []  # will collect (S, CL_out, n_experts) per trial

    for trial_idx in range(N_trials):
        trial_weights = []

        for sweep_idx in range(S):
            sweep = V[trial_idx, :, sweep_idx].astype(np.float32)  # (1000,)

            # Repeat sweep to reach context length
            n_reps = int(np.ceil((CL + T) / len(sweep)))
            ts_long = np.tile(sweep, n_reps)[:CL + T]

            context = torch.tensor(ts_long[:CL].reshape(-1, 1), dtype=torch.float32).to(device)

            with torch.no_grad():
                _, _, W_gen = forecaster.forecast(
                    context=context,
                    horizon=T,
                    preprocessing_method="pos_embedding",
                    standardize=True,
                    fit_nonstationary=True,
                    return_latent=True,
                    return_expert_weights=True,
                )

            w = W_gen.cpu().numpy()[..., 0]  # (T, n_experts)
            trial_weights.append(w)

        # stack sweeps: (S, T, n_experts)
        all_weights.append(np.stack(trial_weights, axis=0))

    # final shape: (N_trials, S, T, n_experts)
    weights_arr = np.stack(all_weights, axis=0)

    out_path = volt_path.parent / "weights.npy"
    np.save(str(out_path), weights_arr)
    print(f"  → weights saved: {out_path}  shape={weights_arr.shape}\n")

print("Done.")