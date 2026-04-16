import torch
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import time
sys.path.append('../../DynaMix-python_vWorks')
CL = 9000

from src.model.forecaster import DynaMixForecaster
from src.utilities.utilities import load_hf_model

# ===== CONFIG =====
DATA_ROOT = "/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/data_1d_vxlbl/ALCIS_155_macroREF"



def inference(collapse=True, step=1000, model_d=3):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # ===== LOAD MODEL ONCE =====
    model_name = "dynamix-3d-alrnn-v1.0" if model_d==3 else "dynamix-6d-alrnn-v1.0"
    print("Loading model...")
    print(f"Dynamix reconstruct  model: {model_name} ")
    time.sleep(3.0)
    model = load_hf_model(model_name)
    model.eval()
    model = model.to(device)
    forecaster = DynaMixForecaster(model)
    print("Model ready.\n")
    print(f"Collapsed:  {collapse}")
    # ===== INFERENCE LOOP =====
    volt_paths = sorted(Path(DATA_ROOT).rglob("voltammograms.npy"))
    print(f"Found {len(volt_paths)} voltammogram file(s).\n")
    T=step
    for volt_path in tqdm(volt_paths, desc='Files',leave=False):

        if model_d == 3:
            out_path = volt_path.parent / "weights_collapsed_3d.npy" if collapse else volt_path.parent / "weights_3d.npy"
        else:
            out_path = volt_path.parent / "weights_collapsed.npy" if collapse else volt_path.parent / "weights.npy"

        if out_path.exists():
            print(f" skipping {volt_path.parent} (weights already exist)\n")
            continue

        V = np.load(str(volt_path))  # shape (N_trials, T=1000, S=sweeps)
        N_trials, _, S = V.shape
        
        all_weights = []  # will collect (S, CL_out, n_experts) per trial

        V_mean = V.mean(axis=2) if collapse else V
        if collapse:
            for trial_idx in tqdm(range(N_trials), desc='Trials', leave=False):
                sweep = V_mean[trial_idx, :].astype(np.float32)  # (1000,)

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
                all_weights.append(w)
        else:
            for trial_idx in tqdm(range(N_trials), desc='Trials', leave=False):
                trial_weights = []

                for sweep_idx in tqdm(range(S), desc='Sweeps', leave=False):
                    sweep = V_mean[trial_idx,:, sweep_idx].astype(np.float32)  # (1000,)

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

        np.save(str(out_path), weights_arr)
        print(f"  → weights saved: {out_path}  shape={weights_arr.shape}\n")

    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collapse', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--dim',type=int, default=3)
    args = parser.parse_args()

    inference(collapse=args.collapse,step=args.step,model_d=args.dim)


if __name__ == '__main__':
    main()
