import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import json
import pickle
import fcntl  # for file locking
import math

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

with open("params.json", "r") as f:
    params = json.load(f)

main_job_dir = params['main_job_dir']
os.chdir(main_job_dir)

local_test = params['local_test']
PATIENCE = params['PATIENCE']
MIN_DELTA = params['MIN_DELTA']

combos = pd.read_csv("combos.csv")

task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', None)
if task_id_str is not None:
    task_id = int(task_id_str)
    combos = combos.iloc[[task_id]]
else:
    if local_test and len(combos) > 0:
        combos = combos.iloc[[0]]

with open("data.pkl", "rb") as f:
    train_input, val_input, test_input, train_condition, val_condition, test_condition = pickle.load(f)

positions_dim = train_input.shape[1]
condition_dim = train_condition.shape[1]

train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
val_input_tensor = torch.tensor(val_input, dtype=torch.float32)
test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
train_condition_tensor = torch.tensor(train_condition, dtype=torch.float32)
val_condition_tensor = torch.tensor(val_condition, dtype=torch.float32)
test_condition_tensor = torch.tensor(test_condition, dtype=torch.float32)

train_dataset = TensorDataset(train_input_tensor, train_condition_tensor)
val_dataset = TensorDataset(val_input_tensor, val_condition_tensor)
test_dataset = TensorDataset(test_input_tensor, test_condition_tensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean_pos = torch.zeros(positions_dim, dtype=torch.float32).to(device)
scale_pos = torch.ones(positions_dim, dtype=torch.float32).to(device)
mean_mom = torch.zeros(condition_dim, dtype=torch.float32).to(device)
scale_mom = torch.ones(condition_dim, dtype=torch.float32).to(device)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

epsilon = 1e-10
mC = 21894.71361
mO = 29164.39289
mS = 58441.80487
SAMPLES_TO_PRINT = 5

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss improved to {self.best_loss:.4f}")
        else:
            self.counter +=1
            print(f"No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

def compute_energy_loss(x_recon, x, y, scale_pos, mean_pos, scale_mom, mean_mom):
    x_recon_original = x_recon * scale_pos + mean_pos
    y_original = y * scale_mom + mean_mom

    momenta_C = y_original[:, 0:3]
    momenta_O = y_original[:, 3:6]
    momenta_S = y_original[:, 6:9]

    KE_C = torch.sum(momenta_C ** 2, dim=1) / (2 * mC)
    KE_O = torch.sum(momenta_O ** 2, dim=1) / (2 * mO)
    KE_S = torch.sum(momenta_S ** 2, dim=1) / (2 * mS)
    KE_total = KE_C + KE_O + KE_S

    positions_C = x_recon_original[:, 0:3]
    positions_O = x_recon_original[:, 3:6]
    positions_S = x_recon_original[:, 6:9]

    rCO = torch.clamp(torch.norm(positions_C - positions_O, dim=1), min=1e-10)
    rCS = torch.clamp(torch.norm(positions_C - positions_S, dim=1), min=1e-10)
    rOS = torch.clamp(torch.norm(positions_O - positions_S, dim=1), min=1e-10)

    PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)
    EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + 1e-10)
    EnergyLoss = torch.mean(EnergyDiff ** 2)
    return EnergyLoss

print("Starting training...")

for combo_index, combo in combos.iterrows():
    LATENT_DIM = int(combo['latent_dim'])
    EPOCHS = int(combo['epochs'])
    BATCH_SIZE = int(combo['batch_size'])
    LEARNING_RATE = float(combo['lr'])
    ACTIVATION_FUNCTION = combo['activation']
    NUM_HIDDEN_LAYERS = int(combo['num_hidden_layers'])
    HIDDEN_DIMS = combo['hidden_layer_sizes']
    if isinstance(HIDDEN_DIMS, str):
        HIDDEN_DIMS = eval(HIDDEN_DIMS)

    MSE_WEIGHT = float(combo['MSE_WEIGHT'])
    KLD_WEIGHT = float(combo['KLD_WEIGHT'])
    MRE2_WEIGHT = float(combo['MRE2_WEIGHT'])
    ENERGY_DIFF_WEIGHT = float(combo['ENERGY_DIFF_WEIGHT'])

    if local_test:
        EPOCHS = 1

    class CVAE(nn.Module):
        def __init__(self, input_dim, condition_dim, latent_dim, hidden_dims, activation_function):
            super(CVAE, self).__init__()
            activation_dict = {
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'leaky_relu': nn.LeakyReLU(),
                'elu': nn.ELU(),
                'selu': nn.SELU(),
                'gelu': nn.GELU(),
                'swish': nn.SiLU(),
                'prelu': nn.PReLU(),
            }
            if activation_function not in activation_dict:
                raise ValueError("Unsupported activation function")
            activation = activation_dict[activation_function]

            self.input_dim = input_dim
            self.condition_dim = condition_dim
            self.latent_dim = latent_dim

            encoder_layers = []
            prev_dim = input_dim + condition_dim
            for h_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, h_dim))
                encoder_layers.append(activation)
                prev_dim = h_dim
            self.encoder = nn.Sequential(*encoder_layers)
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

            decoder_layers = []
            rev_hidden = hidden_dims[::-1]
            prev_dim = latent_dim + condition_dim
            for h_dim in rev_hidden:
                decoder_layers.append(nn.Linear(prev_dim, h_dim))
                decoder_layers.append(activation)
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x, y):
            xy = torch.cat([x, y], dim=1)
            h = self.encoder(xy)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu+eps*std

        def decode(self,z,y):
            zy = torch.cat([z,y],dim=1)
            x_recon = self.decoder(zy)
            return x_recon

        def forward(self,x,y):
            mu, logvar = self.encode(x,y)
            logvar = torch.clamp(logvar, min=-10, max=10)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z,y)
            return x_recon, mu, logvar

    model = CVAE(positions_dim, condition_dim, LATENT_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    early_stopping = EarlyStopping()

    mu_sum = torch.zeros(LATENT_DIM, device=device)
    mu_square_sum = torch.zeros(LATENT_DIM, device=device)
    total_samples = 0

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_unreg_loss = 0
        total_recon_loss_sum = 0
        total_kld_loss_sum = 0
        total_mre2_loss_sum = 0
        total_energy_loss_sum = 0

        epoch_train_mre_sum = 0.0
        epoch_train_energy_diff_sum = 0.0
        epoch_train_recon_loss_sum = 0.0
        epoch_train_kld_sum = 0.0

        samples_this_epoch = 0
        progress_bar = tqdm(train_loader, desc=f"Combo {combo_index}/{len(combos)} - Epoch {epoch+1}/{EPOCHS}")

        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x, y)
            if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                print("NaN detected. Stopping training.")
                sys.exit(1)

            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
            kld_loss = -0.5*torch.mean(1+logvar - mu.pow(2)-logvar.exp())
            mre2_loss = torch.mean(((x - x_recon)/(torch.abs(x)+epsilon))**2)
            EnergyLoss = compute_energy_loss(x_recon, x, y, scale_pos, mean_pos, scale_mom, mean_mom)

            unreg_loss = (MSE_WEIGHT*recon_loss +
                          KLD_WEIGHT*kld_loss +
                          MRE2_WEIGHT*mre2_loss +
                          ENERGY_DIFF_WEIGHT*EnergyLoss)

            loss = unreg_loss
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            samples_this_epoch += bsz
            total_unreg_loss += unreg_loss.item()*bsz
            total_recon_loss_sum += recon_loss.item()*bsz
            total_kld_loss_sum += kld_loss.item()*bsz
            total_mre2_loss_sum += mre2_loss.item()*bsz
            total_energy_loss_sum += EnergyLoss.item()*bsz

            mu_sum += mu.sum(dim=0)
            mu_square_sum += (mu**2).sum(dim=0)
            total_samples += bsz

            epoch_train_recon_loss_sum += recon_loss.item()*bsz
            epoch_train_kld_sum += kld_loss.item()*bsz
            epoch_train_mre_sum += mre2_loss.item()*bsz
            epoch_train_energy_diff_sum += EnergyLoss.item()*bsz

            progress_bar.set_postfix({
                'Total Loss': f"{(total_unreg_loss/samples_this_epoch):.4f}",
                'Recon Loss': f"{(total_recon_loss_sum/samples_this_epoch):.4f}",
                'KLD Loss': f"{(total_kld_loss_sum/samples_this_epoch):.4f}",
                'MRE2 Loss': f"{(total_mre2_loss_sum/samples_this_epoch):.4f}",
                'Energy Loss': f"{(total_energy_loss_sum/samples_this_epoch):.4f}"
            })

            if local_test:
                # break after one batch in local test
                break

        if local_test:
            # no validation in local test, stop now
            break

        model.eval()
        total_val_loss=0
        val_samples=0
        epoch_val_recon_loss_sum=0.0
        epoch_val_kld_sum=0.0
        epoch_val_mre_sum=0.0
        epoch_val_energy_diff_sum=0.0

        for x_val,y_val in val_loader:
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

            with torch.no_grad():
                x_recon, mu, logvar = model(x_val,y_val)
                if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                    print("NaN in validation. Stopping.")
                    sys.exit(1)

                recon_loss_val = F.mse_loss(x_recon,x_val,reduction='mean')
                kld_loss_val = -0.5*torch.mean(1+logvar - mu.pow(2)-logvar.exp())
                mre2_loss_val = torch.mean(((x_val - x_recon)/(torch.abs(x_val)+epsilon))**2)

                x_recon_original = x_recon*scale_pos+mean_pos
                y_original = y_val*scale_mom+mean_mom

                momenta_C = y_original[:,0:3]
                momenta_O = y_original[:,3:6]
                momenta_S = y_original[:,6:9]

                KE_C = torch.sum(momenta_C**2,dim=1)/(2*mC)
                KE_O = torch.sum(momenta_O**2,dim=1)/(2*mO)
                KE_S = torch.sum(momenta_S**2,dim=1)/(2*mS)
                KE_total = KE_C+KE_O+KE_S

                positions_C = x_recon_original[:,0:3]
                positions_O = x_recon_original[:,3:6]
                positions_S = x_recon_original[:,6:9]

                rCO = torch.clamp(torch.norm(positions_C - positions_O,dim=1),min=1e-10)
                rCS = torch.clamp(torch.norm(positions_C - positions_S,dim=1),min=1e-10)
                rOS = torch.clamp(torch.norm(positions_O - positions_S,dim=1),min=1e-10)

                PE_pred = (4/rCO)+(4/rCS)+(4/rOS)
                EnergyDiff_val = (KE_total-PE_pred)/(torch.abs(KE_total)+1e-10)
                EnergyLoss_val = torch.mean(EnergyDiff_val**2)

                unreg_loss_val = (MSE_WEIGHT*recon_loss_val +
                                  KLD_WEIGHT*kld_loss_val +
                                  MRE2_WEIGHT*mre2_loss_val +
                                  ENERGY_DIFF_WEIGHT*EnergyLoss_val)

            bsz = x_val.size(0)
            total_val_loss += unreg_loss_val.item()*bsz
            val_samples+=bsz

            epoch_val_recon_loss_sum += recon_loss_val.item()*bsz
            epoch_val_kld_sum += kld_loss_val.item()*bsz
            epoch_val_mre_sum += mre2_loss_val.item()*bsz
            epoch_val_energy_diff_sum += EnergyLoss_val.item()*bsz

        avg_val_loss = total_val_loss/val_samples
        print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}')
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

end_time = time.time()
mu_train_mean = mu_sum/total_samples
var_train = (mu_square_sum/total_samples)-mu_train_mean**2
std_train = torch.sqrt(var_train+epsilon)

torch.save({
    'mu_train_mean': mu_train_mean.cpu(),
    'std_train': std_train.cpu(),
    'mean_pos': mean_pos.cpu(),
    'scale_pos': scale_pos.cpu(),
    'mean_mom': mean_mom.cpu(),
    'scale_mom': scale_mom.cpu()
}, os.path.join(main_job_dir,'latent_stats.pth'))

torch.save(model.state_dict(), os.path.join(main_job_dir,'cvae_model_final.pth'))
print("Training complete and model saved.")

print("Evaluating on test set...")
model.eval()
with torch.no_grad():
    latent_stats = torch.load(
        os.path.join(main_job_dir,'latent_stats.pth'),
        map_location=device
    )

    mu_train_mean = latent_stats['mu_train_mean'].to(device)
    std_train = latent_stats['std_train'].to(device)
    mean_pos = latent_stats['mean_pos'].to(device)
    scale_pos = latent_stats['scale_pos'].to(device)
    mean_mom = latent_stats['mean_mom'].to(device)
    scale_mom = latent_stats['scale_mom'].to(device)

    positions_test_tensor = test_input_tensor.to(device)
    momenta_test_tensor = test_condition_tensor.to(device)

    # Use the last combo's LATENT_DIM for decoding test set
    z = torch.randn(len(test_dataset), LATENT_DIM, device=device)*std_train+mu_train_mean
    y_test_tensor = momenta_test_tensor

    x_pred = model.decode(z,y_test_tensor)
    x_pred = x_pred*scale_pos+mean_pos
    x_pred = x_pred.cpu().numpy()

    x_test_np = (positions_test_tensor*scale_pos+mean_pos).cpu().numpy()

    MRE = np.mean(np.abs(x_test_np - x_pred)/(np.abs(x_test_np)+epsilon))*100
    MSE = np.mean((x_test_np - x_pred)**2)
    momenta_np = (momenta_test_tensor*scale_mom+mean_mom).cpu().numpy()

    momenta_C = momenta_np[:,0:3]
    momenta_O = momenta_np[:,3:6]
    momenta_S = momenta_np[:,6:9]

    KE_C = np.sum(momenta_C**2,axis=1)/(2*mC)
    KE_O = np.sum(momenta_O**2,axis=1)/(2*mO)
    KE_S = np.sum(momenta_S**2,axis=1)/(2*mS)
    KE_total = KE_C+KE_O+KE_S

    positions_C_pred = x_pred[:,0:3]
    positions_O_pred = x_pred[:,3:6]
    positions_S_pred = x_pred[:,6:9]

    rCO_pred = np.maximum(np.linalg.norm(positions_C_pred - positions_O_pred,axis=1),epsilon)
    rCS_pred = np.maximum(np.linalg.norm(positions_C_pred - positions_S_pred,axis=1),epsilon)
    rOS_pred = np.maximum(np.linalg.norm(positions_O_pred - positions_S_pred,axis=1),epsilon)

    PE_pred = (4/rCO_pred)+(4/rCS_pred)+(4/rOS_pred)
    EnergyDiff = ((KE_total - PE_pred)/(np.abs(KE_total)+epsilon))**2
    EnergyDiff_mean = np.mean(EnergyDiff)

    print(f'Average MRE: {MRE:.4f}%')
    print(f'Average MSE: {MSE:.6f}')
    print(f'Average Energy Difference: {EnergyDiff_mean:.6e}')

    comboIndex = combos.index[0]
    hs_str = json.dumps(HIDDEN_DIMS)
    runtime = end_time - start_time

    # Prepare header and line for results.csv
    headers = [
        "TaskID","latent_dim","epochs","batch_size","lr","activation","num_hidden_layers","hidden_layer_sizes",
        "MSE_WEIGHT","KLD_WEIGHT","MRE2_WEIGHT","ENERGY_DIFF_WEIGHT",
        "MRE","MSE","EnergyDiff","runtime_seconds"
    ]
    header_line = ",".join(headers) + "\n"
    line = (
        f"\"{comboIndex}\",\"{combo['latent_dim']}\",\"{combo['epochs']}\",\"{combo['batch_size']}\","
        f"\"{combo['lr']}\",\"{combo['activation']}\",\"{combo['num_hidden_layers']}\",\"{hs_str}\","
        f"\"{MSE_WEIGHT}\",\"{KLD_WEIGHT}\",\"{MRE2_WEIGHT}\",\"{ENERGY_DIFF_WEIGHT}\","
        f"\"{MRE:.4f}\",\"{MSE:.6f}\",\"{EnergyDiff_mean:.6e}\",\"{runtime:.4f}\"\n"
    )

    results_path = os.path.join(main_job_dir,'results.csv')
    with open(results_path,'a') as rf:
        fcntl.flock(rf, fcntl.LOCK_EX)
        # Check if file is empty (no header)
        if os.stat(results_path).st_size == 0:
            rf.write(header_line)
        rf.write(line)
        fcntl.flock(rf, fcntl.LOCK_UN)

    S = SAMPLES_TO_PRINT
    indices = np.random.choice(len(x_test_np), S, replace=False)
    with open(os.path.join(main_job_dir,'sample_predictions.txt'),'w') as f:
        for idx in indices:
            f.write(f'Sample {idx}:\n')
            f.write('Real Positions:\n')
            f.write(f'Carbon (C): ({x_test_np[idx,0]:.4f}, {x_test_np[idx,1]:.4f}, {x_test_np[idx,2]:.4f})\n')
            f.write(f'Oxygen (O): ({x_test_np[idx,3]:.4f}, {x_test_np[idx,4]:.4f}, {x_test_np[idx,5]:.4f})\n')
            f.write(f'Sulfur (S): ({x_test_np[idx,6]:.4f}, {x_test_np[idx,7]:.4f}, {x_test_np[idx,8]:.4f})\n')
            f.write('Predicted Positions:\n')
            f.write(f'Carbon (C): ({x_pred[idx,0]:.4f}, {x_pred[idx,1]:.4f}, {x_pred[idx,2]:.4f})\n')
            f.write(f'Oxygen (O): ({x_pred[idx,3]:.4f}, {x_pred[idx,4]:.4f}, {x_pred[idx,5]:.4f})\n')
            f.write(f'Sulfur (S): ({x_pred[idx,6]:.4f}, {x_pred[idx,7]:.4f}, {x_pred[idx,8]:.4f})\n')
            f.write('---\n')

        for idx in indices:
            print(f'Sample {idx}:')
            print('Real Positions:')
            print(f'Carbon (C): ({x_test_np[idx,0]:.4f}, {x_test_np[idx,1]:.4f}, {x_test_np[idx,2]:.4f})')
            print(f'Oxygen (O): ({x_test_np[idx,3]:.4f}, {x_test_np[idx,4]:.4f}, {x_test_np[idx,5]:.4f})')
            print(f'Sulfur (S): ({x_test_np[idx,6]:.4f}, {x_test_np[idx,7]:.4f}, {x_test_np[idx,8]:.4f})')
            print('Predicted Positions:')
            print(f'Carbon (C): ({x_pred[idx,0]:.4f}, {x_pred[idx,1]:.4f}, {x_pred[idx,2]:.4f})')
            print(f'Oxygen (O): ({x_pred[idx,3]:.4f}, {x_pred[idx,4]:.4f}, {x_pred[idx,5]:.4f})')
            print(f'Sulfur (S): ({x_pred[idx,6]:.4f}, {x_pred[idx,7]:.4f}, {x_pred[idx,8]:.4f})')
            print('---')

print("All tasks completed successfully.")

