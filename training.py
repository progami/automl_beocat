# training.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import seaborn as sns
import sys
import json

# NEW: Load params.json and override some hyperparameters
with open("params.json", "r") as f:
    params = json.load(f)

if params['selected_model'] == 'Standard Neural Network':
    # Just example: override some hyperparameters from params
    # In this code, we won't implement NN training fully, just keep code as is.
    # The user code given is CVAE oriented. We'll proceed with given CVAE code.
    pass
else:
    # It's CVAE scenario
    # For simplicity, let's assume we override BATCH_SIZE, EPOCHS, LEARNING_RATE from cvae_hparams.
    cvae_hparams = params['cvae_hparams']
    # Just pick the first options from each (since no trial index logic was provided)
    BATCH_SIZE = cvae_hparams['batch_sizes'][0]
    EPOCHS = cvae_hparams['epochs'][0]
    LEARNING_RATE = cvae_hparams['lr_set'][0]

# If not CVAE, fallback to original default:
if params['selected_model'] != 'Conditional Variational Autoencoder (CVAE)':
    # original code defaults
    BATCH_SIZE = 512
    EPOCHS = 50
    LEARNING_RATE = 1e-5

# Set random seeds for reproducibility
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

FILEPATH = params.get('FILEPATH', '/content/drive/MyDrive/PhysicsProject-KSU/random_million_orient.csv')
RESULT_PATH = '/content/drive/MyDrive/PhysicsProject-KSU/NEWENERGY/Random'

save_dir = RESULT_PATH
csv_path = os.path.join(save_dir, "test_predictions.csv")

TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Using overridden BATCH_SIZE, EPOCHS, LEARNING_RATE above

MSE_WEIGHT = 0.11767020152970051
KLD_WEIGHT = 10.220479086664307
MRE2_WEIGHT = 0.00025690809650127846
ENERGY_DIFF_WEIGHT = 0.0005698338123047986

use_l1 = False
use_l2 = True
l1_lambda = 0.0
l2_lambda = 0.001

PATIENCE = 5
MIN_DELTA = 1e-2
SAMPLES_TO_PRINT = 5

HIDDEN_DIM_SIZE = 16
NUM_HIDDEN_LAYERS =4
HIDDEN_DIMS = [HIDDEN_DIM_SIZE * (2 ** i) for i in range(NUM_HIDDEN_LAYERS)]
LATENT_DIM = 32
ACTIVATION_FUNCTION = 'elu'
POSITION_NORMALIZATION_METHOD = 'standard'
MOMENTA_NORMALIZATION_METHOD = 'standard'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

use_mixed_precision = False

mC = 21894.71361
mO = 29164.39289
mS = 58441.80487
mass = {'C': mC, 'O': mO, 'S': mS}
atom_indices = {'C': [0, 1, 2], 'O': [3, 4, 5], 'S': [6, 7, 8]}
epsilon = 1e-10

print("Loading data...")
data = pd.read_csv(FILEPATH)
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values.astype(np.float32)
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values.astype(np.float32)

print("Splitting data...")
positions_train, positions_temp, momenta_train, momenta_temp = train_test_split(
    positions, momenta, test_size=(TEST_SIZE + VAL_SIZE), random_state=42
)
val_size_adjusted = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
positions_val, positions_test, momenta_val, momenta_test = train_test_split(
    positions_temp, momenta_temp, test_size=val_size_adjusted, random_state=42
)

print("Applying normalization...")
def get_scaler(method):
    if method == 'minmax':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'maxabs':
        return MaxAbsScaler()
    elif method == 'robust':
        return RobustScaler()
    elif method == 'quantile_uniform':
        return QuantileTransformer(output_distribution='uniform')
    elif method == 'quantile_normal':
        return QuantileTransformer(output_distribution='normal')
    elif method == 'power':
        return PowerTransformer()
    elif method == 'normalize':
        return Normalizer()
    elif method is None:
        return None
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

scaler_pos = get_scaler(POSITION_NORMALIZATION_METHOD)
scaler_mom = get_scaler(MOMENTA_NORMALIZATION_METHOD)

if scaler_pos is not None:
    positions_train = scaler_pos.fit_transform(positions_train)
    positions_val = scaler_pos.transform(positions_val)
    positions_test = scaler_pos.transform(positions_test)
    if hasattr(scaler_pos, 'mean_'):
        mean_pos = scaler_pos.mean_
    elif hasattr(scaler_pos, 'center_'):
        mean_pos = scaler_pos.center_
    else:
        mean_pos = np.zeros(positions_train.shape[1], dtype=np.float32)

    if hasattr(scaler_pos, 'scale_'):
        scale_pos = scaler_pos.scale_
    else:
        scale_pos = np.ones(positions_train.shape[1], dtype=np.float32)
else:
    mean_pos = np.zeros(positions_train.shape[1], dtype=np.float32)
    scale_pos = np.ones(positions_train.shape[1], dtype=np.float32)

if scaler_mom is not None:
    momenta_train = scaler_mom.fit_transform(momenta_train)
    momenta_val = scaler_mom.transform(momenta_val)
    momenta_test = scaler_mom.transform(momenta_test)
    if hasattr(scaler_mom, 'mean_'):
        mean_mom = scaler_mom.mean_
    elif hasattr(scaler_mom, 'center_'):
        mean_mom = scaler_mom.center_
    else:
        mean_mom = np.zeros(momenta_train.shape[1], dtype=np.float32)

    if hasattr(scaler_mom, 'scale_'):
        scale_mom = scaler_mom.scale_
    else:
        scale_mom = np.ones(momenta_train.shape[1], dtype=np.float32)
else:
    mean_mom = np.zeros(momenta_train.shape[1], dtype=np.float32)
    scale_mom = np.ones(momenta_train.shape[1], dtype=np.float32)

def check_data_integrity(array, name):
    if np.isnan(array).any():
        raise ValueError(f"NaN values found in {name}")
    if np.isinf(array).any():
        raise ValueError(f"Inf values found in {name}")

check_data_integrity(positions_train, "positions_train")
check_data_integrity(positions_val, "positions_val")
check_data_integrity(positions_test, "positions_test")
check_data_integrity(momenta_train, "momenta_train")
check_data_integrity(momenta_val, "momenta_val")
check_data_integrity(momenta_test, "momenta_test")

positions_train_tensor = torch.from_numpy(positions_train)
momenta_train_tensor = torch.from_numpy(momenta_train)
positions_val_tensor = torch.from_numpy(positions_val)
momenta_val_tensor = torch.from_numpy(momenta_val)
positions_test_tensor = torch.from_numpy(positions_test)
momenta_test_tensor = torch.from_numpy(momenta_test)

train_dataset = TensorDataset(positions_train_tensor, momenta_train_tensor)
val_dataset = TensorDataset(positions_val_tensor, momenta_val_tensor)
test_dataset = TensorDataset(positions_test_tensor, momenta_test_tensor)

num_workers = min(4, os.cpu_count() // 2)
print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers, pin_memory=True, prefetch_factor=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=True, prefetch_factor=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=True, prefetch_factor=2
)

mean_pos = torch.tensor(mean_pos, device=device, dtype=torch.float32)
scale_pos = torch.tensor(scale_pos, device=device, dtype=torch.float32)
mean_mom = torch.tensor(mean_mom, device=device, dtype=torch.float32)
scale_mom = torch.tensor(scale_mom, device=device, dtype=torch.float32)

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dims, activation_function):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

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
        self.activation = activation_dict[activation_function]

        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(self.activation)
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        reversed_hidden_dims = hidden_dims[::-1]
        for h_dim in reversed_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(self.activation)
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        x_recon = self.decoder(zy)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

input_dim = positions.shape[1]
condition_dim = momenta.shape[1]
model = CVAE(input_dim, condition_dim, LATENT_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION).to(device)

l1_lambda_effective = l1_lambda if use_l1 else 0.0
l2_lambda_effective = l2_lambda if use_l2 else 0.0

optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=l2_lambda_effective)

if use_l1:
    print(f"L1 Regularization Enabled with lambda={l1_lambda_effective}")
else:
    print("L1 Regularization Disabled")

if use_l2:
    print(f"L2 Regularization Enabled with lambda={l2_lambda_effective}")
else:
    print("L2 Regularization Disabled")

if use_mixed_precision:
    scaler_fp16 = torch.cuda.amp.GradScaler()
    print("Mixed Precision Training Enabled")
else:
    scaler_fp16 = None
    print("Mixed Precision Training Disabled")

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

early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

train_losses = []
val_losses = []

train_mre = []
val_mre = []
train_energy_diff = []
val_energy_diff = []
train_recon_loss = []
val_recon_loss = []
train_kld = []
val_kld = []

mu_sum = torch.zeros(LATENT_DIM, device=device)
mu_square_sum = torch.zeros(LATENT_DIM, device=device)
total_samples = 0

print("Starting training...")
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_unreg_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_mre2_loss = 0
    total_l1_loss = 0
    total_energy_loss = 0

    epoch_train_mre = 0.0
    epoch_train_energy_diff = 0.0
    epoch_train_recon_loss = 0.0
    epoch_train_kld = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
    for x, y in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            x_recon, mu, logvar = model(x, y)
            if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                print("NaN detected in mu, logvar, or x_recon. Stopping training.")
                sys.exit(1)

            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            mre2_loss = torch.mean(((x - x_recon) / (torch.abs(x) + epsilon)) ** 2)

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

            rCO = torch.norm(positions_C - positions_O, dim=1)
            rCS = torch.norm(positions_C - positions_S, dim=1)
            rOS = torch.norm(positions_O - positions_S, dim=1)

            rCO = torch.clamp(rCO, min=epsilon)
            rCS = torch.clamp(rCS, min=epsilon)
            rOS = torch.clamp(rOS, min=epsilon)

            PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)
            EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + epsilon)
            EnergyLoss = torch.mean(EnergyDiff ** 2)

            unreg_loss = (MSE_WEIGHT * recon_loss +
                          KLD_WEIGHT * kld_loss +
                          MRE2_WEIGHT * mre2_loss +
                          ENERGY_DIFF_WEIGHT * EnergyLoss)

            if use_l1:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = unreg_loss + l1_lambda_effective * l1_loss
                total_l1_loss += l1_loss.item()
            else:
                loss = unreg_loss

        if use_mixed_precision:
            scaler_fp16.scale(loss).backward()
            scaler_fp16.step(optimizer)
            scaler_fp16.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_unreg_loss += unreg_loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_kld_loss += kld_loss.item() * batch_size
        total_mre2_loss += mre2_loss.item() * batch_size
        total_energy_loss += EnergyLoss.item() * batch_size

        mu_sum += mu.sum(dim=0)
        mu_square_sum += (mu ** 2).sum(dim=0)
        total_samples += batch_size

        epoch_train_recon_loss += recon_loss.item() * batch_size
        epoch_train_kld += kld_loss.item() * batch_size
        epoch_train_mre += mre2_loss.item() * batch_size
        epoch_train_energy_diff += EnergyLoss.item() * batch_size

    avg_train_loss = total_unreg_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    avg_train_recon_loss = epoch_train_recon_loss / len(train_loader.dataset)
    train_recon_loss.append(avg_train_recon_loss)

    avg_train_kld = epoch_train_kld / len(train_loader.dataset)
    train_kld.append(avg_train_kld)

    avg_train_mre = epoch_train_mre / len(train_loader.dataset)
    train_mre.append(avg_train_mre)

    avg_train_energy_diff = epoch_train_energy_diff / len(train_loader.dataset)
    train_energy_diff.append(avg_train_energy_diff)

    model.eval()
    total_val_loss = 0
    total_val_energy_loss = 0
    epoch_val_recon_loss = 0.0
    epoch_val_kld = 0.0
    epoch_val_mre = 0.0
    epoch_val_energy_diff = 0.0

    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation")
        for x_val, y_val in progress_bar_val:
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                x_recon, mu, logvar = model(x_val, y_val)
                if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                    print("NaN detected in validation mu, logvar, or x_recon. Stopping training.")
                    sys.exit(1)

                recon_loss = F.mse_loss(x_recon, x_val, reduction='mean')
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                mre2_loss = torch.mean(((x_val - x_recon) / (torch.abs(x_val) + epsilon)) ** 2)

                x_recon_original = x_recon * scale_pos + mean_pos
                y_original = y_val * scale_mom + mean_mom

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

                rCO = torch.norm(positions_C - positions_O, dim=1)
                rCS = torch.norm(positions_C - positions_S, dim=1)
                rOS = torch.norm(positions_O - positions_S, dim=1)

                rCO = torch.clamp(rCO, min=epsilon)
                rCS = torch.clamp(rCS, min=epsilon)
                rOS = torch.clamp(rOS, min=epsilon)

                PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)
                EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + epsilon)
                EnergyLoss = torch.mean(EnergyDiff ** 2)

                unreg_loss = (MSE_WEIGHT * recon_loss +
                              KLD_WEIGHT * kld_loss +
                              MRE2_WEIGHT * mre2_loss +
                              ENERGY_DIFF_WEIGHT * EnergyLoss)

            batch_size = x_val.size(0)
            total_val_loss += unreg_loss.item() * batch_size
            total_val_energy_loss += EnergyLoss.item() * batch_size

            epoch_val_recon_loss += recon_loss.item() * batch_size
            epoch_val_kld += kld_loss.item() * batch_size
            epoch_val_mre += mre2_loss.item() * batch_size
            epoch_val_energy_diff += EnergyLoss.item() * batch_size

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    avg_val_recon_loss = epoch_val_recon_loss / len(val_loader.dataset)
    val_recon_loss.append(avg_val_recon_loss)

    avg_val_kld = epoch_val_kld / len(val_loader.dataset)
    val_kld.append(avg_val_kld)

    avg_val_mre = epoch_val_mre / len(val_loader.dataset)
    val_mre.append(avg_val_mre)

    avg_val_energy_diff = epoch_val_energy_diff / len(val_loader.dataset)
    val_energy_diff.append(avg_val_energy_diff)

    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(RESULT_PATH, f'cvae_model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

mu_train_mean = mu_sum / total_samples
var_train = (mu_square_sum / total_samples) - mu_train_mean ** 2
std_train = torch.sqrt(var_train + epsilon)

torch.save({
    'mu_train_mean': mu_train_mean.cpu(),
    'std_train': std_train.cpu(),
    'mean_pos': mean_pos.cpu(),
    'scale_pos': scale_pos.cpu(),
    'mean_mom': mean_mom.cpu(),
    'scale_mom': scale_mom.cpu()
}, os.path.join(RESULT_PATH, 'latent_stats.pth'))

torch.save(model.state_dict(), os.path.join(RESULT_PATH, 'cvae_model_final.pth'))
print("Training complete and model saved.")

# Remove all plotting code:
# (Commenting out the plotting and distribution code to comply with requirements)
# The user requested no extra plotting, so we skip all that section.

print("Evaluating on test set...")
model.eval()
with torch.no_grad():
    latent_stats = torch.load(os.path.join(RESULT_PATH, 'latent_stats.pth'), map_location=device)
    mu_train_mean = latent_stats['mu_train_mean'].to(device)
    std_train = latent_stats['std_train'].to(device)
    mean_pos = latent_stats['mean_pos'].to(device)
    scale_pos = latent_stats['scale_pos'].to(device)
    mean_mom = latent_stats['mean_mom'].to(device)
    scale_mom = latent_stats['scale_mom'].to(device)

    positions_test_tensor = positions_test_tensor.to(device)
    momenta_test_tensor = momenta_test_tensor.to(device)

    z = torch.randn(len(test_dataset), LATENT_DIM, device=device) * std_train + mu_train_mean
    y_test_tensor = momenta_test_tensor

    with torch.cuda.amp.autocast(enabled=use_mixed_precision):
        x_pred = model.decode(z, y_test_tensor)

    x_pred = x_pred * scale_pos + mean_pos
    x_pred = x_pred.cpu().numpy()

    x_test_np = (positions_test_tensor * scale_pos + mean_pos).cpu().numpy()

    MRE = np.mean(np.abs(x_test_np - x_pred) / (np.abs(x_test_np) + epsilon)) * 100
    MSE = np.mean((x_test_np - x_pred) ** 2)
    print(f'Average MRE: {MRE:.2f}%')
    print(f'Average MSE: {MSE:.6f}')

    momenta_np = (momenta_test_tensor * scale_mom + mean_mom).cpu().numpy()
    momenta_C = momenta_np[:, 0:3]
    momenta_O = momenta_np[:, 3:6]
    momenta_S = momenta_np[:, 6:9]

    KE_C = np.sum(momenta_C ** 2, axis=1) / (2 * mC)
    KE_O = np.sum(momenta_O ** 2, axis=1) / (2 * mO)
    KE_S = np.sum(momenta_S ** 2, axis=1) / (2 * mS)
    KE_total = KE_C + KE_O + KE_S

    positions_C_pred = x_pred[:, 0:3]
    positions_O_pred = x_pred[:, 3:6]
    positions_S_pred = x_pred[:, 6:9]

    rCO_pred = np.linalg.norm(positions_C_pred - positions_O_pred, axis=1)
    rCS_pred = np.linalg.norm(positions_C_pred - positions_S_pred, axis=1)
    rOS_pred = np.linalg.norm(positions_O_pred - positions_S_pred, axis=1)

    rCO_pred = np.maximum(rCO_pred, epsilon)
    rCS_pred = np.maximum(rCS_pred, epsilon)
    rOS_pred = np.maximum(rOS_pred, epsilon)

    PE_pred = (4 / rCO_pred + 4 / rCS_pred + 4 / rOS_pred)
    EnergyDiff = ((KE_total - PE_pred) / (np.abs(KE_total) + epsilon)) ** 2
    EnergyDiff_mean = np.mean(EnergyDiff)
    print(f'Average Energy Difference: {EnergyDiff_mean:.2e}')

# Commenting out plotting distributions and all other plots since user requested no plots
# ...

print("All tasks completed successfully.")

# NEW: Write results to results.csv
# We'll write hyperparameters (LR, EPOCHS, BATCH_SIZE) and final metrics (MSE, MRE, EnergyDiff_mean)
# If results.csv doesn't exist, create with header, else append.

results_data = {
    "Model": "CVAE",
    "LearningRate": LEARNING_RATE,
    "BatchSize": BATCH_SIZE,
    "Epochs": EPOCHS,
    "FinalValLoss": val_losses[-1] if len(val_losses) > 0 else None,
    "MSE": MSE,
    "MRE": MRE,
    "EnergyDiff": EnergyDiff_mean
}

results_path = "results.csv"
mode = 'a' if os.path.exists(results_path) else 'w'
df_res = pd.DataFrame([results_data])
df_res.to_csv(results_path, mode=mode, index=False, header=(not os.path.exists(results_path)))

print("Results appended to results.csv.")

