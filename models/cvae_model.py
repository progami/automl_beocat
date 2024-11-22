# models/cvae_model.py

import torch
import torch.nn as nn
import numpy as np

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function):
        super(CVAE, self).__init__()

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        encoder_layers.append(activation_function)
        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(activation_function)
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
        decoder_layers.append(activation_function)
        for i in reversed(range(len(hidden_layers) - 1)):
            decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
            decoder_layers.append(activation_function)
        decoder_layers.append(nn.Linear(hidden_layers[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        combined = torch.cat((z, condition), dim=1)
        return self.decoder(combined)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

class CVAETrainer:
    def __init__(self, model, optimizer, params, device):
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.device = device

    def compute_metrics(self, loader, position_data, position_scaler, name):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, batch_cond in loader:
                batch_x = batch_x.to(self.device)
                batch_cond = batch_cond.to(self.device)
                recon_x, _, _ = self.model(batch_x, batch_cond)
                predictions.append(recon_x.cpu())

        predictions = torch.cat(predictions, dim=0)

        # Convert tensor to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(position_data, torch.Tensor):
            position_data = position_data.cpu().numpy()

        if position_scaler is not None:
            predictions_inv = position_scaler.inverse_transform(predictions)
            targets_inv = position_scaler.inverse_transform(position_data)
        else:
            predictions_inv = predictions
            targets_inv = position_data

        relative_errors = np.abs(predictions_inv - targets_inv) / (np.abs(targets_inv) + 1e-8)
        mre = np.mean(relative_errors)
        mse = np.mean((predictions_inv - targets_inv) ** 2)

        return {f"{name}_mre": float(mre), f"{name}_mse": float(mse)}

    def loss_fn(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= x.size(0) * x.size(1)
        loss = recon_loss + kl_divergence

        if self.params['use_l1']:
            l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
            loss += self.params['L1_LAMBDA'] * l1_loss

        if self.params['use_l2']:
            l2_loss = sum(torch.sum(param ** 2) for param in self.model.parameters())
            loss += self.params['L2_LAMBDA'] * l2_loss

        return loss

    def train_and_evaluate(self, train_loader, val_loader, test_loader,
                          train_position, val_position, test_position,
                          position_scaler):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.params['EPOCHS']):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_cond in train_loader:
                batch_x = batch_x.to(self.device)
                batch_cond = batch_cond.to(self.device)
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(batch_x, batch_cond)
                loss = self.loss_fn(recon_x, batch_x, mu, logvar)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_cond in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_cond = batch_cond.to(self.device)
                    recon_x, mu, logvar = self.model(batch_x, batch_cond)
                    loss = self.loss_fn(recon_x, batch_x, mu, logvar)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss - self.params['MIN_DELTA']:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.params['PATIENCE']:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Evaluation
        metrics = {}
        metrics.update(self.compute_metrics(train_loader, train_position, position_scaler, "train"))
        metrics.update(self.compute_metrics(val_loader, val_position, position_scaler, "val"))
        metrics.update(self.compute_metrics(test_loader, test_position, position_scaler, "test"))

        return metrics, train_losses, val_losses

def prepare_data(data, scaler_method):
    """
    Prepare data with optional normalization
    """
    from sklearn.preprocessing import StandardScaler
    if scaler_method == "StandardScaler":
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    elif scaler_method == "None":
        scaler = None
        data_normalized = data
    return torch.FloatTensor(data_normalized), scaler

