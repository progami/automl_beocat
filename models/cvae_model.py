# models/cvae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_function(name: str):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'Sigmoid':
        return nn.Sigmoid()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f"Invalid activation function: {name}")

def loss_function_cvae(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, activation_name):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)
        self.fc4 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)
        self.activation = get_activation_function(activation_name)

    def encode(self, x, c):
        h1 = self.activation(self.fc1(torch.cat([x, c], dim=1)))
        h2 = self.activation(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h4 = self.activation(self.fc4(torch.cat([z, c], dim=1)))
        h5 = self.activation(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

