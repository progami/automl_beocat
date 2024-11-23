# training.py

import argparse
import os
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from filelock import FileLock
import numpy as np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training script for hyperparameter tuning.')
    parser.add_argument('--task_id', type=int, required=True, help='Task ID from SLURM_ARRAY_TASK_ID')
    parser.add_argument('--total_tasks', type=int, required=True, help='Total number of tasks')
    parser.add_argument('--main_job_dir', type=str, required=True, help='Path to the main job directory')
    args = parser.parse_args()

    task_id = args.task_id
    total_tasks = args.total_tasks
    main_job_dir = args.main_job_dir

    # Load hyperparameter combinations
    combinations_path = os.path.join(main_job_dir, 'combinations.json')
    with open(combinations_path, 'r') as f:
        combinations = json.load(f)

    # Get the hyperparameters for this task
    hyperparams = combinations[task_id]

    # Load data
    data_path = os.path.join(main_job_dir, 'data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Unpack data based on the model
    params_path = os.path.join(main_job_dir, 'params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
    selected_model = params['selected_model']

    if selected_model == 'Standard Neural Network':
        X_train, y_train, X_test, y_test = data
        task_type = params['task_type']
        # Training code for Standard Neural Network
        metrics = train_standard_nn(X_train, y_train, X_test, y_test, hyperparams, task_type)
    elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
        train_input, val_input, test_input, train_condition, val_condition, test_condition = data
        # Training code for CVAE
        metrics = train_cvae(train_input, val_input, test_input, train_condition, val_condition, test_condition, hyperparams)
    else:
        raise ValueError(f"Unknown model type: {selected_model}")

    # Save results to results.csv using file locking to prevent race conditions
    results_csv_path = os.path.join(main_job_dir, 'results.csv')
    lock_path = results_csv_path + '.lock'

    # Prepare the results row
    results_row = hyperparams.copy()
    results_row.update(metrics)
    results_row['Task ID'] = task_id

    with FileLock(lock_path):
        if not os.path.exists(results_csv_path):
            # Create the CSV file with headers
            results_df = pd.DataFrame(columns=list(results_row.keys()))
            results_df.to_csv(results_csv_path, index=False)
        else:
            results_df = pd.read_csv(results_csv_path)

        # Append the new results
        results_df = results_df.append(results_row, ignore_index=True)
        results_df.to_csv(results_csv_path, index=False)

    print(f"Task {task_id} completed and results saved.")

def train_standard_nn(X_train, y_train, X_test, y_test, hyperparams, task_type):
    # Unpack hyperparameters
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    epochs = hyperparams['epochs']
    hidden_size = hyperparams['hidden_size']
    optimizer_name = hyperparams['optimizer']
    loss_function_name = hyperparams['loss_function']

    # Prepare data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    input_size = X_train.shape[1]
    if task_type == 'Regression':
        output_size = 1
    else:  # Classification
        output_size = len(torch.unique(y_train))

    model = StandardNN(input_size, hidden_size, output_size, task_type)

    # Define loss function
    if task_type == 'Regression':
        if loss_function_name == 'MSELoss':
            criterion = nn.MSELoss()
        elif loss_function_name == 'L1Loss':
            criterion = nn.L1Loss()
        elif loss_function_name == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Invalid loss function: {loss_function_name}")
    else:  # Classification
        if loss_function_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif loss_function_name == 'NLLLoss':
            criterion = nn.NLLLoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_function_name}")

    # Define optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if task_type == 'Regression':
                targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    if task_type == 'Regression':
        y_pred = all_outputs.numpy().flatten()
        y_true = all_targets.numpy().flatten()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    else:  # Classification
        y_pred = all_outputs.argmax(dim=1).numpy()
        y_true = all_targets.numpy()
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {
            'Accuracy': accuracy
        }

    return metrics

def train_cvae(train_input, val_input, test_input, train_condition, val_condition, test_condition, hyperparams):
    # Unpack hyperparameters
    LATENT_DIM = hyperparams['LATENT_DIM']
    EPOCHS = hyperparams['EPOCHS']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    LEARNING_RATE = hyperparams['LEARNING_RATE']
    activation_name = hyperparams['activation_name']

    # Prepare data loaders
    train_dataset = TensorDataset(torch.tensor(train_input).float(), torch.tensor(train_condition).float())
    val_dataset = TensorDataset(torch.tensor(val_input).float(), torch.tensor(val_condition).float())
    test_dataset = TensorDataset(torch.tensor(test_input).float(), torch.tensor(test_condition).float())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define the model
    input_dim = train_input.shape[1]
    condition_dim = train_condition.shape[1]

    model = CVAE(input_dim, condition_dim, LATENT_DIM, activation_name)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for x, c in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            loss = loss_function_cvae(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    reconstructions = []
    originals = []
    with torch.no_grad():
        for x, c in test_loader:
            recon_x, mu, logvar = model(x, c)
            reconstructions.append(recon_x)
            originals.append(x)

    reconstructions = torch.cat(reconstructions)
    originals = torch.cat(originals)

    y_pred = reconstructions.numpy()
    y_true = originals.numpy()

    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = np.mean(np.abs(y_true - y_pred))

    metrics = {
        'MSE': mse,
        'MAE': mae
    }

    return metrics

# Define the Standard Neural Network model
class StandardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, task_type):
        super(StandardNN, self).__init__()
        self.task_type = task_type
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        if self.task_type == 'Regression':
            self.fc2 = nn.Linear(hidden_size, output_size)
        else:  # Classification
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.task_type == 'Classification':
            out = self.softmax(out)
        return out

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, activation_name):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)  # Mean of the latent space
        self.fc32 = nn.Linear(256, latent_dim)  # Log variance of the latent space
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

def loss_function_cvae(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def get_activation_function(name):
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

if __name__ == '__main__':
    main()

