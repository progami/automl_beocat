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
import time
import traceback
import optuna
from datetime import datetime
import sqlalchemy as sa

from models.cvae_model import CVAE, loss_function_cvae, get_activation_function

def print_with_time(message: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")

def ensure_study_created(main_job_dir, task_id, study_name, storage):
    # Use a lock file to ensure only task_id=0 creates the study
    lock_file = os.path.join(main_job_dir, 'study_created.lock')

    if task_id == 0:
        if not os.path.exists(lock_file):
            # create the study
            study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
            with open(lock_file,'w') as f:
                f.write("study created\n")
        else:
            study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        while not os.path.exists(lock_file):
            print_with_time("Waiting for study to be created by task_id=0...")
            time.sleep(2)
        study = optuna.load_study(study_name=study_name, storage=storage)

    return study

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Training script for hyperparameter tuning.')
    parser.add_argument('--task_id', type=int, required=True, help='Task ID')
    parser.add_argument('--total_tasks', type=int, required=True, help='Total number of tasks')
    parser.add_argument('--main_job_dir', type=str, required=True, help='Path to the main job directory')
    args = parser.parse_args()

    task_id = args.task_id
    total_tasks = args.total_tasks
    main_job_dir = args.main_job_dir

    print_with_time(f"Starting training for task_id={task_id} in {main_job_dir}...")

    data_path = os.path.join(main_job_dir, 'data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    params_path = os.path.join(main_job_dir, 'params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
    selected_model = params['selected_model']
    print_with_time(f"Model selected: {selected_model}")

    study_info_path = os.path.join(main_job_dir, 'study_info.json')
    with open(study_info_path, 'r') as f:
        study_info = json.load(f)
    study_name = study_info['study_name']
    study_db_path = os.path.join(main_job_dir, 'study.db')

    print_with_time("Connecting to Optuna study and enabling WAL mode...")
    engine = sa.create_engine(f"sqlite:///{study_db_path}")
    with engine.connect() as conn:
        conn.execute(sa.text("PRAGMA journal_mode=WAL;"))

    storage = f"sqlite:///{study_db_path}"
    study = ensure_study_created(main_job_dir, task_id, study_name, storage)

    try:
        trial = study.ask()

        if selected_model == 'Standard Neural Network':
            nn_hparams = params['nn_hparams']
            X_train, y_train, X_test, y_test = data
            task_type = params['task_type']
            metrics = run_nn_trial(trial, X_train, y_train, X_test, y_test, task_type, nn_hparams)
        else:
            cvae_hparams = params['cvae_hparams']
            train_input, val_input, test_input, train_condition, val_condition, test_condition = data
            metrics = run_cvae_trial(trial, train_input, val_input, test_input, train_condition, val_condition, test_condition, cvae_hparams)

        if 'MSE' in metrics:
            study.tell(trial, metrics['MSE'])
        elif 'Accuracy' in metrics:
            study.tell(trial, 1.0 - metrics['Accuracy'])
        else:
            study.tell(trial, 0.0)

        end_time = time.time()
        runtime_seconds = end_time - start_time
        metrics['runtime_seconds'] = runtime_seconds
        write_results(main_job_dir, task_id, {}, metrics)
        print_with_time(f"Task {task_id} completed and results saved. Runtime: {runtime_seconds:.2f} s")

    except Exception as e:
        print_with_time(f"An error occurred during training: {e}")
        traceback.print_exc()
        end_time = time.time()
        runtime_seconds = end_time - start_time
        error_metrics = {
            'error': str(e),
            'runtime_seconds': runtime_seconds
        }
        if 'trial' in locals():
            study.tell(trial, float('inf'))
        write_results(main_job_dir, task_id, {}, error_metrics)
        raise e

def run_nn_trial(trial, X_train, y_train, X_test, y_test, task_type, nn_hparams):
    learning_rate = trial.suggest_categorical("learning_rate", nn_hparams['lr_set'])
    batch_size = trial.suggest_categorical("batch_size", nn_hparams['batch_sizes'])
    epochs = trial.suggest_categorical("epochs", nn_hparams['epochs'])
    hidden_size = trial.suggest_categorical("hidden_size", nn_hparams['hidden_sizes'])
    optimizer_name = trial.suggest_categorical("optimizer", nn_hparams['optimizers'])
    loss_function_name = trial.suggest_categorical("loss_function", nn_hparams['loss_functions'])

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[1]
    if task_type == 'Regression':
        output_size = 1
    else:
        output_size = len(torch.unique(y_train))

    model = StandardNN(input_size, hidden_size, output_size, task_type)
    criterion = select_criterion(task_type, loss_function_name)
    optimizer = select_optimizer(model, optimizer_name, learning_rate)

    print_with_time(f"Training NN trial: lr={learning_rate}, bs={batch_size}, epochs={epochs}, hs={hidden_size}, opt={optimizer_name}, loss={loss_function_name}")
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
        print_with_time(f"NN Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

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
            'R2': r2,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'hidden_size': hidden_size,
            'optimizer': optimizer_name,
            'loss_function': loss_function_name
        }
    else:
        y_pred = all_outputs.argmax(dim=1).numpy()
        y_true = all_targets.numpy()
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {
            'Accuracy': accuracy,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'hidden_size': hidden_size,
            'optimizer': optimizer_name,
            'loss_function': loss_function_name
        }
    return metrics

def run_cvae_trial(trial, train_input, val_input, test_input, train_condition, val_condition, test_condition, cvae_hparams):
    learning_rate = trial.suggest_categorical("LEARNING_RATE", cvae_hparams['lr_set'])
    latent_dim = trial.suggest_categorical("LATENT_DIM", cvae_hparams['latent_dims'])
    epochs = trial.suggest_categorical("EPOCHS", cvae_hparams['epochs'])
    batch_size = trial.suggest_categorical("BATCH_SIZE", cvae_hparams['batch_sizes'])
    activation_name = trial.suggest_categorical("activation_name", cvae_hparams['activations'])
    num_hidden_layers = trial.suggest_categorical("num_hidden_layers", cvae_hparams['num_hidden_layers'])
    hidden_layer_size = trial.suggest_categorical("hidden_layer_size", cvae_hparams['hidden_layer_sizes'])

    if cvae_hparams['l1_set']:
        L1_LAMBDA = trial.suggest_categorical("L1_LAMBDA", cvae_hparams['l1_set'])
        use_l1 = (L1_LAMBDA != 0.0)
    else:
        L1_LAMBDA = 0.0
        use_l1 = False

    if cvae_hparams['l2_set']:
        L2_LAMBDA = trial.suggest_categorical("L2_LAMBDA", cvae_hparams['l2_set'])
        use_l2 = (L2_LAMBDA != 0.0)
    else:
        L2_LAMBDA = 0.0
        use_l2 = False

    train_dataset = TensorDataset(torch.tensor(train_input).float(), torch.tensor(train_condition).float())
    test_dataset = TensorDataset(torch.tensor(test_input).float(), torch.tensor(test_condition).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_input.shape[1]
    condition_dim = train_condition.shape[1]

    model = CVAE(input_dim, condition_dim, latent_dim, activation_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print_with_time(f"Training CVAE trial: LATENT_DIM={latent_dim}, EPOCHS={epochs}, BATCH_SIZE={batch_size}, LR={learning_rate}, activation={activation_name}, layers={num_hidden_layers}, hsize={hidden_layer_size}, use_l1={use_l1}, L1_LAMBDA={L1_LAMBDA}, use_l2={use_l2}, L2_LAMBDA={L2_LAMBDA}")
    for epoch in range(epochs):
        model.train()
        for x, c in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            loss = loss_function_cvae(recon_x, x, mu, logvar)
            if use_l1:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss += L1_LAMBDA * l1_loss
            if use_l2:
                l2_loss = sum((p**2).sum() for p in model.parameters())
                loss += L2_LAMBDA * l2_loss
            loss.backward()
            optimizer.step()
        print_with_time(f"CVAE Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    model.eval()
    reconstructions = []
    originals = []
    with torch.no_grad():
        for x, c in test_loader:
            recon_x, mu, logvar = model(x, c)
            if torch.isnan(recon_x).any():
                print_with_time("NaN detected in recon_x, replacing with 0")
                recon_x = torch.nan_to_num(recon_x)
            reconstructions.append(recon_x)
            originals.append(x)

    reconstructions = torch.cat(reconstructions)
    originals = torch.cat(originals)

    y_pred = reconstructions.numpy()
    y_true = originals.numpy()

    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        print_with_time("NaN detected in predictions or targets, replacing with 0")
        y_pred = np.nan_to_num(y_pred)
        y_true = np.nan_to_num(y_true)

    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = np.mean(np.abs(y_true - y_pred))

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'LATENT_DIM': latent_dim,
        'EPOCHS': epochs,
        'BATCH_SIZE': batch_size,
        'LEARNING_RATE': learning_rate,
        'activation_name': activation_name,
        'num_hidden_layers': num_hidden_layers,
        'hidden_layer_size': hidden_layer_size,
        'use_l1': use_l1,
        'L1_LAMBDA': L1_LAMBDA,
        'use_l2': use_l2,
        'L2_LAMBDA': L2_LAMBDA
    }

    return metrics

def write_results(main_job_dir, task_id, hyperparams, metrics):
    results_csv_path = os.path.join(main_job_dir, 'results.csv')
    lock_path = results_csv_path + '.lock'

    results_row = hyperparams.copy()
    results_row.update(metrics)
    results_row['Task ID'] = task_id

    from filelock import FileLock
    with FileLock(lock_path):
        if not os.path.exists(results_csv_path):
            results_df = pd.DataFrame(columns=list(results_row.keys()))
            results_df.to_csv(results_csv_path, index=False)
        else:
            results_df = pd.read_csv(results_csv_path)

        new_row_df = pd.DataFrame([results_row])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        results_df.to_csv(results_csv_path, index=False)

    print_with_time(f"Results for Task {task_id} written to {results_csv_path}")

def select_criterion(task_type, loss_name):
    if task_type == 'Regression':
        if loss_name == 'MSELoss':
            return nn.MSELoss()
        elif loss_name == 'L1Loss':
            return nn.L1Loss()
        elif loss_name == 'SmoothL1Loss':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")
    else:
        if loss_name == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif loss_name == 'NLLLoss':
            return nn.NLLLoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")

def select_optimizer(model, opt_name, lr):
    if opt_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer: {opt_name}")

class StandardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, task_type):
        super(StandardNN, self).__init__()
        self.task_type = task_type
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        if self.task_type == 'Classification':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.task_type == 'Classification':
            out = self.softmax(out)
        return out

if __name__ == '__main__':
    main()

