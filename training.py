# training.py

import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training script for AutoML')
    parser.add_argument('--job_dir', type=str, required=True, help='Job directory where outputs are stored')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to train')
    parser.add_argument('--main_job_dir', type=str, required=True, help='Main job directory where data and params are stored')
    args = parser.parse_args()

    job_dir = args.job_dir
    algorithm = args.algorithm
    main_job_dir = args.main_job_dir
    print(f"Job directory: {job_dir}")
    print(f"Algorithm: {algorithm}")
    print(f"Main job directory: {main_job_dir}")

    # Load data and task_type from main_job_dir
    with open(os.path.join(main_job_dir, 'data.pkl'), 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    with open(os.path.join(main_job_dir, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    task_type = params['task_type']

    # Train the neural network
    print(f"Training {algorithm}...")
    best_model, best_params = optimize_model(task_type, X_train, y_train, X_test, y_test)
    print(f"Training completed for {algorithm}.")

    # Evaluate the model
    if task_type == 'Regression':
        # Evaluation metrics
        y_pred = best_model(X_test).detach().numpy()
        y_test_np = y_test.numpy()

        mse = np.mean((y_pred - y_test_np.ravel()) ** 2)
        rmse = np.sqrt(mse)
        ss_total = np.sum((y_test_np - y_test_np.mean()) ** 2)
        ss_res = np.sum((y_test_np - y_pred.reshape(-1, 1)) ** 2)
        r2 = 1 - (ss_res / ss_total)

        print(f"{algorithm} - Regression Results:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")

        # Prepare results
        results = {
            'Algorithm': algorithm,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Best Params': best_params
        }
    else:
        # Evaluation metrics
        y_pred = best_model(X_test).detach()
        y_pred_labels = torch.argmax(y_pred, dim=1).numpy()
        y_test_labels = y_test.numpy()

        accuracy = np.mean(y_pred_labels == y_test_labels)
        report = classification_report_numpy(y_test_labels, y_pred_labels)
        conf_matrix = confusion_matrix_numpy(y_test_labels, y_pred_labels)

        print(f"{algorithm} - Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")

        # Prepare results
        results = {
            'Algorithm': algorithm,
            'Accuracy': accuracy,
            'Best Params': best_params,
            'Classification Report': report,
            'Confusion Matrix': conf_matrix
        }

    # Save results
    with open(os.path.join(job_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print("Training completed. Results saved to 'results.pkl'.")

def optimize_model(task_type, X_train, y_train, X_test, y_test):
    import optuna

    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 10, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        epochs = trial.suggest_int('epochs', 50, 200)
        batch_size = trial.suggest_int('batch_size', 16, 128)

        if task_type == 'Regression':
            model = nn.Sequential(
                nn.Linear(X_train.shape[1], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            loss_fn = nn.MSELoss()
        else:
            num_classes = len(torch.unique(y_train))
            model = nn.Sequential(
                nn.Linear(X_train.shape[1], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            )
            loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # Optionally, print epoch loss
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            if task_type == 'Regression':
                val_loss = loss_fn(val_outputs, y_test).item()
                return val_loss  # MSE for minimization
            else:
                val_preds = torch.argmax(val_outputs, dim=1)
                accuracy = (val_preds == y_test).sum().item() / len(y_test)
                return 1.0 - accuracy  # 1 - accuracy for minimization

    # Create an Optuna study and optimize
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # Train the final model with the best hyperparameters
    hidden_size = best_params['hidden_size']
    learning_rate = best_params['learning_rate']
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']

    if task_type == 'Regression':
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        loss_fn = nn.MSELoss()
    else:
        num_classes = len(torch.unique(y_train))
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model, best_params

def classification_report_numpy(y_true, y_pred):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

def confusion_matrix_numpy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_true, y_pred)
    return matrix

if __name__ == '__main__':
    main()

