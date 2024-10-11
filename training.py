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

    # Train the specified algorithm
    print(f"Training {algorithm}...")
    best_model, best_params = optimize_model(algorithm, task_type, X_train, y_train)
    print(f"Training completed for {algorithm}.")

    # Evaluate the model
    if task_type == 'Regression':
        # Evaluation metrics
        if algorithm in ['Linear Regression', 'Neural Network']:
            y_pred = best_model(X_test).detach().numpy()
        else:
            y_pred = best_model.predict(X_test.numpy())
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
        if algorithm in ['Logistic Regression', 'Neural Network']:
            y_pred = best_model(X_test).detach()
            y_pred_labels = torch.argmax(y_pred, dim=1).numpy()
        else:
            y_pred_labels = best_model.predict(X_test.numpy())
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

def optimize_model(algorithm, task_type, X_train, y_train):
    import optuna

    def objective(trial):
        # Define hyperparameters for each algorithm
        if algorithm == 'Linear Regression':
            # Hyperparameter: fit_intercept
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])

            # Define model
            model = LinearRegressionModel(X_train.shape[1], fit_intercept)

            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # Loss function
            loss_fn = nn.MSELoss()

        elif algorithm == 'Logistic Regression':
            # Hyperparameter: C (Inverse of regularization strength)
            C = trial.suggest_loguniform('C', 1e-4, 1e2)

            # Define model
            num_classes = len(torch.unique(y_train))
            model = LogisticRegressionModel(X_train.shape[1], num_classes)

            # Optimizer with weight decay as regularization
            optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1.0 / C)

            # Loss function
            loss_fn = nn.CrossEntropyLoss()

        elif algorithm == 'Random Forest Regressor':
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 20)

            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train.numpy(), y_train.numpy().ravel())
            y_pred = model.predict(X_train.numpy())
            mse = np.mean((y_pred - y_train.numpy().ravel()) ** 2)
            return -mse  # Negative MSE for maximization

        elif algorithm == 'Random Forest Classifier':
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 20)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train.numpy(), y_train.numpy())
            accuracy = model.score(X_train.numpy(), y_train.numpy())
            return accuracy

        elif algorithm == 'Neural Network':
            hidden_size = trial.suggest_int('hidden_size', 10, 100)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            epochs = trial.suggest_int('epochs', 50, 200)

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

            # Training loop
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = loss_fn(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_train)
                if task_type == 'Regression':
                    val_loss = loss_fn(val_outputs, y_train).item()
                    return -val_loss  # Negative MSE for maximization
                else:
                    val_preds = torch.argmax(val_outputs, dim=1)
                    accuracy = (val_preds == y_train).sum().item() / len(y_train)
                    return accuracy
        else:
            return None

        # For PyTorch models, evaluate on training data
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_train)
            if task_type == 'Regression':
                val_loss = loss_fn(val_outputs, y_train).item()
                return -val_loss  # Negative MSE for maximization
            else:
                val_preds = torch.argmax(val_outputs, dim=1)
                accuracy = (val_preds == y_train).sum().item() / len(y_train)
                return accuracy

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # Train the final model with the best hyperparameters
    if algorithm in ['Linear Regression', 'Logistic Regression', 'Neural Network']:
        # Extract hyperparameters
        if algorithm == 'Linear Regression':
            fit_intercept = best_params['fit_intercept']
            model = LinearRegressionModel(X_train.shape[1], fit_intercept)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            epochs = 100  # Set a default value
        elif algorithm == 'Logistic Regression':
            C = best_params['C']
            num_classes = len(torch.unique(y_train))
            model = LogisticRegressionModel(X_train.shape[1], num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1.0 / C)
            loss_fn = nn.CrossEntropyLoss()
            epochs = 100  # Set a default value
        elif algorithm == 'Neural Network':
            hidden_size = best_params['hidden_size']
            learning_rate = best_params['learning_rate']
            epochs = best_params['epochs']

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
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
    else:
        # For sklearn models
        if algorithm == 'Random Forest Regressor':
            n_estimators = best_params['n_estimators']
            max_depth = best_params['max_depth']
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train.numpy(), y_train.numpy().ravel())
        elif algorithm == 'Random Forest Classifier':
            n_estimators = best_params['n_estimators']
            max_depth = best_params['max_depth']
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train.numpy(), y_train.numpy())

    return model, best_params

def classification_report_numpy(y_true, y_pred):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

def confusion_matrix_numpy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_true, y_pred)
    return matrix

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, fit_intercept=True):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=fit_intercept)

    def forward(self, x):
        return self.linear(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

if __name__ == '__main__':
    main()

