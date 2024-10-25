import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training script for AutoML Job Array')
    parser.add_argument('--task_id', type=int, required=True, help='SLURM Array Task ID')
    parser.add_argument('--total_tasks', type=int, required=True, help='Total number of tasks')
    parser.add_argument('--main_job_dir', type=str, required=True, help='Main job directory where data and params are stored')
    args = parser.parse_args()

    task_id = args.task_id
    main_job_dir = args.main_job_dir

    print(f"Task ID: {task_id}")
    print(f"Main job directory: {main_job_dir}")

    # Load data and parameters
    with open(os.path.join(main_job_dir, 'data.pkl'), 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    with open(os.path.join(main_job_dir, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    task_type = params['task_type']

    # Load hyperparameter combinations
    with open(os.path.join(main_job_dir, 'combinations.pkl'), 'rb') as f:
        combinations = pickle.load(f)

    # Get the hyperparameter combination for this task
    hyperparams = combinations[task_id]

    # Unpack hyperparameters
    learning_rate, batch_size, epochs, hidden_size, optimizer_name, loss_function_name = hyperparams

    print(f"Hyperparameters for Task ID {task_id}:")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Loss Function: {loss_function_name}")

    # Proceed with training using these hyperparameters
    if task_type == 'Regression':
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        loss_fn = getattr(nn, loss_function_name)()
    else:
        num_classes = len(torch.unique(y_train))
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        loss_fn = getattr(nn, loss_function_name)()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
    else:
        device = torch.device('cpu')

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Default

    # DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

    # Training loop
    model.train()
    for epoch in range(int(epochs)):
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

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        if task_type == 'Regression':
            y_pred = outputs.cpu().numpy()
            y_true = y_test.cpu().numpy()

            mse = np.mean((y_pred - y_true) ** 2)
            rmse = np.sqrt(mse)
            ss_total = np.sum((y_true - y_true.mean()) ** 2)
            ss_res = np.sum((y_true - y_pred) ** 2)
            r2 = 1 - (ss_res / ss_total)

            print(f"Task ID {task_id} - Regression Results:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")

            # Prepare results
            results = {
                'Task ID': task_id,
                'Hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'hidden_size': hidden_size,
                    'optimizer': optimizer_name,
                    'loss_function': loss_function_name
                },
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }

        else:
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            y_true = y_test.cpu().numpy()

            accuracy = np.mean(y_pred == y_true)

            print(f"Task ID {task_id} - Classification Results:")
            print(f"Accuracy: {accuracy:.4f}")

            # Prepare results
            results = {
                'Task ID': task_id,
                'Hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'hidden_size': hidden_size,
                    'optimizer': optimizer_name,
                    'loss_function': loss_function_name
                },
                'Accuracy': accuracy
            }

    # Save results specific to this task
    results_dir = os.path.join(main_job_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'result_{task_id}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to '{results_path}'.")

if __name__ == '__main__':
    main()

