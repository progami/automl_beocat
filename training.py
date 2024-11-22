import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        data = pickle.load(f)
    with open(os.path.join(main_job_dir, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    selected_model = params.get('selected_model', 'Standard Neural Network')

    # Load hyperparameter combinations
    with open(os.path.join(main_job_dir, 'combinations.pkl'), 'rb') as f:
        combinations = pickle.load(f)

    # Get the hyperparameter combination for this task
    hyperparams = combinations[task_id]

    if selected_model == 'Standard Neural Network':
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
        task_type = params['task_type']
        X_train, y_train, X_test, y_test = data

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

    elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
        # Unpack hyperparameters specific to CVAE
        hyperparams = hyperparams
        LATENT_DIM = hyperparams['LATENT_DIM']
        EPOCHS = hyperparams['EPOCHS']
        BATCH_SIZE = hyperparams['BATCH_SIZE']
        LEARNING_RATE = hyperparams['LEARNING_RATE']
        PATIENCE = hyperparams['PATIENCE']
        MIN_DELTA = hyperparams['MIN_DELTA']
        activation_name = hyperparams['activation_name']
        position_norm_method = hyperparams['position_norm_method']
        momenta_norm_method = hyperparams['momenta_norm_method']
        use_l1 = hyperparams['use_l1']
        L1_LAMBDA = hyperparams['L1_LAMBDA']
        use_l2 = hyperparams['use_l2']
        L2_LAMBDA = hyperparams['L2_LAMBDA']
        num_hidden_layers = hyperparams['num_hidden_layers']
        hidden_layer_size = hyperparams['hidden_layer_size']

        print(f"Hyperparameters for Task ID {task_id}:")
        print(hyperparams)

        # Proceed with the CVAE training procedure
        # Data preparation
        train_input, val_input, test_input, train_condition, val_condition, test_condition = data

        # Prepare position data with optional normalization
        train_position_norm, position_scaler = prepare_data(train_input, position_norm_method)
        val_position_norm = (torch.FloatTensor(position_scaler.transform(val_input))
                             if position_scaler else torch.FloatTensor(val_input))
        test_position_norm = (torch.FloatTensor(position_scaler.transform(test_input))
                              if position_scaler else torch.FloatTensor(test_input))

        # Prepare condition data with optional normalization
        train_momenta_norm, momenta_scaler = prepare_data(train_condition, momenta_norm_method)
        val_momenta_norm = (torch.FloatTensor(momenta_scaler.transform(val_condition))
                            if momenta_scaler else torch.FloatTensor(val_condition))
        test_momenta_norm = (torch.FloatTensor(momenta_scaler.transform(test_condition))
                             if momenta_scaler else torch.FloatTensor(test_condition))

        # Create data loaders
        train_loader = DataLoader(TensorDataset(train_position_norm, train_momenta_norm),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_position_norm, val_momenta_norm),
                                batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_position_norm, test_momenta_norm),
                                 batch_size=BATCH_SIZE, shuffle=False)

        # Set up the CVAE model and trainer
        input_dim = train_position_norm.shape[1]
        condition_dim = train_momenta_norm.shape[1]
        activation_function = getattr(nn, activation_name)()
        hidden_layers = [hidden_layer_size * (2 ** i) for i in range(num_hidden_layers)]

        model = CVAE(input_dim, LATENT_DIM, condition_dim, hidden_layers, activation_function)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using device: {device}")

        # Train and evaluate
        trainer = CVAETrainer(model, optimizer, hyperparams, device)
        metrics, train_losses, val_losses = trainer.train_and_evaluate(
            train_loader, val_loader, test_loader,
            train_input, val_input, test_input,
            position_scaler
        )

        # Prepare results
        results = {
            'Task ID': task_id,
            'Hyperparameters': hyperparams,
            'Metrics': metrics,
            'Train Losses': train_losses,
            'Validation Losses': val_losses
        }

        # Save results specific to this task
        results_dir = os.path.join(main_job_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'result_{task_id}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to '{results_path}'.")

# Include CVAE and CVAETrainer classes from the professor's code
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
    if scaler_method == "StandardScaler":
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    elif scaler_method == "None":
        scaler = None
        data_normalized = data
    return torch.FloatTensor(data_normalized), scaler

if __name__ == '__main__':
    main()

