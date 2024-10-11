# Save this code as automl_app.py

import streamlit as st
import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    st.title("AutoML App")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Please upload your dataset (CSV format):", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully.")

        # Select the target column
        target_column = st.selectbox("Select the target (label) column:", data.columns)

        if target_column:
            # Select the task type
            task_type = st.radio("Is this a regression or classification task?", ('Regression', 'Classification'))

            # Select algorithms to train
            if task_type == 'Regression':
                algorithms = st.multiselect(
                    "Select algorithms to train:",
                    ['Linear Regression'],
                    default=['Linear Regression']
                )
            else:
                algorithms = st.multiselect(
                    "Select algorithms to train:",
                    ['Logistic Regression'],
                    default=['Logistic Regression']
                )

            if algorithms:
                if st.button("Start Training"):
                    # Preprocess the data
                    X = data.drop(columns=[target_column])
                    y = data[target_column]

                    # Encode categorical features using pandas
                    X = pd.get_dummies(X)

                    # Feature scaling using PyTorch
                    X = X.values.astype(np.float32)
                    X_mean = X.mean(axis=0)
                    X_std = X.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
                    X = (X - X_mean) / X_std

                    # Convert to PyTorch tensors
                    X = torch.tensor(X, dtype=torch.float32)

                    # Process the target variable
                    if task_type == 'Classification':
                        # Encode target labels
                        if y.dtype == 'object':
                            y = pd.Categorical(y).codes
                        else:
                            y = y.astype(int)
                        y = torch.tensor(y.values, dtype=torch.long)
                    else:
                        y = y.values.reshape(-1, 1).astype(np.float32)
                        y = torch.tensor(y, dtype=torch.float32)

                    # Split the data
                    dataset = TensorDataset(X, y)
                    train_size = int(0.8 * len(dataset))
                    test_size = len(dataset) - train_size
                    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

                    X_train, y_train = train_dataset[:]
                    X_test, y_test = test_dataset[:]

                    # Initialize a list to store results
                    results = []

                    # Iterate over the selected algorithms
                    for algorithm in algorithms:
                        with st.spinner(f"Training {algorithm}..."):
                            best_model, best_params = optimize_model(algorithm, task_type, X_train, y_train)
                            y_pred = best_model(X_test).detach()

                            st.success(f"Training completed for {algorithm}.")

                            st.subheader(f"{algorithm} - Best Hyperparameters:")
                            st.write(best_params)

                            if task_type == 'Regression':
                                # Evaluation metrics
                                y_pred_np = y_pred.numpy()
                                y_test_np = y_test.numpy()

                                mse = np.mean((y_pred_np - y_test_np) ** 2)
                                rmse = np.sqrt(mse)
                                ss_total = np.sum((y_test_np - y_test_np.mean()) ** 2)
                                ss_res = np.sum((y_test_np - y_pred_np) ** 2)
                                r2 = 1 - (ss_res / ss_total)

                                st.subheader(f"{algorithm} - Regression Results:")
                                st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                                st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                                st.write(f"R-squared (R²): {r2:.4f}")

                                # Append results to the list
                                results.append({
                                    'Algorithm': algorithm,
                                    'MSE': mse,
                                    'RMSE': rmse,
                                    'R2': r2,
                                    'Best Params': best_params
                                })
                            else:
                                # Convert logits to class labels
                                y_pred_labels = torch.argmax(y_pred, dim=1)
                                y_test_labels = y_test

                                # Evaluation metrics
                                accuracy = (y_pred_labels == y_test_labels).sum().item() / len(y_test_labels)
                                report = classification_report_numpy(y_test_labels.numpy(), y_pred_labels.numpy())
                                conf_matrix = confusion_matrix_numpy(y_test_labels.numpy(), y_pred_labels.numpy())

                                st.subheader(f"{algorithm} - Classification Results:")
                                st.write(f"Accuracy: {accuracy:.4f}")

                                st.subheader("Classification Report:")
                                st.dataframe(pd.DataFrame(report).transpose())

                                st.subheader("Confusion Matrix:")
                                st.write(conf_matrix)

                                # Append results to the list
                                results.append({
                                    'Algorithm': algorithm,
                                    'Accuracy': accuracy,
                                    'Best Params': best_params
                                })

                    # Display a summary table of results
                    st.subheader("Summary:")
                    if task_type == 'Regression':
                        st.dataframe(pd.DataFrame(results).sort_values(by='RMSE'))
                    else:
                        st.dataframe(pd.DataFrame(results).sort_values(by='Accuracy', ascending=False))
                else:
                    st.warning("Please click 'Start Training' to begin.")
            else:
                st.warning("Please select at least one algorithm to train.")
        else:
            st.warning("Please select a target column.")
    else:
        st.info("Awaiting for CSV file to be uploaded.")

def optimize_model(algorithm, task_type, X_train, y_train):
    def objective(trial):
        # Common hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 50, 200)

        if algorithm == 'Linear Regression':
            # Hyperparameter: fit_intercept
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])

            # Define model
            model = LinearRegressionModel(X_train.shape[1], fit_intercept)

            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            # Loss function
            loss_fn = nn.MSELoss()

        elif algorithm == 'Logistic Regression':
            # Hyperparameter: C (Inverse of regularization strength)
            C = trial.suggest_loguniform('C', 1e-4, 1e2)

            # Define model
            num_classes = len(torch.unique(y_train))
            model = LogisticRegressionModel(X_train.shape[1], num_classes)

            # Optimizer with weight decay as regularization
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1.0 / C)

            # Loss function
            loss_fn = nn.CrossEntropyLoss()
        else:
            return None

        # DataLoader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
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
                return accuracy  # Maximize accuracy

    # Create an Optuna study and optimize
    if task_type == 'Regression':
        study = optuna.create_study(direction='maximize')
    else:
        study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # Train the final model with the best hyperparameters
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']

    if algorithm == 'Linear Regression':
        fit_intercept = best_params['fit_intercept']
        model = LinearRegressionModel(X_train.shape[1], fit_intercept)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
    elif algorithm == 'Logistic Regression':
        C = best_params['C']
        num_classes = len(torch.unique(y_train))
        model = LogisticRegressionModel(X_train.shape[1], num_classes)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1.0 / C)
        loss_fn = nn.CrossEntropyLoss()
    else:
        return None, {}

    # DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model.eval(), best_params

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

def classification_report_numpy(y_true, y_pred):
    # Calculate precision, recall, f1-score
    classes = np.unique(y_true)
    report = {}
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = np.sum(y_true == cls)
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    # Add accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    report['accuracy'] = {'precision': accuracy}
    return report

def confusion_matrix_numpy(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            matrix[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))
    return matrix

if __name__ == '__main__':
    main()

