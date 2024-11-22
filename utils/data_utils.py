# utils/data_utils.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

def prepare_standard_nn_data(data, feature_columns, target_column, task_type):
    try:
        X = data[feature_columns]
        y = data[target_column]
    except KeyError as e:
        print(f"Error: Column not found in the dataset: {e}")
        return None

    # Encode categorical features
    X = pd.get_dummies(X)

    # Feature scaling
    X = X.values.astype(np.float32)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)

    # Process target variable
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
    if len(dataset) < 2:
        print("Error: The dataset is too small to split into training and testing sets.")
        return None
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    X_train, y_train = zip(*train_dataset)
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)

    X_test, y_test = zip(*test_dataset)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    return X_train, y_train, X_test, y_test

def prepare_cvae_data(data, input_columns, condition_columns):
    try:
        input_data = data[input_columns]
        condition_data = data[condition_columns]
    except KeyError as e:
        print(f"Error: Column not found in the dataset: {e}")
        return None

    # Convert to numpy arrays
    input_data = input_data.values
    condition_data = condition_data.values

    # Split data into train, validation, test sets
    from sklearn.model_selection import train_test_split

    train_input, temp_input, train_condition, temp_condition = train_test_split(
        input_data, condition_data, test_size=0.3, random_state=85, shuffle=True
    )
    val_input, test_input, val_condition, test_condition = train_test_split(
        temp_input, temp_condition, test_size=0.5, random_state=42, shuffle=True
    )

    return train_input, val_input, test_input, train_condition, val_condition, test_condition

