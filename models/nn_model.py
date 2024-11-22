# models/nn_model.py

import torch
import torch.nn as nn

class StandardNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, task_type):
        super(StandardNN, self).__init__()
        self.task_type = task_type
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def get_loss_function(loss_function_name, task_type):
    if task_type == 'Regression':
        loss_fn = getattr(nn, loss_function_name)()
    else:
        loss_fn = getattr(nn, loss_function_name)()
    return loss_fn

