import torch
import torch.nn as nn

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

