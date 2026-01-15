import torch
import torch.nn as nn

class ReasoningModel(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)
