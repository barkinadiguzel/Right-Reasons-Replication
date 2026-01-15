import torch

class ExplanationLoss:
    def __init__(self, lambda1):
        self.lambda1 = lambda1

    def __call__(self, gradients, mask):
        penalty = (mask * gradients.pow(2)).sum()
        return self.lambda1 * penalty
