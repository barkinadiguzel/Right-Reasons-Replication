import torch

class MaskGenerator:
    def generate(self, x, forbidden_idx):
        A = torch.zeros_like(x)
        A[:, forbidden_idx] = 1.0
        return A
