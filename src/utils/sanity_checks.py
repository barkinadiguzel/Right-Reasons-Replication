import torch

def check_gradient_suppression(gradients, mask):
    forbidden_grad = (gradients * mask).abs().mean().item()
    allowed_grad = (gradients * (1-mask)).abs().mean().item()

    print("Forbidden region gradient:", forbidden_grad)
    print("Allowed region gradient:", allowed_grad)
