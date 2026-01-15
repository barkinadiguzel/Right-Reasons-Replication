import torch

class GradientExplainer:
    def compute(self, logits, x):
        log_probs = torch.log_softmax(logits, dim=1)
        score = log_probs.sum()

        grads = torch.autograd.grad(
            outputs=score,
            inputs=x,
            create_graph=True
        )[0]

        return grads
