class ForwardPipeline:
    def __init__(self, encoder, model, explainer, mask_gen, expl_loss, total_loss):
        self.encoder = encoder
        self.model = model
        self.explainer = explainer
        self.mask_gen = mask_gen
        self.expl_loss = expl_loss
        self.total_loss = total_loss

    def forward(self, x, y, forbidden_idx):
        x.requires_grad_(True)

        features = self.encoder(x)
        logits = self.model(features)

        gradients = self.explainer.compute(logits, x)
        mask = self.mask_gen.generate(x, forbidden_idx)

        explanation_loss = self.expl_loss(gradients, mask)
        total_loss = self.total_loss(logits, y, self.model, explanation_loss)

        return logits, gradients, mask, total_loss
