import torch
import torch.nn as nn
import torch.nn.functional as F

class SGDrop(nn.Module):
    def __init__(self, rho=0.1):
        super().__init__()
        self.rho = rho

    def forward(self, features, class_scores=None):
        # features: [B, C, H, W] veya [B, N] latent features
        if not self.training or class_scores is None:
            return features

        # Attribution map: gradient * features
        features.retain_grad()  # hold gradient
        grad = torch.autograd.grad(class_scores.sum(), features, create_graph=True)[0]
        attribution = F.relu(grad * features)

        flat_attr = attribution.view(attribution.size(0), -1)
        k = int(self.rho * flat_attr.size(1))
        topk_val, _ = torch.topk(flat_attr, k, dim=1)
        threshold = topk_val[:, -1].view(-1,1,1,1)
        mask = (attribution <= threshold).float()
        return features * mask
