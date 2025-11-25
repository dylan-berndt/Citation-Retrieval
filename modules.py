import torch
import torch.nn as nn


class MAP(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(predRelevance, targetRelevance):
        pass


class InfoNCE(nn.Module):
    def __init__(self, axis=1, temperature=0.03):
        super().__init__()
        self.entropy = nn.CrossEntropyLoss()

        self.axis = axis
        self.temperature = temperature

    def forward(self, y1, y2):
        y1 = nn.functional.normalize(y1, axis=self.axis)
        y2 = nn.functional.normalize(y2, axis=self.axis)

        logits = y1 @ y2.t()

        labels = torch.arange(len(y1))

        return self.entropy(logits / self.temperature, labels)