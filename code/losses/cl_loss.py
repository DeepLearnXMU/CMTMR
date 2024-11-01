
from torch import nn

import torch

CE = nn.CrossEntropyLoss()

def contrastive_loss(v1, v2, smoothing=0):
  logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  num_classes = logits.shape[1]
  smooth_labels = torch.full_like(logits, smoothing / (num_classes - 1))
  smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing)
  return CE(logits, smooth_labels) + CE(torch.transpose(logits, 0, 1), smooth_labels)

BCEL = nn.BCEWithLogitsLoss()