import torch.nn as nn

class Perceptron(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 800),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(800, 800),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(800, 10)
        )

  def forward(self, x):
    return self.layers(x)
