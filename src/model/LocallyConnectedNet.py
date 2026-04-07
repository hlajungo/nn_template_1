import torch.nn as nn


class LocallyConnectedNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.lc1 = LocallyConnected2d(1, 16, 28, 28)
    self.lc2 = LocallyConnected2d(16, 32, 13, 13)
    self.fc1 = nn.Linear(32 * 5 * 5, 128)  #  "Regular" fully connected layer for image classification 800 -> 128
    self.fc2 = nn.Linear(128, 10)  # "Regular" fully connected output layer 128 -> 10

  def forward(self, x):
    x = self.lc1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)  # Take the maximum in each 2×2 window, size is halved. We won't write max_pool2d ourselves; it's a standard PyTorch layer

    # There may be pixel loss here with odd image dimensions (due to integer division in the LocallyConnected2d code), but for MNIST everything works
    x = self.lc2(x)         # Second locally connected layer, 32 output channels
    x = F.relu(x)
    x = F.max_pool2d(x, 2)  # Again halve the size (11//2 = 5)

    x = x.flatten(1)        # Flatten 32×5×5 into a flat tensor of length 800, merging all tensor axes except the zeroth
    x = self.fc1(x)         # 128 fully connected neurons, 800 → 128
    x = F.relu(x)

    x = self.fc2(x)         # Fully connected layer 128 → 10
    return x                # Return logits (Softmax will be applied later, when computing the loss function)


