import torch.nn as nn

# If you want to see what's inside the model, uncomment the code below
# for name, param in model.named_parameters():
#    print(f"Layer: {name} Size: {param.size()}  Values: {param[:2]} \n")

# Locally connected layer class. PyTorch does not have an implementation of such a layer, so we implement the logic ourselves
class LocallyConnected2D(nn.Module):
  def __init__(self, in_channels, out_channels, input_height, input_width, kernel=3, stride=1):
    super().__init__()
    # Kernel size
    self.kernel = kernel

    # Stride step
    self.stride = stride

    # There should also be padding here to allow preserving the image size, as in "real" CNNs, but for our educational example it'll do
    self.out_height = (input_height - kernel) // stride + 1
    self.out_width = (input_width - kernel) // stride + 1

    # Kaiming initialization. You could also use the more straightforward nn.init.kaiming_normal_,
    # but for our non-standard architecture additional code would be needed
    self.weight = nn.Parameter(torch.randn(self.out_height, self.out_width, out_channels, in_channels * kernel ** 2) * (2.0 / (in_channels * kernel ** 2)) ** 0.5)

    # bias should be initialized with zeros
    self.bias = nn.Parameter(torch.zeros(self.out_height, self.out_width, out_channels))

    # If the code below isn't entirely clear — don't worry, this is just a concept
    # intended to demonstrate that behind every PyTorch method lies concrete mathematics,
    # and that sometimes you can implement the functionality you need bypassing the standard approaches
  def forward(self, x):
    # Slide a kernel × kernel window across height (axis 2) and width (axis 3) with the given stride
    # We get all patches — cropped image fragments matching the kernel size, to which weights will be applied
    patches = x.unfold(2, self.kernel, self.stride).unfold(3, self.kernel, self.stride)

    # Rearrange axes: move spatial coordinates (h, w) to the second and third positions,
    # and channels and patch pixels to the end. contiguous() rearranges data in memory to match the new order
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    # Flatten channels and pixels of each patch into one long vector
    # Now each patch is simply a list of numbers ready to be multiplied by weights
    patches = patches.view(x.size(0), self.out_height, self.out_width, -1)

    # For each position (h, w), multiply its patch by its own weights — not shared across the entire map
    # This is exactly where local connectivity lies: each position has its own independent set of weights
    out = torch.einsum('bhwi,hwoi->bhwo', patches, self.weight) + self.bias

    # Return channels to the second position — bring the tensor back to the standard format (batch, channels, height, width)
    # This is needed so that the next layer receives data in the axis order expected by PyTorch
    return out.permute(0, 3, 1, 2)


