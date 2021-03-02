import torch
from torch_baidu_ctc import ctc_loss, CTCLoss

# Activations. Shape T x N x D.
# T -> max number of frames/timesteps
# N -> minibatch size
# D -> number of output labels (including the CTC blank)
x = torch.rand(10, 3, 6)
# Target labels
y = torch.tensor(
  [
    # 1st sample
    1, 1, 2, 5, 2,
    # 2nd
    1, 5, 2,
    # 3rd
    4, 4, 2, 3,
  ],
  dtype=torch.int,
)
# Activations lengths
xs = torch.tensor([10, 6, 9], dtype=torch.int)
# Target lengths
ys = torch.tensor([5, 3, 4], dtype=torch.int)

# By default, the costs (negative log-likelihood) of all samples are summed.
# This is equivalent to:
#   ctc_loss(x, y, xs, ys, average_frames=False, reduction="sum")
loss1 = ctc_loss(x, y, xs, ys)

# You can also average the cost of each sample among the number of frames.
# The averaged costs are then summed.
loss2 = ctc_loss(x, y, xs, ys, average_frames=True)

# Instead of summing the costs of each sample, you can perform
# other `reductions`: "none", "sum", or "mean"
#
# Return an array with the loss of each individual sample
losses = ctc_loss(x, y, xs, ys, reduction="none")
#
# Compute the mean of the individual losses
loss3 = ctc_loss(x, y, xs, ys, reduction="mean")
#
# First, normalize loss by number of frames, later average losses
loss4 = ctc_loss(x, y, xs, ys, average_frames=True, reduction="mean")


# Finally, there's also a nn.Module to use this loss.
ctc = CTCLoss(average_frames=True, reduction="mean", blank=0)
loss4_2 = ctc(x, y, xs, ys)

# Note: the `blank` option is also available for `ctc_loss`.
# By default it is 0.