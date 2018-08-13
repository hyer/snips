import torch

loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
input = torch.autograd.Variable(torch.randn(3,4))
target = torch.autograd.Variable(torch.randn(3,4))
loss = loss_fn(input, target)
print(input); print(target); print(loss)
print(input.size(), target.size(), loss.size())