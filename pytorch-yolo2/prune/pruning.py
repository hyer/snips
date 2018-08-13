import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# this code implements pruning using register buffer feature to save input mask

def compute_mask(weights):
    thresh = weights.std()
    m1 = weights > thresh
    m2 = weights < (-thresh)
    mask = torch.ones(weights.size())
    mask = mask - m1.float()
    mask = mask - m2.float()
    return mask


class PrunedSqueezenet(nn.Module):
    def __init__(self, to_prune, pretrained_weight):
        """
        takes a list of layers to prune, model, weights
        to_prune: a list of all the layers on which pruning should be applied
        model: architecture of the model
        weights: pretrained weights to use for the model
        """
        super(self, PrunedSqueezenet).__init__()
        self.to_prune = to_prune
        # get the model ready
        self.base_model = model.SqueezeNet()
        pretrained_weights = torch.load(pretrained_weight)
        base_model.load_state_dict(pretrained_weights)

        self.layers = self.base_model._modules.keys()
        # compute the mask for the weights
        for l in to_prune:
            if "fire" in l:
                curr_layer = self.base_model._modules.get(l)._modules.get('conv3')
                weights = curr_layer.weight.data
                # save the mask
                curr_layer.register_buffer('mask', compute_mask(weights))
                # change the computed output of conv3 layer in the fire
                curr_layer.register_forward_hook(
                    lambda m, i, o: print "Hello this is ok")


            elif "conv" in l:
                curr_layer = self.base_model._modules.get(l)
                weights = curr_layer.weight.data
                # save the mask
                curr_layer.register_buffer('mask', compute_mask(weights))
                # change the computed output of conv3 layer in the fire
                curr_layer.register_forward_hook(
                    lambda m, i, o: \
                        print("Hello this is ok"))
                )
            else:
                print("I dont understand what you are talking about")

    def forward(self, x):
        return self.base_model(x)


if __name__ == '__main__':
    net = PrunedSqueezenet(to_prune=['fire9'], pretrained_weight='pretrained_models/squeezedet_compatible.pth')
    x = Variable(torch.randn(1, 3, 32, 32))
    print(net(x))