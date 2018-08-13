import torch
from torch.autograd import Variable
from torch.utils import model_zoo
import torch.onnx

from darknet import Darknet


cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/split_sqrt_416x416/000005.weights"

torch_model = Darknet(cfgfile)
torch_model.print_network()
torch_model.load_weights(weightfile)


# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage


# set the train mode to false since we will only run the forward pass.
torch_model.train(False)

# Input to the model
x = Variable(torch.randn(batch_size, 1, 800, 800), requires_grad=True)

# Export the model
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "super_resolution.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file