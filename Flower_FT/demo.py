import Image
import json

import time
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import torchvision.models as models


normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def preprocess(image, transformer):
    x = transformer(image)
    return Variable(x.unsqueeze(0))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    name_file = "/home/hyer/workspace/algo/FlowerR/data/cls_label.txt"
    with open(name_file, "r") as f:
        data = f.readlines()

    label_name_chi = []
    for li in data:
        line = li.strip().split()
        chi_name = line[2]
        label_name_chi.append(chi_name)


    arch = 'resnet50'
    num_classes = 531
    checkpoint_file = "/home/hyer/workspace/algo/FlowerR/model/resnet18-c531/model_best.pth.tar"


    # print('Loading index-class map')
    # idx_to_class = "/home/hyer/workspace/algo/model_comporession/ShuffleNet/pytorch_impl/imagenet/imagenet_class_index-chi.json"
    # with open(idx_to_class, 'r') as f:
    #     mapping = json.load(f)

    # original_model = models.__dict__[arch](pretrained=True)

    # net = models.__dict__[arch]()
    # net = torch.nn.DataParallel(net).cuda()

    # print("=> loading checkpoint ")
    # checkpoint = torch.load(checkpoint_file)
    # net.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint")

    # create model

    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = model_ft
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint.")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    while True:
        img_path = raw_input("Input Image Path: ")
        # args.image = img_path
        # image = Image.open(img_path)
        # print('Preprocessing')
        # # image transformer
        # transformer = get_transformer()
        # x = preprocess(image, transformer)
        # x = x.cuda()
        image = Image.open(img_path).convert('RGB')
        if test_transform is not None:
            image = test_transform(image)
        inputs = image
        inputs = Variable(inputs, volatile=True)
        inputs = inputs.cuda()
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

        # predict output
        # print('Inferring on image {}'.format(image))

        start = time.time()
        outputs = model(inputs)
        softmax_res = softmax(outputs.data.cpu().numpy()[0])
        score_np = np.array(softmax_res)
        ind = np.argsort(score_np)
        topk = list(ind[-5:])  # top5
        topk.reverse()

        for idx in topk:
            print idx, "      ", idx, "      ", label_name_chi[idx], softmax_res[idx]

        # y = model(x)
        # print "test time: ", time.time() - start
        # top_idxs = np.argsort(y.data.cpu().numpy().ravel()).tolist()[-5:][::-1]
        # print('==========================================')
        # print "idx_in_namefile == cls_idx == cls_name"
        # for i, idx in enumerate(top_idxs):
        #     key = str(idx)
        #     # class_name = mapping[key][1]
        #     print idx, "            ", idx, "            ", label_name_chi[idx], y.data.cpu().numpy().ravel()[idx]
        #     # print i + 1, class_name
        #     # print('{}.\t{}'.format(i + 1, class_name))
        # print('==========================================')