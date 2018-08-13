# coding=utf-8
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES, config
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time

print("######################")
print("MUST running with python3.6 !!!!!!!!!")
print("MUST running till 120000 iter !!!!!!!!!")
print("######################")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='SSD OCR-MER Training')
# parser.add_argument('--version', default='v2', help='network setting')
parser.add_argument('--basenet', default='mobilenet_feature.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size for training')
# parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--resume', default="./models/mobv1-ssd-k1/T16184/VOC_mobile_300_19105321_150000.pth", type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()
version = config.VOC_mobile_300_19105321
save_interval = 10000
gpus = [0, 1]
bg_label = 0
ngpus = len(gpus)
# cfg = COCO_mobile_300   # network setting, 可参考Caffe-mobilenet-ssd的设置

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('trainval')]
# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (128, 128, 128)  # only support voc now
num_classes = len(VOC_CLASSES) + 1
# num_classes = 200 + 1
batch_size = args.batch_size
# accum_batch_size = 32
# iter_size = accum_batch_size / batch_size
max_iter = args.iterations
weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
gamma = 0.1
momentum = 0.9
log_inter = 50
conf_thresh = 0.01 # default 0.01
nms_thresh = 0.45

if args.visdom:
    import visdom

    viz = visdom.Visdom(env="SSD OCR-MER")

ssd_net = build_ssd('train', version, 300, num_classes, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
net = ssd_net

if args.cuda and ngpus > 1:
    net = torch.nn.DataParallel(ssd_net)
    # cudnn.benchmark = True
elif args.cuda:
    net = net.cuda()

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    pretrained_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.base.load_state_dict(pretrained_weights)

if ngpus > 1:
    cur_model = net.module
else:
    cur_model = net


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, bg_label, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    lr = args.lr
    logging.info('Loading Dataset...')

    dataset = VOCDetection(args.voc_root,
                           train_sets,
                           SSDAugmentation(ssd_dim, means),
                           AnnotationTransform(),
                           dataset_name="VOC_OCR_K1_GAN")

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )

        lr_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='lr',
                title='Iteration SSD learning rate',
                # legend=['lr']
            )
        )

    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                                    loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % log_inter == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]))
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                    win=lot,
                    update='append'
                )
                viz.line(
                    X=torch.ones((1,)).cpu() * iteration,
                    Y=torch.Tensor([lr, ]).cpu(),
                    win=lr_lot,
                    update='append'
                )

                # hacky fencepost solution for 0th epoch plot
                if iteration == 0:
                    viz.line(
                        X=torch.zeros((1, 3)).cpu(),
                        Y=torch.Tensor([loc_loss, conf_loss,
                                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                        win=epoch_lot,
                        update=True
                    )
        if iteration != 0 and iteration % save_interval == 0:
            print('Saving state, iter:', iteration)
            torch.save(cur_model.state_dict(), args.save_folder + version['name'] + '_' + repr(iteration) + '.pth')
            print("Save done.")
    torch.save(cur_model.state_dict(), args.save_folder + version['name'] + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
