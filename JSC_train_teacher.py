#!/usr/bin/python3
import argparse
import os
import sys
import math
import random
import numpy as np

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

from models.cls_resnet50 import resnet50

from utils.MyImageFolder import MyDataset
from utils.losses import CrossEntropyLabelSmooth
from utils.utils import accuracy, save_checkpoint, Logger
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log', help='save model root')
parser.add_argument('--name', type=str, default='log_srnet_resnet50', help='train name')
parser.add_argument('--arch_sr', type=str, default='srnet', help='super-resolution name')
parser.add_argument('--arch_cls', type=str, default='resnet50', help='classification name')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
parser.add_argument('--size_src', type=int, default=224, help='the size of the input hr image')
parser.add_argument('--size_dst', type=int, default=56, help='the size of the lr image')
parser.add_argument('--batchSize', type=int, default=12, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./train_test', help='root directory of the dataset')
parser.add_argument('--lr_cls', type=float, default=0.01, help='initial cls learning rate')
parser.add_argument('--weight_decay', type=float, default=10 ** -5, help='weight decay (default: 1e-4)')
parser.add_argument('--cuda', type=str, default='3,2', help='use GPU computation')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
writer = SummaryWriter(os.path.join(opt.logdir, opt.name))

sys.stdout = Logger(os.path.join(opt.logdir, opt.name, 'log_train.txt'))
print(opt)


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(2022)

### Data processing
print(opt.name)
aug_size = opt.size_src // 7
org_size = opt.size_src + aug_size
src_size = opt.size_src
dst_size = opt.size_dst
transform_so_train = T.Compose(
    [
        T.Resize((org_size, org_size), Image.BICUBIC),
        T.RandomCrop((src_size, src_size), padding=8),
        T.RandomHorizontalFlip(),
    ])
transform_so_test = T.Compose(
    [
        T.Resize((org_size, org_size), Image.BICUBIC),
        T.CenterCrop((src_size, src_size)),
    ])
transform_mo = T.Compose(
    [
        T.Resize((dst_size, dst_size), Image.BICUBIC),
    ])
transform_eo = T.Compose(
    [
        T.ToTensor(),
    ])

train_path = os.path.join(opt.dataroot, 'train')
test_path = os.path.join(opt.dataroot, 'test')

trainset = MyDataset(train_path, transform_s=transform_so_train,
                     transform_e=transform_eo, transform_m=transform_mo)
train_loader = DataLoader(trainset, batch_size=opt.batchSize,
                          shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

testset = MyDataset(test_path, transform_s=transform_so_test,
                    transform_e=transform_eo, transform_m=transform_mo)
test_loader = DataLoader(testset, batch_size=opt.batchSize,
                         shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
dataset_num_classes = len(trainset.classes)

# CLS Networks
print(opt.arch_cls)
if opt.arch_cls == 'resnet50':
    cls_model = resnet50(num_classes=dataset_num_classes)
elif opt.arch_cls == 'resnet18':
    from models.cls_resnet18 import resnet18
    cls_model = resnet18(num_classes=dataset_num_classes)
elif opt.arch_cls == 'shufflenetv2':
    from models.shufflenetv2 import shufflenetv2
    cls_model = shufflenetv2(num_classes=dataset_num_classes)
else:
    print('cls_model:%s is not find.'%(opt.arch_cls))


# SR Networks
print(opt.arch_sr)
if opt.arch_sr == 'srnet':
    from models.SRGAN_pretrained import SRResNet
    sr_model = SRResNet(upscale_factor=4, in_channels=3, out_channels=3, channels=64, num_rcb=16)
    checkpoint = torch.load(
        'pretrained/SRResNet_x4-ImageNet-6dd5216c.pth.tar',
        map_location=lambda storage, loc: storage)
    sr_model.load_state_dict(checkpoint["state_dict"])

elif opt.arch_sr == 'swinir':
    from models.SwinIR_pretrained import SwinIR
    sr_model = SwinIR(upscale=4, img_size=(64, 64),
                      window_size=8, img_range=1., depths=[6, 6, 6, 6],
                      embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    checkpoint = torch.load(
        'pretrained/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
        map_location=lambda storage, loc: storage)
    sr_model.load_state_dict(checkpoint["params"])
else:
    print('sr_model:%s is not find.'%(opt.arch_sr))


if opt.cuda:
    sr_model = sr_model.cuda()
    cls_model = cls_model.cuda()
    sr_model = torch.nn.DataParallel(sr_model)
    cls_model = torch.nn.DataParallel(cls_model)

# Lossess
criterion_C = CrossEntropyLabelSmooth(num_classes=dataset_num_classes, use_gpu=True)

# Optimizers & LR schedulers
cls_optimizer = optim.SGD(cls_model.parameters(), lr=opt.lr_cls, momentum=0.9, weight_decay=opt.weight_decay)
cls_scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_optimizer, opt.n_epochs)


def test(epoch, test_loader):
    cls_model.eval()

    out_hr = []
    out_lr = []
    label = []

    top1hr = 0
    top5hr = 0
    top1sr = 0
    top5sr = 0
    with torch.no_grad():
        for iter, (imhr, imlr, targets) in enumerate(test_loader):
            if opt.cuda:
                imhr, imlr, targets = imhr.cuda(), imlr.cuda(), targets.cuda()
            imsr = sr_model(imlr)
            outputslr = cls_model(imsr)
            outputshr = cls_model(imhr)

            out_hr.append(outputshr)
            out_lr.append(outputslr)
            label.append(targets)

            prec1hr, prec5hr = accuracy(outputshr.data, targets.data, topk=(1, 5))
            top1hr += prec1hr
            top5hr += prec5hr

            prec1lr, prec5lr = accuracy(outputslr.data, targets.data, topk=(1, 5))
            top1sr += prec1lr
            top5sr += prec5lr

            print('[epcho:%d][%d/%d]' % (epoch, iter+1, len(test_loader)))
            print('HR-HR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1hr / (iter+1), 100 * top5hr / (iter+1)))
            print('SR-SR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1sr / (iter+1), 100 * top5sr / (iter+1)))

        out_hr2 = torch.cat(out_hr, dim=0)
        out_sr2 = torch.cat(out_lr, dim=0)
        label2 = torch.cat(label, dim=0)

        top1hr, top5hr = accuracy(out_hr2.data, label2.data, topk=(1, 5))
        top1sr, top5sr = accuracy(out_sr2.data, label2.data, topk=(1, 5))
        print('Final HR-HR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1hr, 100 * top5hr))
        print('Final SR-SR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1sr, 100 * top5sr))

        writer.add_scalar('data/TestTop1SR', top1sr, epoch)
        writer.add_scalars('data/TestTopAcc', {'HR-HR Top1': top1hr,
                                               'HR-HR Top5': top5hr,
                                               'SR-SR Top1': top1sr,
                                               'SR-SR Top5': top5sr
                                               }, epoch)
    return top1hr


def train(epoch, train_loader):
    cls_model.train()
    cls_loss = 0.0
    top1 = 0
    top5 = 0

    for iter, (imhr, imlr, train_label) in enumerate(train_loader):
        if opt.cuda:
            imhr, imlr, train_label = imhr.cuda(), imlr.cuda(), train_label.cuda()

        cls_optimizer.zero_grad()
        imsr = sr_model(imlr)
        outsr = cls_model(imsr)
        loss = criterion_C(outsr, train_label)
        cls_loss += loss
        loss.backward()
        cls_optimizer.step()

        # measure accuracy
        prec1, prec5 = accuracy(outsr.data, train_label.data, topk=(1, 5))
        top1 += prec1
        top5 += prec5

        lr = 0
        for param_group in cls_optimizer.param_groups:
            lr = param_group['lr']
            break

        print('[epcho:%d][%d/%d][LR:%.5f] CLoss:%.3f' % (
            epoch, iter+1, len(train_loader), lr, cls_loss / (iter+1)))
        print('SR-SR Top1: %.3f%%|Top5: %.3f%%' % (100*top1/(iter+1), 100*top5/(iter+1)))

    writer.add_scalar('data/TrainLoss', cls_loss / len(train_loader), epoch)
    writer.add_scalars('data/TrainTopAcc', {'SR-SR Top1': top1/len(train_loader),
                                            'SR-SR Top5': top5/len(train_loader)}, epoch)

    # Update learning rates
    cls_scheduler.step()

    return top1


# def main_train():
print('**********************Training**********************')
best_acc = 0
for epoch in range(opt.epoch, opt.n_epochs):
    train(epoch, train_loader)
    acc = test(epoch, test_loader)

    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint(cls_model.state_dict(),
                    os.path.join(opt.logdir, opt.name),
                    is_best, name=opt.name+'_base')
