#!/usr/bin/python3
import argparse
import os
import sys
import random
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

from models.cls_resnet50 import resnet50
from models.cls_resnet18 import resnet18

from utils.MyImageFolder import MyDataset
from utils.utils import accuracy, Logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='res50', help='train name')
parser.add_argument('--arch_sr', type=str, default='srnet', help='super-resolution name')
parser.add_argument('--arch_cls', type=str, default='resnet50', help='classification name')
parser.add_argument('--size_src', type=int, default=224, help='the size of the input hr image')
parser.add_argument('--size_dst', type=int, default=56, help='the size of the lr image')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./CUB_200_2011/train_test', help='root directory of the dataset')
parser.add_argument('--cuda', type=str, default='1,0', help='use GPU computation')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)

sys.stdout = Logger(os.path.join('log_test', opt.name, 'log_test.txt'))
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
aug_size = opt.size_src // 7
org_size = opt.size_src + aug_size
src_size = opt.size_src
dst_size = opt.size_dst
transform_so_train = T.Compose(
    [
         T.Resize((org_size, org_size), Image.BICUBIC),
         T.RandomCrop((src_size, src_size)),
         T.RandomHorizontalFlip(),
    ])
transform_so_test = T.Compose(
    [
        T.Resize((org_size, org_size), Image.BICUBIC),
        T.CenterCrop((src_size, src_size)),
    ])
transform_eo = T.Compose(
    [
         T.ToTensor(),
    ])
transform_mo = T.Compose(
    [
        T.Resize((dst_size, dst_size), Image.BICUBIC),
    ])

train_path = os.path.join(opt.dataroot, 'train')
test_path = os.path.join(opt.dataroot, 'test')
trainset = MyDataset(train_path, transform_s=transform_so_train, transform_e=transform_eo, transform_m=transform_mo)
train_loader = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

testset = MyDataset(test_path, transform_s=transform_so_test, transform_e=transform_eo, transform_m=transform_mo)
test_loader = DataLoader(testset, batch_size=opt.batchSize, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
dataset_num_classes = len(trainset.classes)

print('Train and Test images:',len(trainset.imgs),len(testset.imgs))
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


# CLS Networks
print(opt.arch_cls)
if opt.arch_cls == 'resnet50':
    cls_model = resnet50(dataset_num_classes)
elif opt.arch_cls == 'resnet18':
    cls_model = resnet18(dataset_num_classes)
elif opt.arch_cls == 'shufflenetv2':
    from models.shufflenetv2 import shufflenetv2
    cls_model = shufflenetv2(num_classes=dataset_num_classes)
else:
    print('cls_model:%s is not find.'%(opt.arch_cls))

state_dict = torch.load('log/log_train_student_srnet/'+opt.name+'/last_'+opt.name+'_student.pth.tar', map_location=torch.device('cpu'))
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k[7:]] = v  #remove `module.`
cls_model.load_state_dict(new_state_dict)


if opt.cuda:
    sr_model = sr_model.cuda()
    cls_model = cls_model.cuda()
    sr_model = torch.nn.DataParallel(sr_model)
    cls_model = torch.nn.DataParallel(cls_model)


best_acc = 0


def test():
    sr_model.eval()
    cls_model.eval()

    out_hr = []
    out_lr = []
    out_sr = []
    label = []

    top1hr = 0
    top5hr = 0
    top1lr = 0
    top5lr = 0
    top1sr = 0
    top5sr = 0
    with torch.no_grad():
        for iter, (imhr, imlr, targets) in enumerate(test_loader):
            if opt.cuda:
                imhr, imlr, targets = imhr.cuda(), imlr.cuda(), targets.cuda()

            outputshr = cls_model(imhr)
            outputslr = cls_model(imlr)
            outputssr = cls_model(sr_model(imlr))

            out_hr.append(outputshr)
            out_lr.append(outputslr)
            out_sr.append(outputssr)
            label.append(targets)

            prec1hr, prec5hr = accuracy(outputshr.data, targets.data, topk=(1, 5))
            top1hr += prec1hr
            top5hr += prec5hr

            prec1lr, prec5lr = accuracy(outputslr.data, targets.data, topk=(1, 5))
            top1lr += prec1lr
            top5lr += prec5lr

            prec1sr, prec5sr = accuracy(outputssr.data, targets.data, topk=(1, 5))
            top1sr += prec1sr
            top5sr += prec5sr

            print('[%d/%d]' % (iter + 1, len(test_loader)))
            print('HR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1hr / (iter + 1), 100 * top5hr / (iter + 1)))
            print('LR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1lr / (iter + 1), 100 * top5lr / (iter + 1)))
            print('SR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1sr / (iter + 1), 100 * top5sr / (iter + 1)))

        out_hr2 = torch.cat(out_hr, dim=0)
        out_lr2 = torch.cat(out_lr, dim=0)
        out_sr2 = torch.cat(out_sr, dim=0)
        label2 = torch.cat(label, dim=0)

        top1hr, top5hr = accuracy(out_hr2.data, label2.data, topk=(1, 5))
        top1lr, top5lr = accuracy(out_lr2.data, label2.data, topk=(1, 5))
        top1sr, top5sr = accuracy(out_sr2.data, label2.data, topk=(1, 5))
        print('Final HR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1hr, 100 * top5hr))
        print('Final LR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1lr, 100 * top5lr))
        print('Final SR Top1: %.3f%%|Top5: %.3f%%' % (100 * top1sr, 100 * top5sr))

    return top1sr

test()
###################################
