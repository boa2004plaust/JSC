import torch
import torch.nn as nn
from torchvision import models


resnet = models.resnet18(pretrained=True)


class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.feature = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                     resnet.layer1, resnet.layer2, resnet.layer3,
                                     resnet.layer4)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
       )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, is_feat=False):
        x = self.feature(x)
        f = self.gap(x)
        f = f.view(f.size(0), -1)
        f = self.map(f)
        c = self.fc(f)
        if is_feat:
            return f, c
        else:
            return c


class resnet18dfl_cnn(nn.Module):
    def __init__(self, num_classes, k=5):
        super(resnet18dfl_cnn, self).__init__()

        # k channels for one class, nclass is total classes,
        # therefore k * nclass for conv6
        self.k = k
        self.nclass = num_classes

        # Feature extraction root
        self.base = nn.Sequential(resnet.conv1, resnet.bn1,
                                  resnet.relu, resnet.maxpool,
                                  resnet.layer1, resnet.layer2,
                                  resnet.layer3
                                 )

        # G-Stream
        self.conv5 = nn.Sequential(resnet.layer4[0],
                                   resnet.layer4[1])
        self.map5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
       )
        self.cls5 = nn.Conv2d(512, self.nclass, kernel_size=1)

        # # P-Stream
        self.conv6 = nn.Conv2d(256, k * self.nclass, kernel_size=1, bias=False)
        self.pool6 = nn.MaxPool2d((14, 14), stride=(1, 1), return_indices=True)
        self.map6 = nn.Sequential(
            nn.Conv2d(k * self.nclass, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
       )
        self.cls6 = nn.Conv2d(512, self.nclass, kernel_size=1)

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x, is_feat=False):
        batchsize = x.size(0)

        # Stem: Feature extraction
        inter4 = self.base(x)
        # print(inter4.shape)

        # G-stream
        x_g = self.conv5(inter4)
        f_g = self.map5(x_g)
        out1 = self.cls5(f_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        f_p = self.map6(x_p)
        out2 = self.cls6(f_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        if is_feat:
            f_g = f_g.view(f_g.size(0), -1)
            f_p = f_p.view(f_p.size(0), -1)
            return [f_g, f_p], [out1, out2, out3, indices]
        else:
            return [out1, out2, out3, indices]


def compute_FLOPS_PARAMS(x, model):
    import thop
    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print("FLOPs={:.2f}G".format(flops / 1e9))
    print("params={:.2f}M".format(params / 1e6))


if __name__ == '__main__':
    print('###############Net4################')
    model = resnet18(200)
    x = torch.randn(1, 3, 224, 224)
    compute_FLOPS_PARAMS(x, model)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #
    # flops = FlopCountAnalysis(model, x)
    # print(flops.total())
    # params = parameter_count_table(model)
    # print("FLOPs={:.2f}G".format(flops.total() * 0.5 / 1e9))
    # print(params)
