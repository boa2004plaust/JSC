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
    # print( "FLOPs={:.2f}G".format( flops.total() * 0.5 / 1e9 ) )
    # print(params)
