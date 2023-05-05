import torch
import torch.nn as nn
from torchvision import models


shuf = models.shufflenet_v2_x1_0(pretrained=True)


class shufflenetv2(nn.Module):
    def __init__(self, num_classes):
        super(shufflenetv2, self).__init__()
        self.feature = nn.Sequential(shuf.conv1, shuf.maxpool,
                                     shuf.stage2, shuf.stage3, shuf.stage4,
                                     shuf.conv5)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, is_feat=False):
        x = self.feature(x)
        # print(x.shape)
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
    model = shufflenetv2(200)
    x = torch.randn(1, 3, 224, 224)
    compute_FLOPS_PARAMS(x, model)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #
    # flops = FlopCountAnalysis(model, x)
    # print(flops.total())
    # params = parameter_count_table(model)
    # print( "FLOPs={:.2f}G".format( flops.total() * 0.5 / 1e9 ) )
    # print(params)
