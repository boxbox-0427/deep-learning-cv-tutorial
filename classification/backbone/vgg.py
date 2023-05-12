from torch import nn
from typing import Literal

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = None
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    @staticmethod
    def make_layers(vgg: Literal["vgg11", "vgg13", "vgg16", "vgg19"]):
        layers = []
        in_channels = 3
        for x in cfgs[vgg]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11(VGG):

    def __init__(self):
        super(VGG11, self).__init__()
        self.features = VGG.make_layers(vgg="vgg11")


class VGG13(VGG):

    def __init__(self):
        super(VGG13, self).__init__()
        self.features = VGG.make_layers(vgg="vgg13")


class VGG16(VGG):

    def __init__(self):
        super(VGG16, self).__init__()
        self.features = VGG.make_layers(vgg="vgg16")


class VGG19(VGG):

    def __init__(self):
        super(VGG19, self).__init__()
        self.features = VGG.make_layers(vgg="vgg19")
