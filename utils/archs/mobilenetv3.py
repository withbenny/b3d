"""mobilenetv3 in pytorch

[1] Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc V. Le, Mark Sandler,
    Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu,
    Vijay Vasudevan, Yukun Zhu

    Searching for MobileNetV3
    https://arxiv.org/abs/1905.02244
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        squeezed = max(1, in_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, in_channels, 1),
            HardSigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel, stride, use_se, use_hs):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        activation = HardSwish() if use_hs else nn.ReLU(inplace=True)

        layers = []
        if exp_channels != in_channels:
            layers += [
                nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                nn.BatchNorm2d(exp_channels),
                activation,
            ]
        layers += [
            nn.Conv2d(exp_channels, exp_channels, kernel, stride=stride,
                      padding=kernel // 2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            activation,
        ]
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))
        layers += [
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3Large(nn.Module):
    # (kernel, exp, out, use_se, use_hs, stride)
    _cfg = [
        (3,  16,  16, False, False, 1),
        (3,  64,  24, False, False, 2),
        (3,  72,  24, False, False, 1),
        (5,  72,  40, True,  False, 2),
        (5, 120,  40, True,  False, 1),
        (5, 120,  40, True,  False, 1),
        (3, 240,  80, False, True,  2),
        (3, 200,  80, False, True,  1),
        (3, 184,  80, False, True,  1),
        (3, 184,  80, False, True,  1),
        (3, 480, 112, True,  True,  1),
        (3, 672, 112, True,  True,  1),
        (5, 672, 160, True,  True,  2),
        (5, 960, 160, True,  True,  1),
        (5, 960, 160, True,  True,  1),
    ]

    def __init__(self, class_num=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
        )

        layers = []
        in_ch = 16
        for kernel, exp, out, use_se, use_hs, stride in self._cfg:
            layers.append(InvertedResidual(in_ch, exp, out, kernel, stride, use_se, use_hs))
            in_ch = out
        self.layers = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Conv2d(960, 1280, 1, bias=False),
            HardSwish(),
            nn.Conv2d(1280, class_num, 1),
        )

    def forward(self, x, return_hidden=False):
        x = self.stem(x)
        x = self.layers(x)
        x = self.head(x)
        x = self.pool(x)
        if return_hidden:
            hidden = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        if return_hidden:
            return x, hidden
        return x

    def freeze_fc(self):
        for name, param in self.named_parameters():
            if "classifier" in name:
                param.requires_grad = False

    def unfreeze_fc(self):
        for name, param in self.named_parameters():
            if "classifier" in name:
                param.requires_grad = True


class MobileNetV3Small(nn.Module):
    # (kernel, exp, out, use_se, use_hs, stride)
    _cfg = [
        (3,  16,  16, True,  False, 2),
        (3,  72,  24, False, False, 2),
        (3,  88,  24, False, False, 1),
        (5,  96,  40, True,  True,  2),
        (5, 240,  40, True,  True,  1),
        (5, 240,  40, True,  True,  1),
        (5, 120,  48, True,  True,  1),
        (5, 144,  48, True,  True,  1),
        (5, 288,  96, True,  True,  2),
        (5, 576,  96, True,  True,  1),
        (5, 576,  96, True,  True,  1),
    ]

    def __init__(self, class_num=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
        )

        layers = []
        in_ch = 16
        for kernel, exp, out, use_se, use_hs, stride in self._cfg:
            layers.append(InvertedResidual(in_ch, exp, out, kernel, stride, use_se, use_hs))
            in_ch = out
        self.layers = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Conv2d(576, 1024, 1, bias=False),
            HardSwish(),
            nn.Conv2d(1024, class_num, 1),
        )

    def forward(self, x, return_hidden=False):
        x = self.stem(x)
        x = self.layers(x)
        x = self.head(x)
        x = self.pool(x)
        if return_hidden:
            hidden = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        if return_hidden:
            return x, hidden
        return x

    def freeze_fc(self):
        for name, param in self.named_parameters():
            if "classifier" in name:
                param.requires_grad = False

    def unfreeze_fc(self):
        for name, param in self.named_parameters():
            if "classifier" in name:
                param.requires_grad = True


def mobilenetv3_large(num_classes=10):
    return MobileNetV3Large(class_num=num_classes)


def mobilenetv3_small(num_classes=10):
    return MobileNetV3Small(class_num=num_classes)


# default alias used in config.arch
mobilenetv3 = mobilenetv3_large
