from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Normalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std',  torch.tensor(std))

    def forward(self, x):
        x = (x - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        return x

    def extra_repr(self):
        return f'mean={self.mean}, std={self.std}'


class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1)


def get_mlp(dims, act_fn=nn.ReLU, bn=False, bias=True):
    layers = []
    for i in range(len(dims)-1):
        if i < len(dims)-2:
            layers.append(nn.Linear(dims[i], dims[i+1], bias=not bn and bias))
            if bn:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(act_fn())
        else:
            layers.append(nn.Linear(dims[i], dims[i+1], bias=bias))
    mlp = nn.Sequential(*layers)
    mlp.out_dim = dims[-1]
    return mlp


class ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    '''from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        x = x + shortcut
        return x


class ResNet(nn.Module):
    '''from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py'''
    def __init__(self, layers, projection_layers: list[int], mean, std):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)

        in_channels = layers[0][0]
        self.stem = nn.Sequential(
            Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))

        stages = []
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)

        self.pool = GlobalAveragePooling()
        self.head = get_mlp([in_channels, 2048, projection_layers[0]],
                            act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=True)
        self.projection = get_mlp(projection_layers, 
                                  act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=False)
        self.projection_layers = projection_layers

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.stem(x)
        out = self.stages(out)
        out = self.pool(self.act(out))
        f = self.head(out)
        z = self.projection(f)
        return f, z


class MSResNet(nn.Module):
    '''from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py'''
    def __init__(self, layers, projection_layers: list[int], mean, std):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)

        in_channels = layers[0][0]
        self.stem1 = nn.Sequential(
            Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))
        self.stem2 = nn.Sequential(
            Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))
        self.stem3 = nn.Sequential(
            Normalize(mean=mean, std=std),
            nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1))

        stages = []
        in_channels = layers[0][0]
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages1 = nn.Sequential(*stages)

        stages = []
        in_channels = layers[0][0]
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages2 = nn.Sequential(*stages)

        stages = []
        in_channels = layers[0][0]
        for i, (out_channels, num_blocks) in enumerate(layers):
            stage = []
            if i > 0:
                stage.append(nn.AvgPool2d(2))
            for _ in range(num_blocks):
                stage.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages3 = nn.Sequential(*stages)

        self.pool = GlobalAveragePooling()
        self.head = get_mlp([in_channels*3, 2048, projection_layers[0]],
                            act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=True)
        self.projection = get_mlp(projection_layers,
                                  act_fn=lambda: nn.LeakyReLU(0.2), bn=False, bias=False)
        self.projection_layers = projection_layers

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out1 = self.stages1(self.stem1(x))
        out2 = self.stages2(self.stem2(F.avg_pool2d(x, 2)))
        out3 = self.stages3(self.stem3(F.avg_pool2d(x, 4)))
        out = torch.cat([self.pool(self.act(out1)), self.pool(self.act(out2)), self.pool(self.act(out3))], dim=1)
        f = self.head(out)
        z = self.projection(f)
        return f, z


def get_ebm(name: str, input_shape: tuple[int], projection_layers: list[int],
            mean: list[float], std: list[float]):
    if name == 'resnet':
        layers = [(128, 2), (128, 2), (256, 2), (256, 2)]
        encoder = ResNet(layers, projection_layers=projection_layers, mean=mean, std=std)

    elif name == 'resnet_small':
        layers = [(64, 1), (64, 1), (128, 1), (128, 1)]
        encoder = ResNet(layers, projection_layers=projection_layers, mean=mean, std=std)

    elif name == 'resnet_large':
        layers = [(256, 2), (256, 2), (256, 2), (256, 2)]
        encoder = MSResNet(layers, projection_layers=projection_layers, mean=mean, std=std)

    else:
        raise Exception(f'Unknown Encoder: {name}')

    for m in encoder.modules():
        if isinstance(m, nn.Conv2d):
            nn.utils.parametrizations.spectral_norm(m)

    return encoder


def get_encoder(name: str, input_shape: tuple[int], out_dim: int,
                mean: list[float], std: list[float],
                pretrained=None):
    encoder = torchvision.models.__dict__[name](zero_init_residual=True)
    encoder.out_dim = out_dim
    encoder.fc = get_mlp([encoder.fc.weight.shape[1], 2048, out_dim], act_fn=nn.ReLU, bn=True)

    if input_shape == (3, 32, 32):
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    encoder.conv1 = nn.Sequential(Normalize(mean, std), encoder.conv1)
    return encoder

