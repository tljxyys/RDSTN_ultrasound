# modified from https://github.com/JayPatwardhan/ResNet-PyTorch
from argparse import Namespace
import torch
import torch.nn as nn

from models import register


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResNet(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(ResNet, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_body = []
        for i in range(n_resblock):
            m_body.append(ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)
        self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res)
        res += x
        return x


@register('resnet50')
def make_ResNet50(n_resblocks=50, n_colors=3, n_feats=64, res_scale=0.1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.n_colors = n_colors
    return ResNet(args)
