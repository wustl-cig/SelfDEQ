import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.functional import pad

activation_fn = {
    'relu': lambda: nn.ReLU(inplace=True),
    'lrelu': lambda: nn.LeakyReLU(inplace=True),
    'prelu': lambda: nn.PReLU()
}


class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, times=1, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        super().__init__()

        if dimension == 3:
            conv_fn = lambda in_c: torch.nn.Conv3d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda in_c: torch.nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        layers = []
        for i in range(times):
            if i == 0:
                layers.append(spectral_norm(conv_fn(in_channels)) if is_spe_norm else conv_fn(in_channels))
            else:
                layers.append(spectral_norm(conv_fn(out_channels)) if is_spe_norm else conv_fn(out_channels))

            if is_bn:
                layers.append(bn_fn())

            if activation is not None:
                layers.append(activation_fn[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvtranBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        self.is_bn = is_bn
        super().__init__()
        if dimension == 3:
            conv_fn = lambda: torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 2, 2),
                padding=kernel_size // 2,
                output_padding=(0, 1, 1)
            )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda: torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1
            )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        self.net1 = spectral_norm(conv_fn()) if is_spe_norm else conv_fn()
        if self.is_bn:
            self.net2 = bn_fn()
        self.net3 = activation_fn[activation]()

    def forward(self, x):
        ret = self.net1(x)
        if self.is_bn:
            ret = self.net2(ret)

        ret = self.net3(ret)

        return ret


class UNet(nn.Module):
    def __init__(self, dimension, i_nc=1, o_nc=1, f_root=32, conv_times=3, is_bn=False, activation='relu',
                 is_residual=False, up_down_times=3, is_spe_norm=False, padding=(1, 4)):

        print("padding:", padding)

        self.is_residual = is_residual
        self.up_down_time = up_down_times
        self.dimension = dimension
        self.padding = padding

        super().__init__()

        if dimension == 2:
            self.down_sample = nn.MaxPool2d((2, 2))
        elif dimension == 3:
            self.down_sample = nn.MaxPool3d((1, 2, 2))
        else:
            raise ValueError()

        self.conv_in = ConvBnActivation(
            in_channels=i_nc,
            out_channels=f_root,
            is_bn=is_bn,
            activation=activation,
            dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.conv_out = ConvBnActivation(
            in_channels=f_root,
            out_channels=o_nc,
            kernel_size=1,
            dimension=dimension,
            times=1,
            is_bn=False,
            activation=None,
            is_spe_norm=is_spe_norm
        )

        self.bottom = ConvBnActivation(
            in_channels=f_root * (2 ** (up_down_times - 1)),
            out_channels=f_root * (2 ** up_down_times),
            times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.down_list = nn.ModuleList([
                                           ConvBnActivation(
                                               in_channels=f_root * 1,
                                               out_channels=f_root * 1,
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                           )
                                       ] + [
                                           ConvBnActivation(
                                               in_channels=f_root * (2 ** i),
                                               out_channels=f_root * (2 ** (i + 1)),
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                            )
                                           for i in range(up_down_times - 1)
                                       ])

        self.up_conv_list = nn.ModuleList([
            ConvBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

        self.up_conv_tran_list = nn.ModuleList([
            ConvtranBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

    def forward(self, x):

        input_ = x

        x = pad(x, [0, self.padding[0], 0, self.padding[1]])

        x = self.conv_in(x)

        skip_layers = []
        for i in range(self.up_down_time):
            x = self.down_list[i](x)

            skip_layers.append(x)
            x = self.down_sample(x)

        x = self.bottom(x)

        for i in range(self.up_down_time):
            x = self.up_conv_tran_list[i](x)
            x = torch.cat([x, skip_layers[self.up_down_time - i - 1]], 1)
            x = self.up_conv_list[i](x)

        x = self.conv_out(x)

        if self.padding[0] > 0:
            x = x[..., :-self.padding[0]]
        if self.padding[1] > 0:
            x = x[..., :-self.padding[1], :]

        # x = x[..., :-self.padding[1], :-self.padding[0]]

        ret = input_ - x if self.is_residual else x

        return ret


class DnCNN(nn.Module):
    def __init__(self, dimension, depth=13, n_channels=64, i_nc=1, o_nc=1, kernel_size=3, is_batch_normalize=False, is_residual=True):

        self.is_residual = is_residual

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d

        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d

        else:
            raise ValueError()

        super().__init__()
        padding = kernel_size // 2

        layers = [
            spectral_norm(conv_fn(
            in_channels=i_nc,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)),
            nn.ReLU(inplace=True)
        ]

        for _ in range(depth - 1):
            layers.append(spectral_norm(conv_fn(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False if is_batch_normalize else True)))

            if is_batch_normalize:
                layers.append(bn_fn(n_channels))

            layers.append(nn.ReLU(inplace=True))

        layers.append(
            spectral_norm(conv_fn(
                in_channels=n_channels,
                out_channels=o_nc,
                kernel_size=kernel_size,
                padding=padding,
                bias=False)))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        input_ = x

        x = self.net(x)

        ret = input_ - x if self.is_residual else x

        return ret


class ResBlock(nn.Module):
    def __init__(self, dimension, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, is_speNorm=False):

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d
        elif dimension == 2.5:
            conv_fn = Conv2halfD
        else:
            raise ValueError()

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if not is_speNorm:
                m.append(conv_fn(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            else:
                m.append(spectral_norm(conv_fn(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias)))

            if bn:
                m.append(bn_fn(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Conv2halfD(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_size, padding, bias=True):
        super().__init__()

        self.n_feats = n_feats

        self.conv_xy = nn.Conv2d(in_channels, n_feats, kernel_size, padding=padding, bias=bias)
        self.conv_yz = nn.Conv2d(in_channels, n_feats, kernel_size, padding=padding, bias=bias)
        self.conv_xz = nn.Conv2d(in_channels, n_feats, kernel_size, padding=padding, bias=bias)

    def forward(self, x):

        num_batch, num_channel, num_x, num_y, num_z = x.shape

        in_xy = x.permute([0, 2, 1, 3, 4])
        in_xy = in_xy.reshape([num_batch * num_x, num_channel, num_y, num_z])

        re_xy = self.conv_xy(in_xy)

        re_xy = re_xy.reshape([num_batch, num_x, self.n_feats, num_y, num_z])
        re_xy = re_xy.permute([0, 2, 1, 3, 4])

        in_yz = x.permute([0, 3, 1, 2, 4])
        in_yz = in_yz.reshape([num_batch * num_y, num_channel, num_x, num_z])

        re_yz = self.conv_xy(in_yz)

        re_yz = re_yz.reshape([num_batch, num_y, self.n_feats, num_x, num_z])
        re_yz = re_yz.permute([0, 2, 3, 1, 4])

        in_xz = x.permute([0, 4, 1, 2, 3])
        in_xz = in_xz.reshape([num_batch * num_z, num_channel, num_x, num_y])

        re_xz = self.conv_xy(in_xz)

        re_xz = re_xz.reshape([num_batch, num_z, self.n_feats, num_x, num_y])
        re_xz = re_xz.permute([0, 2, 3, 4, 1])

        return (re_xy + re_yz + re_xz) / 3


class EDSR(nn.Module):
    def __init__(self, dimension, n_resblocks, n_feats, res_scale, in_channels=1, out_channels=1, act='relu', is_speNorm=False):
        super().__init__()

        if dimension == 2:
            conv_fn = nn.Conv2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
        elif dimension == 2.5:
            conv_fn = Conv2halfD
        else:
            raise ValueError()

        if not is_speNorm:
            m_head = [conv_fn(in_channels, n_feats, 3, padding=3 // 2)]
        else:
            m_head = [spectral_norm(conv_fn(in_channels, n_feats, 3, padding=3 // 2))]

        m_body = [
            ResBlock(
                dimension, n_feats, 3, res_scale=res_scale, act=activation_fn[act](), is_speNorm=is_speNorm
            ) for _ in range(n_resblocks)
        ]

        if not is_speNorm:
            m_body.append(conv_fn(n_feats, n_feats, 3, padding=3 // 2))
        else:
            m_body.append(spectral_norm(conv_fn(n_feats, n_feats, 3, padding=3 // 2)))

        if not is_speNorm:
            m_tail = [
                conv_fn(n_feats, out_channels, 3, padding=3 // 2)
            ]
        else:
            m_tail = [
                spectral_norm(conv_fn(n_feats, out_channels, 3, padding=3 // 2))
            ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
