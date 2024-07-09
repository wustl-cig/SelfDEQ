import torch
import torchvision
from pkg.torch.modules.common import UNet

def normalization_fn(ipt):
    assert ipt.shape[0] == 2

    ipt_abs, ipt_angle = torch.sqrt(ipt[0] ** 2 + ipt[1] ** 2), torch.atan2(ipt[1], ipt[0])

    ipt_abs_mean, ipt_abs_std, ipt_abs_min, ipt_abs_max = \
        torch.mean(ipt_abs), torch.std(ipt_abs), torch.min(ipt_abs), torch.max(ipt_abs)

    minus_cof = ipt_abs_min
    divis_cof = ipt_abs_mean + 2 * ipt_abs_std - ipt_abs_min

    ret_abs = (ipt_abs - minus_cof) / divis_cof
    ret = torch.unsqueeze(ret_abs, 0) * torch.stack([torch.cos(ipt_angle), torch.sin(ipt_angle)], 0)

    return ret, minus_cof, divis_cof


def batch_normalization_fn(ipt):
    num_batch = ipt.shape[0]

    ret, minus_cof, divis_cof = [], [], []
    for i in range(num_batch):
        ret_cur, minus_cof_cur, divis_cof_cur = normalization_fn(ipt[i])

        ret.append(ret_cur)

        minus_cof.append(minus_cof_cur)
        divis_cof.append(divis_cof_cur)

    ret = torch.stack(ret, 0)

    return ret, minus_cof, divis_cof


def renormalization_fn(ipt, minus_cof, divis_cof):
    assert ipt.shape[0] == 2

    ipt_abs = torch.sqrt(ipt[0] ** 2 + ipt[1] ** 2)
    ipt_angle = torch.atan2(ipt[1], ipt[0])

    ret_abs = ipt_abs * divis_cof + minus_cof
    ret = torch.unsqueeze(ret_abs, 0) * torch.stack([torch.cos(ipt_angle), torch.sin(ipt_angle)], 0)

    return ret


def batch_renormalization_fn(ipt, minus_cof, divis_cof):
    num_batch = ipt.shape[0]

    ret = torch.stack([
        renormalization_fn(ipt[i], minus_cof[i], divis_cof[i]) for i in range(num_batch)
    ], 0)

    return ret


def make_grid(x):

    x = x.detach().cpu()

    assert x.dim() == 6

    NUM_BATCH, NUM_PHASE, NUM_COIL, COMPLEX, WIDTH, HEIGHT = x.shape

    x = torch.sqrt((x ** 2).sum(3, keepdim=True))

    for batch in range(NUM_BATCH):
        for phase in range(NUM_PHASE):
            for coil in range(NUM_COIL):
                x[batch, phase, coil] -= torch.min(x[batch, phase, coil])
                x[batch, phase, coil] /= torch.max(x[batch, phase, coil])

    x = x.reshape([NUM_BATCH, NUM_PHASE * NUM_COIL, 1, WIDTH, HEIGHT])

    ret = torch.stack([
        torchvision.utils.make_grid(x[i], nrow=NUM_COIL, padding=5) for i in range(NUM_BATCH)
    ], 0)

    return ret


def b1_divided_by_rss(b1):
    b1_rss = b1[..., 0, :, :] ** 2 + b1[..., 1, :, :] ** 2
    b1_rss = b1_rss.sum(-3)
    b1_rss = torch.sqrt(b1_rss).unsqueeze(2).unsqueeze(2)

    b1 = b1 / b1_rss

    return b1


def architecture_dict(config):

    if config.setting.dataset == 'brain':
        unet_padding = (1, 0)
    else:
        unet_padding = (0, 0)

    return {
        'UNet': lambda: UNet(
            dimension=config.method.dimension,
            i_nc=2,
            o_nc=2,
            f_root=config.method.architecture.UNet.f_root,
            conv_times=config.method.architecture.UNet.conv_times,
            is_bn=False,
            activation='relu',
            is_residual=config.method.architecture.UNet.is_residual,
            up_down_times=config.method.architecture.UNet.up_down_times,
            is_spe_norm=config.method.architecture.UNet.is_speNorm,
            padding=unet_padding
        ),
    }


def loss_fn_dict():
    return {
        'l2': torch.nn.MSELoss,
    }
