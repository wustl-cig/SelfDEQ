import torch
import torch.nn.functional as f
import numpy as np
import math


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def ncc_loss(i, j, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(i.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).cuda()

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = 1
        padding = pad_no
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    i_var, j_var, cross = compute_local_sums(i, j, sum_filt, stride, padding, win)

    cc = cross * cross / (i_var * j_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(i, j, filt, stride, padding, win):
    i2 = i * i
    j2 = j * j
    ij = i * j

    i_sum = f.conv2d(i, filt, stride=stride, padding=padding)
    j_sum = f.conv2d(j, filt, stride=stride, padding=padding)
    i2_sum = f.conv2d(i2, filt, stride=stride, padding=padding)
    j2_sum = f.conv2d(j2, filt, stride=stride, padding=padding)
    ij_sum = f.conv2d(ij, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_i = i_sum / win_size
    u_j = j_sum / win_size

    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_size
    i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * win_size
    j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * win_size

    return i_var, j_var, cross
