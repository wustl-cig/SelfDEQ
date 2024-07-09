from method.shared import make_grid
import torch


def make_grid_brain(x):

    x = x.detach().cpu()
    x = x.unsqueeze(1)
    x = x.permute([0, 3, 1, 2, 4, 5])

    ret = make_grid(x)

    return ret


def predict2dicom(x):
    x = x.detach().cpu()
    x = x[..., 128:-128, :]
    x = torch.sqrt(torch.sum(x ** 2, 1))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] -= torch.min(x[i, j])
            x[i, j] /= torch.max(x[i, j])

    return x


def predict2dicom_2d(x, is_crop=True):
    x = x.detach().cpu()

    if is_crop:
        x = x[..., 128:-128, :]

    x = torch.sqrt(torch.sum(x ** 2, 1))

    for i in range(x.shape[0]):
        x[i] -= torch.min(x[i])
        x[i] /= torch.max(x[i])

    return x


def grappa2dicom_2d(x, is_crop=True):
    x = x.detach().cpu()
    x = torch.abs(x)

    if is_crop:
        x = x[..., 128:-128, :]

    for i in range(x.shape[0]):
        x[i] -= torch.min(x[i])
        x[i] /= torch.max(x[i])

    return x
