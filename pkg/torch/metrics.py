from collections import defaultdict
import numpy as np
import torch


class Metrics:
    def __init__(self):
        self._value = defaultdict(list)

    def reset_state(self):
        self._value = defaultdict(list)

    def update_state(self, value: dict):
        for key in value:
            self._value[key].append(value[key])

    def result(self):
        """Computes and returns the metric value tensor.
        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        """
        raise NotImplementedError('Must be implemented in subclasses.')


class Mean(Metrics):
    def __init__(self):
        super().__init__()

    def result(self):
        rec = {}
        for key in self._value:
            rec[key] = np.array(self._value[key]).mean()

        return rec


class Stack(Metrics):
    def __init__(self):
        super().__init__()

    def result(self):
        rec = {}
        for key in self._value:
            rec[key] = np.stack(self._value[key], 0)

        return rec


class Concatenate(Metrics):
    def __init__(self):
        super().__init__()

    def result(self):
        rec = {}
        for key in self._value:
            rec[key] = np.concatenate(self._value[key], 0)

        return rec


def compute_dice(vol1: torch.Tensor, vol2: torch.Tensor, labels=None, nargout=1, size_average=True):
    if not size_average:
        raise NotImplementedError()

    if labels is None:
        labels = torch.unique(torch.cat((vol1, vol2)))
        labels = labels[labels != 0]  # remove background

    dicem = torch.zeros_like(labels)
    for idx, lab in enumerate(labels):
        vol1l = (vol1 == lab)
        vol2l = (vol2 == lab)

        top = 2 * torch.sum(vol1l & vol2l, dtype=torch.float32)
        bottom = torch.sum(vol1l, dtype=torch.float32) + torch.sum(vol2l, dtype=torch.float32)
        if bottom == 0:
            bottom = 1e-10  # avoid 0 as Denominator.

        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem.mean()

    else:
        raise NotImplementedError
