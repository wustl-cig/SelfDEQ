import torchvision
import torch
import numpy as np
import logging
import pprint
import shutil
import warnings
import math
from collections import defaultdict
import jsonargparse
import argparse

from ..utility.common import *


def pformat_module(module: torch.nn.Module):
    structure = ''
    for name, p in module.named_parameters():
        if p.requires_grad:
            structure += str(name) + ': ' + str(type(p.data)) + " " + str(p.size()) + "\n"

    num_params = format(sum(p.numel() for p in module.parameters() if p.requires_grad), ',')

    return structure, num_params


def normalize_(x: torch.Tensor):
    try:
        x = x - torch.min(x)
        x = x / torch.max(x)
    except:
        pass

    try:
        x = x - np.amin(x)
        x = x / np.amax(x)
    except:
        pass

    return x


class Callback:
    def __init__(self):
        self.params = {}
        self.module = torch.nn.Module()

    def set_params(self, params: dict):
        self.params = params

    def set_module(self, module: torch.nn.Module):
        self.module = module

    def on_train_begin(self, image):
        pass

    def on_batch_end(self, log, batch):
        pass

    def on_epoch_end(self, log, image, epoch):
        pass


class CallbackList:
    def __init__(self, callbacks: [Callback] = None):
        self.callbacks = callbacks or []
        self.params = {}
        self.module = torch.nn.Module()

    def set_params(self, params: dict):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_module(self, module: torch.nn.Module):
        self.module = module
        for callback in self.callbacks:
            callback.set_module(module)

    def call_train_begin_hook(self, image):
        for callback in self.callbacks:
            callback.on_train_begin(image)

    def call_batch_end_hook(self, log, batch):
        for callback in self.callbacks:
            callback.on_batch_end(log, batch)

    def call_epoch_end_hook(self, log, image, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(log, image, epoch)


class FileLogger(Callback):
    import scipy.io as sio

    def __init__(self, file_path=None):
        self.file_path = file_path

        if file_path is not None:
            check_and_mkdir(self.file_path)

        self.__logs = defaultdict(list)

        super().__init__()

    def on_train_begin(self, image):
        if self.file_path:
            mdict = {}
            for i, k in enumerate(image, start=1):
                mdict[k] = image[k].detach().cpu().numpy()

            self.sio.savemat(self.file_path + 'image_initial.mat', mdict)

    def on_epoch_end(self, log, image, epoch):
        if self.file_path:
            mdict = {}
            for i, k in enumerate(image, start=1):
                try:
                    img = image[k].detach().cpu().numpy()
                except:
                    img = image[k]

                mdict[k] = img

            if mdict:
                self.sio.savemat(self.file_path + 'image_epoch_%.3d.mat' % epoch, mdict)

            for i, k in enumerate(log, start=1):
                self.__logs[k].append(log[k])

            cvs_data = np.transpose(np.array(list(self.__logs.values())))

            cvs_header = ''
            for k in self.__logs:
                cvs_header = cvs_header + k + ','

            np.savetxt(self.file_path + 'scalar.csv', cvs_data, delimiter=',', fmt='%.5f', header=cvs_header)


class Tensorboard(Callback):
    from torch.utils.tensorboard import SummaryWriter

    def __init__(self, file_path=None, per_batch=1, hparams=None):
        self.file_path = file_path
        self.per_batch = per_batch

        self.tb_writer = None

        if self.file_path is not None:
            check_and_mkdir(self.file_path)
            self.tb_writer = self.SummaryWriter(self.file_path)

            if hparams is not None:
                self.tb_writer.add_hparams(hparam_dict=hparams['hparam_dict'], metric_dict=hparams['metric_dict'])

        super().__init__()

    def on_train_begin(self, image):
        if self.tb_writer is not None:
            for i, k in enumerate(image, start=1):
                self.tb_writer.add_images(tag='init/' + k, img_tensor=normalize_(image[k]), global_step=0)

        if self.tb_writer is not None and 'config' in self.params:
            if isinstance(self.params['config'], argparse.Namespace):
                text_string = pprint.pformat((self.params['config']).clone().as_dict()) + '\n'
            else:
                text_string = pprint.pformat(self.params['config']) + '\n'

            self.tb_writer.add_text(tag='config',
                                    text_string=text_string.replace('\n', '\n\n'),
                                    global_step=0)

    def on_batch_end(self, log, batch):
        if self.tb_writer is not None:
            if batch % self.per_batch == 0:
                for i, k in enumerate(log, start=1):
                    self.tb_writer.add_scalar(tag='batch/' + k, scalar_value=log[k], global_step=batch)

    def on_epoch_end(self, log, image, epoch):
        if self.tb_writer is not None:
            for i, k in enumerate(log, start=1):
                self.tb_writer.add_scalar(tag='epoch/' + k, scalar_value=log[k], global_step=epoch)

            for i, k in enumerate(image, start=1):
                if image[k].shape.__len__() == 4 and (image[k].shape[1] == 1 or image[k].shape[1] == 3):
                    self.tb_writer.add_images(tag='epoch/' + k, img_tensor=normalize_(image[k]), global_step=epoch)


class Matplotlib(Callback):
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    def __init__(self, file_path=None):
        self.file_path = file_path

        if file_path is not None:
            check_and_mkdir(self.file_path)

        self.__logs = defaultdict(list)

        super().__init__()

    def on_train_begin(self, image):
        num_image = len(image.keys())

        num_col = 2
        num_row = math.ceil(num_image / num_col)

        self.plt.figure(1, figsize=[6 * num_col, 2 * num_row])

        for i, k in enumerate(image, start=1):
            image_ = torchvision.utils.make_grid(image[k]).permute([1, 2, 0])
            image_ = image_.detach().cpu()
            image_ = normalize_(image_)

            self.plt.subplot(num_row, num_col, i)
            self.plt.imshow(image_)
            self.plt.title(k)
            self.plt.axis('off')

        self.plt.suptitle('Initial')

        if self.file_path:
            self.plt.savefig(self.file_path + 'image_initial.png')
        else:
            self.plt.show()

        self.plt.close()  # Clean up memory

    def on_epoch_end(self, log, image, epoch):
        num_image = len(image.keys())

        num_col = 2
        num_row = math.ceil(num_image / num_col)

        self.plt.figure(1, figsize=[6 * num_col, 2 * num_row])

        for i, k in enumerate(image, start=1):
            image_ = torchvision.utils.make_grid(image[k]).permute([1, 2, 0])
            image_ = image_.detach().cpu()
            image_ = normalize_(image_)

            self.plt.subplot(num_row, num_col, i)
            self.plt.imshow(image_)
            self.plt.title(k)
            self.plt.axis('off')

        self.plt.suptitle('Image Epoch_%.3d' % epoch)

        if self.file_path:
            self.plt.savefig(self.file_path + 'image_epoch_%.3d' % epoch + '.png')
        else:
            self.plt.show()

        num_log = len(log.keys())

        num_col = 2
        num_row = math.ceil(num_log / num_col)

        self.plt.figure(2, figsize=[6.4 * num_col, 4.8 * num_row])

        for i, k in enumerate(log, start=1):
            self.__logs[k].append(log[k])

            self.plt.subplot(num_row, num_col, i)
            self.plt.plot(self.__logs[k], '-bo')
            self.plt.title(k)

        self.plt.suptitle('Scalar Epoch_%.3d' % epoch)

        if self.file_path:
            self.plt.savefig(self.file_path + 'scalar_epoch_%.3d' % epoch + '.png')
        else:
            self.plt.show()

        self.plt.close()  # Clean up memory


class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """

    def __init__(self, file_path=None):
        super().__init__()

        self.__logger = logging.getLogger('main')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        self.__logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.__logger.addHandler(stream_handler)

        if file_path is not None:
            check_and_mkdir(file_path)

            file_handler = logging.FileHandler(file_path + 'logging.txt', mode='w')
            file_handler.setFormatter(formatter)

            self.__logger.addHandler(file_handler)

    def on_train_begin(self, image):
        structure, num_params = pformat_module(self.module)
        self.__logger.info("Module Structure \n\n" + structure + '\n')
        self.__logger.info('Module Params Amount: ' + num_params + '\n')

        try:
            if isinstance(self.params['config'], argparse.Namespace):
                text_string = pprint.pformat(jsonargparse.core.namespace_to_dict(self.params['config'])) + '\n'
            else:
                text_string = pprint.pformat(self.params['config']) + '\n'

            self.__logger.info('Configuration \n\n' + text_string)
        except:
            self.__logger.info("CANNOT FIND CONFIG INFORMATION")

    def on_epoch_end(self, log, image, epoch):
        try:
            train_epoch = self.params['train_epoch']
            train_epoch = "%.3d" % train_epoch

        except:
            # Not Found in self.params[\'train_epoch\']
            train_epoch = "???"

        if log is not None:
            logger_pformat = '\n===============================\n' \
                             '  [BaseLogger] [%.3d/%s]\n' \
                             '===============================\n' % (epoch, train_epoch)
            for k in log:
                logger_pformat += ('[%s]: [%s]\n' % (k, str(log[k])))

            self.__logger.critical(logger_pformat)


class CodeBackupER(Callback):
    def __init__(self, src_path=None, file_path=None):
        super().__init__()

        if (file_path is not None) and (src_path is not None):
            check_and_mkdir(file_path)

            max_code_save = 100  # only 100 copies can be saved
            for i in range(max_code_save):
                code_path = file_path + 'code%d/' % i
                if not os.path.exists(code_path):
                    shutil.copytree(src=src_path, dst=code_path)
                    break


class ModelCheckpoint(Callback):
    def __init__(self,
                 file_path=None,
                 monitors: [str] = None,
                 modes: [str] = None,
                 period: int = 10):

        super().__init__()

        self.file_path = file_path
        if self.file_path is not None:
            check_and_mkdir(self.file_path)

        self.monitors = monitors or []
        self.period = period

        self.monitor_ops = []
        self.best_epochs = []
        self.best_values = []

        modes = modes or []
        for mode in modes:
            if mode not in ['min', 'max']:
                warnings.warn('ModelCheckpoint mode %s is unknown' % mode, RuntimeWarning)

            if mode == 'min':
                self.monitor_ops.append(np.less)
                self.best_epochs.append(0)
                self.best_values.append(np.Inf)

            elif mode == 'max':
                self.monitor_ops.append(np.greater)
                self.best_epochs.append(0)
                self.best_values.append(-np.Inf)

        self.__logger = logging.getLogger('main')

    def on_epoch_end(self, log, image, epoch):
        try:
            state_dict = self.module.module.state_dict()
        except AttributeError:
            state_dict = self.module.state_dict()

        torch.save(state_dict, self.file_path + 'latest.pt')

        try:
            train_epoch = self.params['train_epoch']
            train_epoch = "%.3d" % train_epoch

        except:
            # Not Found in self.params[\'train_epoch\']
            train_epoch = "???"

        if (epoch % self.period == 0) and (self.file_path is not None):
            torch.save(state_dict, self.file_path + 'epoch%.3d.pt' % epoch)

        checkpoint_pformat = '\n===============================\n' \
                             '  [ModelCheckpoint] [%.3d/%s]\n' \
                             '===============================\n' % (epoch, train_epoch)

        for i, monitor in enumerate(self.monitors):
            current = log.get(monitor)

            if self.monitor_ops[i](current, self.best_values[i]):
                checkpoint_pformat += '[%s] Improved: [%.5f] -> [%.5f] \n' % (
                    monitor, self.best_values[i], current)

                self.best_values[i] = current
                self.best_epochs[i] = epoch

                if self.file_path is not None:
                    torch.save(state_dict, self.file_path + 'best_%s.pt' % monitor)

            else:
                checkpoint_pformat += '[%s] Maintained: Current is [%.5f] Best is [%.5f] in Epoch [%.5d] \n' % (
                    monitor, current, self.best_values[i], self.best_epochs[i])

        self.__logger.critical(checkpoint_pformat)
