import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

from .stimulation.dataset_simulation import Dataset as Dataset_Simulation

from method_brain.shared_brain import make_grid_brain, grappa2dicom_2d, make_grid
from method_brain.tester_brain import predict2dicom, evaluate_helper, predict2dicom_2d, \
    evaluate_helper_base, test_helper

from pkg.torch.callback import CallbackList, FileLogger, Tensorboard, Matplotlib, BaseLogger, CodeBackupER, \
    ModelCheckpoint
from pkg.utility.common import dict2pformat
from pkg.torch.metrics import Mean
from pkg.utility.common import check_and_mkdir

import statistics
import time
import datetime


def train(
        config,
        network,
        hparams=None,
):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    if config.setting.dataset == 'simulation':
        Dataset = Dataset_Simulation
    else:
        raise NotImplementedError()

    train_dataset = Dataset(
        mode='train',
        config=config)

    print("[train_dataset] total_len: ", train_dataset.__len__())

    valid_dataset = Dataset(
        mode='valid',
        config=config)
    print("[valid_dataset] total_len: ", valid_dataset.__len__())

    test_dataset = Dataset(
        mode='test',
        config=config)
    print("[test_dataset] total_len: ", test_dataset.__len__())

    ########################
    # Load Configuration
    ########################
    batch_size = config.train.batch_size

    num_workers = config.train.num_workers
    verbose_batch = config.train.verbose_batch
    train_epoch = config.train.train_epoch
    checkpoint_epoch = config.train.checkpoint_epoch
    tensorboard_batch = config.train.tensorboard_batch
    gradient_clipping = config.train.gradient_clipping

    src_path = config.train.src_path
    lr = float(config.train.lr)
    file_path = config.setting.root_path + config.setting.save_folder + '/'

    coff_k_space_target = config.method.loss_fn.coff_k_space_target
    coff_k_space_input = config.method.loss_fn.coff_k_space_input

    ########################
    # Metrics
    ########################
    metrics = Mean()

    # global_variables._init()

    ########################
    # Extra-Definition
    ########################
    network = torch.nn.DataParallel(network)
    network.cuda()

    ########################
    # Dataset
    ########################
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size * torch.cuda.device_count(),
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / batch_size)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size * torch.cuda.device_count(),
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=0)

    image_init = {}

    ########################
    # Begin Training
    ########################
    optimizer = Adam(network.parameters(), lr=lr)

    check_and_mkdir(file_path)

    callbacks = CallbackList(callbacks=[
        BaseLogger(file_path=file_path),
        Tensorboard(file_path=file_path, per_batch=tensorboard_batch, hparams=hparams),
        ModelCheckpoint(file_path=file_path + 'recon_model/', period=checkpoint_epoch,
                monitors=['valid_ssim', 'valid_psnr'],
                modes=['max', 'max'])
    ])

    callbacks.set_module(network)
    callbacks.set_params({
        'config': config,
        "lr": lr,
        'train_epoch': train_epoch
    })

    callbacks.call_train_begin_hook(image_init)

    torch.autograd.set_detect_anomaly(True)

    time.sleep(1)  # wait for queue

    global_batch = 1
    for global_epoch in range(1, train_epoch + 1):

        network.train()

        iter_ = tqdm(train_dataloader, desc='Train [%.3d/%.3d]' % (global_epoch, train_epoch),
                     total=len(train_dataloader))
        for i, train_data in enumerate(iter_):
            dicom, x0, y_input, y_label, smps_input, smps_label, mask_input, mask_label, grappa = train_data

            _, x_hat, loss_total, loss_to_input, loss_to_target, forward_res = network(x0, y_input, smps_input,
                                                                                       mask_input, True, y_label,
                                                                                       smps_label, mask_label)

            optimizer.zero_grad()
            loss_total.mean().backward()

            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=gradient_clipping)

            optimizer.step()

            network.zero_grad()

            x_hat = x_hat[-1]

            log_batch = {
                'loss_total': loss_total.mean().item(),
            }

            metrics.update_state(log_batch)

            if (verbose_batch > 0) and (global_batch % verbose_batch == 0):
                iter_.write(("Batch [%.7d]:" % global_batch) + dict2pformat(log_batch))
                iter_.update()

            callbacks.call_batch_end_hook(log_batch, global_batch)
            global_batch += 1


        del iter_
        network.eval()

        with torch.no_grad():
            iter_ = tqdm(valid_dataloader, desc='Valid [%.3d/%.3d]' % (global_epoch, train_epoch),
                         total=len(valid_dataloader))
            for i, valid_data in enumerate(iter_):
                dicom, x0, y_input, smps_input, mask_input, grappa = valid_data

                _, x_hat = network(x0.cuda(), y_input.cuda(), smps_input.cuda(), mask_input.cuda(), is_trained=False)

                x_hat = x_hat[-1]

                x_hat = predict2dicom_2d(x_hat, is_crop=config.setting.dataset == 'brain')
                grappa = grappa2dicom_2d(grappa, is_crop=config.setting.dataset == 'brain')

                grappa = grappa.unsqueeze(1)
                x_hat = x_hat.unsqueeze(1)
                dicom = dicom.unsqueeze(1)

                snr_grappa, psnr_grappa, ssim_grappa, rsnr_grappa, rpsnr_grappa, rssim_grappa = evaluate_helper_base(
                    x_hat, grappa, is_normalized=config.method.dimension == 3)

                metrics.update_state({
                    'valid_psnr': psnr_grappa,
                    'valid_ssim': ssim_grappa,
                })

        del iter_

        log_epoch = metrics.result()
        metrics.reset_state()

        callbacks.call_epoch_end_hook(log_epoch, {}, global_epoch)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)

    save_path = file_path + '/TEST_' + datetime.datetime.now().strftime("%m%d%H%M") + \
                        '_' + config.setting.save_folder + '/'

    test_helper(
        config=config,
        dataloader=test_dataloader,
        save_path=save_path,
        checkpoint=file_path + 'recon_model/best_valid_psnr.pt',
        network=network,
        dimension=config.method.dimension
    )
