from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import torch
import datetime
import shutil
import pprint

from pkg.utility.common import check_and_mkdir
from pkg.torch.metrics import Concatenate, Stack, Mean
from pkg.utility.io import to_nifti1
from pkg.utility.measure import compare_snr, compare_rsnr, compare_psnr, compare_rpsnr, compare_ssim, compare_rssim
from .shared_brain import predict2dicom, predict2dicom_2d, grappa2dicom_2d
from .stimulation.dataset_simulation import Dataset as Dataset_Simulation


def evaluate_helper_base(im_test, im_true, is_normalized=False):

    total, snr, psnr, ssim, rsnr, rpsnr, rssim = 0, 0, 0, 0, 0, 0, 0

    for i in range(im_test.shape[0]):
        for j in range(im_test.shape[1]):
            x_hat_tmp = im_test[i, j].unsqueeze(0).unsqueeze(0).clone()
            dicom_tmp = im_true[i, j].unsqueeze(0).unsqueeze(0).clone()

            if is_normalized:
                x_hat_tmp = x_hat_tmp - torch.min(x_hat_tmp)
                x_hat_tmp = x_hat_tmp / torch.max(x_hat_tmp)

                dicom_tmp = dicom_tmp - torch.min(dicom_tmp)
                dicom_tmp = dicom_tmp / torch.max(dicom_tmp)

            snr_tmp, psnr_tmp, ssim_tmp, rsnr_tmp, rpsnr_tmp, rssim_tmp = [f(x_hat_tmp, dicom_tmp).item() for f in [
                compare_snr, compare_psnr, compare_ssim, compare_rsnr, compare_rpsnr, compare_rssim
            ]]

            snr, psnr, ssim, rsnr, rpsnr, rssim = [i + j for i, j in zip(
                [snr_tmp, psnr_tmp, ssim_tmp, rsnr_tmp, rpsnr_tmp, rssim_tmp],
                [snr, psnr, ssim, rsnr, rpsnr, rssim])]

            total += 1

    snr, psnr, ssim, rsnr, rpsnr, rssim = [i / total for i in [snr, psnr, ssim, rsnr, rpsnr, rssim]]

    return snr, psnr, ssim, rsnr, rpsnr, rssim


def evaluate_helper(im_test, im_true, is_normalized=False):

    return evaluate_helper_base(im_test, im_true[..., :-1], is_normalized=is_normalized)


def abs_helper(x):
    x = abs(x).detach().cpu()
    return x[..., 128:-128, :]


def test_helper(
        config,
        dataloader,
        save_path,
        checkpoint,
        network,
        dimension=3,
        global_epoch=1,
        total_epoch=1,
):

    images = Concatenate()
    metrics = Stack()
    metrics_mean = Mean()

    with torch.no_grad():

        iter_ = tqdm(dataloader, desc='Test [%.3d/%.3d]' % (global_epoch, total_epoch), total=len(dataloader))
        for i, test_data in enumerate(iter_):
            dicom, x0, y_input, smps_input, mask_input, grappa = test_data

            _, x_hat = network(x0, y_input, smps_input, mask_input, is_trained=False, is_inference=True, is_verbose=True if config.method.dimension == 3 else False)

            x_hat = x_hat[-1]

            x_hat = predict2dicom_2d(x_hat, is_crop=config.setting.dataset == 'brain')
            x0 = predict2dicom_2d(x0, is_crop=config.setting.dataset == 'brain')
            grappa = grappa2dicom_2d(grappa, is_crop=config.setting.dataset == 'brain')

            if config.setting.dataset == 'simulation':
                dicom = grappa2dicom_2d(grappa, is_crop=False)

            grappa = grappa.unsqueeze(1)
            x0 = x0.unsqueeze(1)
            x_hat = x_hat.unsqueeze(1)
            dicom = dicom.unsqueeze(1)

            if config.setting.dataset == 'brain':
                dicom = dicom[..., :-1]

            snr, psnr, ssim, rsnr, rpsnr, rssim = evaluate_helper_base(x_hat, grappa, is_normalized=config.method.dimension == 3)

            metrics_batch = {
                'psnr(x_hat)': psnr,
                'ssim(x_hat)': ssim,
            }

            snr, psnr, ssim, rsnr, rpsnr, rssim = evaluate_helper_base(x0, grappa, is_normalized=config.method.dimension == 3)

            metrics_batch.update({
                'psnr(x0)': psnr,
                'ssim(x0)': ssim,
            })

            metrics.update_state(metrics_batch)
            metrics_mean.update_state(metrics_batch)

            images.update_state({
                'x0': x0.detach().cpu().numpy(),
                'x_hat': x_hat.detach().cpu().numpy(),
                'x': grappa.detach().cpu().numpy(),
            })

        print("Quantitative results: ")
        print("")
        pprint.pprint(metrics_mean.result())
        print("")

        check_and_mkdir(save_path)

        check_and_mkdir(save_path + 'recon_model/')
        shutil.copy(checkpoint, save_path + 'recon_model/')

        print("Writing results....")

        img_dict = images.result()
        img_dict_reshape = {}
        for k in img_dict:
            img_ = img_dict[k]

            img_ = np.transpose(img_, [3, 2, 0, 1])

            img_dict_reshape[k] = img_

        to_nifti1(img_dict=img_dict_reshape, save_path=save_path)

        log_dict = metrics.result()

        cvs_data = np.array(list(log_dict.values()))
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ''
        for k in log_dict:
            cvs_header = cvs_header + k + ','

        np.savetxt(save_path + 'metrics.csv', cvs_data_with_index, delimiter=',', fmt='%.5f',
                   header='index,' + cvs_header)
        np.savetxt(save_path + 'metrics_mean.csv', cvs_data_mean, delimiter=',', fmt='%.5f', header=cvs_header)

def test(
        config,
        network,
):

    if config.setting.dataset == 'simulation':
        Dataset = Dataset_Simulation
    else:
        raise NotImplementedError()

    test_dataset = Dataset(
        mode='test',
        config=config)

    print("[test_dataset] total_len: ", test_dataset.__len__())

    file_path = config.setting.root_path + config.setting.save_folder + '/'

    network.load_state_dict(torch.load(file_path + config.test.checkpoint), strict=True)
    network.cuda()

    save_path = file_path + '/TEST_' + datetime.datetime.now().strftime("%m%d%H%M") + \
        '_' + config.setting.save_folder + '/'

    checkpoint = file_path + config.test.checkpoint

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    test_helper(
        config=config,
        dataloader=test_dataloader,
        save_path=save_path,
        checkpoint=checkpoint,
        network=network,
        dimension=config.method.dimension
    )
