import h5py
import numpy as np
import torch
import os
import tqdm
import matplotlib.pyplot as plt


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uniformly_cartesian_mask(img_size, fold, ACS_percentage: float = 0.2, is_complete=True, is_fixed_ACS = False, is_weight_compensate_incomplete=True):
    
    ny = img_size[-1]

    if not is_fixed_ACS:
        ACS_START_INDEX = (ny // 2) - (int(ny * ACS_percentage * (2 / fold)) // 2)
        ACS_END_INDEX = (ny // 2) + (int(ny * ACS_percentage * (2 / fold)) // 2)

    else:
        ACS_START_INDEX = (ny // 2) - (int(ny * ACS_percentage) // 2)
        ACS_END_INDEX = (ny // 2) + (int(ny * ACS_percentage) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    sampling_rate = []
    mask = np.zeros(shape=(fold,) + img_size, dtype=np.float32)
    weight = np.ones(shape=img_size, dtype=np.float32)

    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1
    if is_weight_compensate_incomplete:
        weight[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1 / (fold if is_complete else fold // 2)
    else:
        weight[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1 / fold

    for i in range(ny):
        for j in range(fold):
            if i % fold == j:
                mask[j, ..., i] = 1

    for i in range(fold):
        sampling_rate.append(np.nonzero(mask[i])[0].size / mask[i].size)

    acs_size = ACS_END_INDEX - ACS_START_INDEX

    return mask, weight, sampling_rate, acs_size


def load_MoDL_dataset(
        ACS_percentage,
        is_fixed_ACS,
        root_path: str = '',
        verbose: bool = False,
        factor=4,
        sigma=0.0,
        acquisition_times=2,
        is_complete: bool = True,
):
    if verbose:
        print("RUNNING load_MoDL_dataset ...")

    if verbose:
        print("")
        print("========================================================")
        print("attempt to find mri h5 files")
        print("========================================================")
        print("")

    mri_folder = root_path + 'mri_factor%d_sigma%.2f_acquisition_times%d_is_complete%s_ACS_percentage%s_is_fixed_ACS%s/' % (
    factor, sigma, acquisition_times, str(is_complete), str(ACS_percentage), str(is_fixed_ACS))
    check_and_mkdir(mri_folder)

    mri_folder_qc = root_path + 'mri_factor%d_sigma%.2f_acquisition_times%d_is_complete%s_ACS_percentage%s_is_fixed_ACS%s_qc/' % (
    factor, sigma, acquisition_times, str(is_complete), str(ACS_percentage), str(is_fixed_ACS))
    check_and_mkdir(mri_folder_qc)

    mri_h5 = mri_folder + 'mri_factor%d_sigma%.2f_acquisition_times%d_is_complete%s_ACS_percentage%s_is_fixed_ACS%s.h5' % (
    factor, sigma, acquisition_times, str(is_complete), str(ACS_percentage), str(is_fixed_ACS))

    if os.path.exists(mri_h5):
        if verbose:
            print("find mri h5 file:", mri_h5)

    else:

        with h5py.File(root_path + 'dataset.hdf5', 'r') as f:
            # shape: 256, 232

            num_slice, num_coil, num_x, num_y = f['trnCsm'].shape

            x_train = torch.zeros(size=[num_slice, num_x, num_y], dtype=torch.complex64)
            smps_train = torch.zeros(size=[num_slice, num_coil, num_x, num_y], dtype=torch.complex64)
            y_train = torch.zeros(size=[acquisition_times, num_slice, num_coil, num_x, num_y], dtype=torch.complex64)

            y_label = torch.zeros(size=[num_slice, num_coil, num_x, num_y], dtype=torch.complex64)
            x0_train = torch.zeros(size=[acquisition_times, num_slice, 2, num_x, num_y], dtype=torch.float32)

            mask_train_total, weight_train, _, _ = uniformly_cartesian_mask(
                (num_x, num_y),
                fold=factor,
                ACS_percentage=ACS_percentage,
                is_fixed_ACS=is_fixed_ACS
            )

            if not is_complete:
                mask_train_total = mask_train_total[:mask_train_total.shape[0] // 2]

            mask_train = torch.zeros(size=[acquisition_times, num_slice, num_x, num_y], dtype=torch.float32)

            iter_ = tqdm.tqdm(range(num_slice))
            for i in iter_:
                x_tmp = f['trnOrg'][i]
                smps_tmp = f['trnCsm'][i]

                x_tmp_angle = np.angle(x_tmp)

                x_tmp_abs = abs(x_tmp)
                x_tmp_abs -= np.amin(x_tmp_abs)
                x_tmp_abs /= np.amax(x_tmp_abs)

                x_tmp = x_tmp_abs * np.exp(1j * x_tmp_angle)

                x_tmp = torch.from_numpy(x_tmp)
                y_tmp = torch.unsqueeze(x_tmp, 0) * smps_tmp
                y_tmp = torch.fft.fft2(y_tmp)
                y_tmp = torch.fft.fftshift(y_tmp, [-1, -2])

                y_label[i] = y_tmp

                mask_indexes = torch.randperm(mask_train_total.shape[0])[:acquisition_times]

                iter_.write("mask_indexes: %s" % str(mask_indexes))
                iter_.update()

                mask_train_tmp = mask_train_total[mask_indexes]

                for ii in range(acquisition_times):
                    y_tmp_ii = y_tmp * torch.unsqueeze(torch.from_numpy(mask_train_tmp[ii]), 0)

                    if sigma > 0:
                        y_tmp_ii = y_tmp_ii + torch.randn(y_tmp_ii.shape) * sigma

                    x0_tmp_ii = ftran(
                        torch.view_as_real(y_tmp_ii),
                        torch.view_as_real(torch.from_numpy(smps_tmp)),
                        torch.from_numpy(mask_train_tmp[ii]),
                        dim=0)

                    x0_train[ii, i] = x0_tmp_ii
                    y_train[ii, i] = y_tmp_ii
                    mask_train[ii, i] = torch.from_numpy(mask_train_tmp[ii])

                x_train[i] = x_tmp
                smps_train[i] = torch.from_numpy(smps_tmp)

            num_slice, num_coil, num_x, num_y = f['tstCsm'].shape

            x_test = torch.zeros(size=[num_slice, num_x, num_y], dtype=torch.complex64)
            smps_test = torch.zeros(size=[num_slice, num_coil, num_x, num_y], dtype=torch.complex64)
            y_test = torch.zeros(size=[acquisition_times, num_slice, num_coil, num_x, num_y], dtype=torch.complex64)
            mask_test_total, weight_test, _, _ = uniformly_cartesian_mask(
                (num_x, num_y), fold=factor,
                ACS_percentage=ACS_percentage,
                is_fixed_ACS=is_fixed_ACS)
            x0_test = torch.zeros(size=[acquisition_times, num_slice, 2, num_x, num_y], dtype=torch.float32)

            if not is_complete:
                mask_test_total = mask_test_total[:mask_test_total.shape[0] // 2]

            mask_test = torch.zeros(size=[acquisition_times, num_slice, num_x, num_y], dtype=torch.float32)

            iter_ = tqdm.tqdm(range(num_slice))
            for i in iter_:
                x_tmp = f['tstOrg'][i]
                smps_tmp = f['tstCsm'][i]

                x_tmp_angle = np.angle(x_tmp)

                x_tmp_abs = abs(x_tmp)
                x_tmp_abs -= np.amin(x_tmp_abs)
                x_tmp_abs /= np.amax(x_tmp_abs)

                x_tmp = x_tmp_abs * np.exp(1j * x_tmp_angle)

                x_tmp = torch.from_numpy(x_tmp)
                y_tmp = torch.unsqueeze(x_tmp, 0) * smps_tmp
                y_tmp = torch.fft.fft2(y_tmp)
                y_tmp = torch.fft.fftshift(y_tmp, [-1, -2])

                mask_indexes = torch.randperm(mask_test_total.shape[0])[:acquisition_times]

                iter_.write("mask_indexes: %s" % str(mask_indexes))
                iter_.update()

                mask_test_tmp = mask_test_total[mask_indexes]

                for ii in range(acquisition_times):
                    y_tmp_ii = y_tmp * torch.unsqueeze(torch.from_numpy(mask_test_tmp[ii]), 0)

                    if sigma > 0:
                        y_tmp_ii = y_tmp_ii + torch.randn(y_tmp_ii.shape) * sigma

                    x0_tmp_ii = ftran(
                        torch.view_as_real(y_tmp_ii),
                        torch.view_as_real(torch.from_numpy(smps_tmp)),
                        torch.from_numpy(mask_test_tmp[ii]),
                        dim=0)

                    x0_test[ii, i] = x0_tmp_ii
                    y_test[ii, i] = y_tmp_ii
                    mask_test[ii, i] = torch.from_numpy(mask_test_tmp[ii])

                x_test[i] = x_tmp
                smps_test[i] = torch.from_numpy(smps_tmp)

            dict_ = {
                'x_train': x_train,
                'smps_train': smps_train,
                'y_train': y_train,
                'mask_train': mask_train,
                'weight_train': weight_train,
                'x0_train': x0_train,
                'y_label': y_label,

                'x_test': x_test,
                'x0_test': x0_test,
                'smps_test': smps_test,
                'y_test': y_test,
                'mask_test': mask_test,
                'weight_test': weight_test,
            }

        with h5py.File(mri_h5, 'w') as f:

            for k in dict_:
                f.create_dataset(name=k, data=dict_[k])

    return mri_h5


class Dataset:
    def __init__(
            self,
            config,
            mode
    ):

        self.mode = mode
        self.config = config

        factor = config.dataset.simulation.acceleration_factor

        h5_file = load_MoDL_dataset(
            root_path=config.dataset.simulation.root_path,
            factor=factor,
            sigma=config.dataset.simulation.sigma,
            acquisition_times=config.dataset.simulation.acquisition_times,
            is_complete=config.dataset.simulation.is_complete,
            ACS_percentage=config.dataset.simulation.ACS_percentage,
            is_fixed_ACS=config.dataset.simulation.is_fixed_ACS,
        )

        with h5py.File(h5_file, 'r') as f:

            if mode == 'train':

                self.x = f['x_train'][:300]
                self.smps = f['smps_train'][:300]
                self.y = f['y_train'][:, :300]
                self.mask = f['mask_train'][:, :300]

                self.x0 = f['x0_train'][:, :300]
                self.y_label = f['y_label'][:300]

                self.weight = f['weight_train'][:]

            elif mode == 'valid':

                self.x = f['x_train'][300:]
                self.smps = f['smps_train'][300:]
                self.y = f['y_train'][:, 300:]
                self.mask = f['mask_train'][:, 300:]

                self.x0 = f['x0_train'][:, 300:]
                self.y_label = f['y_label'][300:]

                self.weight = f['weight_train'][:]

            elif mode == 'test':


                self.x = f['x_test'][:]

                self.smps = f['smps_test'][:]
                self.y = f['y_test'][:]
                self.mask = f['mask_test'][:]
                self.weight = f['weight_test'][:]

                self.x0 = f['x0_test'][:]

            else:

                raise ValueError()

        self.indexes_map = []

        num_slice = self.x.shape[0]
        factor = self.y.shape[0]

        for index_slice in range(num_slice):

            if mode == 'train':

                for index_input_mask in range(self.mask.shape[0]):
                    for index_label_mask in range(self.mask.shape[0]):

                        if index_label_mask != index_input_mask:
                            self.indexes_map.append([index_slice, index_input_mask, index_label_mask])

            else:

                self.indexes_map.append([index_slice, ])

    def __len__(self):
        return len(self.indexes_map)

    def __getitem__(self, item):
        if self.mode == 'train':

            index_slice, index_input_mask, index_label_mask = self.indexes_map[item]

            x = torch.from_numpy(self.x[index_slice])
            smps = torch.from_numpy(self.smps[index_slice])

            y_input = torch.from_numpy(self.y[index_input_mask, index_slice])
            mask_input = torch.from_numpy(self.mask[index_input_mask, index_slice])

            y_label = torch.from_numpy(self.y[index_label_mask, index_slice])
            mask_label = torch.from_numpy(self.mask[index_label_mask, index_slice])

            y_input = torch.view_as_real(y_input)
            y_label = torch.view_as_real(y_label)

            smps = torch.view_as_real(smps)

            x0 = torch.from_numpy(self.x0[index_input_mask, index_slice])

            if self.config.dataset.simulation.is_supervised:
                mask_label = torch.ones_like(mask_label)

                y_label = torch.from_numpy(self.y_label[index_slice])

            return x, x0, y_input, y_label, smps, smps, mask_input, mask_label, x

        else:

            index_slice, = self.indexes_map[item]

            x = torch.from_numpy(self.x[index_slice])
            y = torch.from_numpy(self.y[0, index_slice])
            mask = torch.from_numpy(self.mask[0, index_slice])
            smps = torch.from_numpy(self.smps[index_slice])

            y = torch.view_as_real(y)
            smps = torch.view_as_real(smps)

            # x0 = ftran(y, smps, mask, dim=0)

            x0 = torch.from_numpy(self.x0[0, index_slice])

            return x, x0, y, smps, mask, x


def ftran(y, smps, mask, dim=1, is_3D=True, is_combined=True):
    # assert is_combined is True
    # assert is_3D is False

    y = torch.view_as_complex(y)

    #
    if is_combined:
        smps = torch.view_as_complex(smps)
    #
    # if is_3D:
    #     y = y * mask.unsqueeze(dim).unsqueeze(dim)
    # else:
    #     y = y * mask.unsqueeze(dim)

    y = y * mask.unsqueeze(dim)

    img = torch.fft.ifft2(y)
    # img = torch.fft.fftshift(img, -1)
    # img = torch.fft.fftshift(img, -2)

    # if is_combined:

    img = img * torch.conj(smps)
    img = img.sum(dim)

    img = torch.stack([img.real, img.imag], dim)

    y = torch.view_as_real(y)

    if is_combined:
        smps = torch.view_as_real(smps)

    return img


def fmult(x, smps, mask, dim=1, is_3D=True, is_combined=True):
    if is_combined:
        smps = torch.view_as_complex(smps)

    if dim == 1:
        x = torch.complex(x[:, 0], x[:, 1])
    elif dim == 0:
        x = torch.complex(x[0], x[1])
    else:
        raise NotImplementedError()

    x = x.unsqueeze(dim)

    # print(x.shape, x.dtype, smps.shape, smps.dtype)

    x = x * smps

    # if is_combined:
    #     x = x * smps
    #
    # x = torch.fft.ifftshift(x, -2)
    # x = torch.fft.ifftshift(x, -1)

    y = torch.fft.fft2(x)
    # y = torch.fft.fftshift(y, [-1, -2])

    # if is_3D:
    #     y = y * mask.unsqueeze(dim).unsqueeze(dim)
    # else:
    #     y = y * mask.unsqueeze(dim)

    y = y * mask.unsqueeze(dim)

    y = torch.view_as_real(y)

    # if is_combined:
    #     smps = torch.view_as_real(smps)

    return y
