import numpy as np
import tifffile as tiff
import torch
import nibabel as nib


def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[i] -= np.amin(x[i])
                x[i] /= np.amax(x[i])

                x[i] *= 255

    else:
        n_slice, n_x, n_y, n_c = x.shape

    if is_normalized:
        x = x.astype(np.uint8)

    tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})


def to_nifti1(img_dict, save_path):

    for key_ in img_dict:
        print(key_, img_dict[key_].shape)
        img_ = img_dict[key_]

        nib.nifti1.save(nib.Nifti1Image(img_, None), save_path + key_ + '.img')
