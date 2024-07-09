import os
import json
import jsonargparse
from pkg.utility.common import get_dict_key_iterate
import torch
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def main():

    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = jsonargparse.ArgumentParser(default_config_files=['config.json'])
    for k in get_dict_key_iterate(config):
        parser.add_argument('--' + k)

    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.setting.gpu_index

    from method_brain import deq_ured_brain as selfDEQ

    method_dict = {
        'selfDEQ': selfDEQ,
    }
    
    if config.setting.mode == 'train':

        dec = config.setting.save_folder

        config.setting.save_folder = dec + '[%d]_SelfDEQ_B[%d]Ep[%d]_%s_simulation_F[%d]S[%.1f]ATimes[%d]_is_full_rank=%s_is_supervised=%s_is_weighted=%s' % (
            0, config.train.batch_size, config.train.train_epoch, config.setting.method, config.dataset.simulation.acceleration_factor, config.dataset.simulation.sigma, config.dataset.simulation.acquisition_times, config.dataset.simulation.is_complete, config.dataset.simulation.is_supervised, config.method.loss_fn.is_weighted)

        if config.setting.mode == 'train':
            method_dict[config.setting.method].train(config)

    elif config.setting.mode == 'test':

            print("################")
            print("# Testing Full Rank Weighted R = 8")
            print("################")
            config.setting.save_folder = 'full_rank_weighted'
            method_dict[config.setting.method].test(config)

if __name__ == '__main__':
    main()
