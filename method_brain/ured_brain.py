import torch
import torch.nn.functional as f

from method.shared import architecture_dict, loss_fn_dict

from method_brain.stimulation.dataset_simulation import fmult as fmult_simulation
from method_brain.stimulation.dataset_simulation import ftran as ftran_simulation

from method_brain import trainer_brain

from pkg.torch.callback import pformat_module
from method_brain.stimulation.dataset_simulation import uniformly_cartesian_mask as mri_uniformly_undersampling_mask_simulation


class UREDBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert self.config.method.ured.unfolding_method in ['pnp_fixed']

        arch_dict = architecture_dict(self.config)
        self.net = arch_dict[self.config.method.network]()

        if self.config.method.ured.unfolding_method == 'pnp_fixed':

            self.alpha = torch.nn.Parameter(torch.ones(1) * self.config.method.ured.alpha, requires_grad=False)
            self.gamma = torch.nn.Parameter(torch.ones(1) * self.config.method.ured.gamma_init, requires_grad=False)
            self.tau = torch.nn.Parameter(torch.ones(1) * self.config.method.ured.tau_init, requires_grad=False)

        else:
            raise NotImplementedError()

        if self.config.setting.dataset == 'simulation':
            self.fmult = fmult_simulation
            self.ftran = ftran_simulation
        
    def forward(self, x, y, smps, mask, is_trained=True, is_inference=False):

        is_dc_in_cpu = (is_inference and self.config.method.dimension == 3 and self.config.setting.mode == 'test')

        if not is_dc_in_cpu:
            x, y, smps, mask = [i.cuda() for i in [x, y, smps, mask]]

        if 'pnp' in self.config.method.ured.unfolding_method:
            dc = self.fmult(x, smps, mask, is_3D=(self.config.method.dimension == 3))
            dc = dc - y
            dc = self.ftran(dc, smps, mask, is_3D=(self.config.method.dimension == 3))

            x = x - self.gamma * dc

            prior = x

            prior = self.alpha * self.tau * self.net(prior / self.tau) + (1 - self.alpha) * prior

            return prior

    def load_saved_state_dict(self, state_dict):
        own_state = self.net.state_dict()

        total_parameters = own_state.keys().__len__()

        loaded_parameters = 0
        for name, param in state_dict.items():
            new_name = name.replace('net.', '', 1)
            if new_name not in own_state:
                continue

            if isinstance(param, torch.nn.Parameter):
                param = param.data

            own_state[new_name].copy_(param)
            loaded_parameters += 1
            print('Loading parameter [%d/%d] with name=' % (
                loaded_parameters, total_parameters), new_name, "from the pre_trained model with name=", name)

        assert loaded_parameters == total_parameters


class UnfoldedRegularizerByDenoiser(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net = UREDBlock(config)

        if config.setting.mode == 'train' and config.method.ured.pre_trained_model_path is not None:
            state_dict = torch.load(config.method.ured.pre_trained_model_path)

            print("*********Load pre-trained model from: %s for net in the U-RED*********" % (
                config.method.ured.pre_trained_model_path))
            self.net.load_saved_state_dict(state_dict)
            print("*********End Loading*********")

        print("Parameters of UnfoldedRegularizerByDenoiser: ", pformat_module(self.net)[-1])

        self.rec_fn = loss_fn_dict()[config.method.loss_fn.rec_fn]()

        if self.config.setting.dataset == 'simulation':
            self.weight = torch.from_numpy(
                mri_uniformly_undersampling_mask_simulation(img_size=(256, 232),
                                                            fold=self.config.dataset.simulation.acceleration_factor,
                                                            is_weight_compensate_incomplete=self.config.dataset.simulation.is_weight_compensate_incomplete,
                                                            ACS_percentage=config.dataset.simulation.ACS_percentage,
                                                            is_fixed_ACS=config.dataset.simulation.is_fixed_ACS,
                                                            is_complete=config.dataset.simulation.is_complete
                                                            )[1]
            )
            print("simulation", "self.weight.shape", self.weight.shape)

            self.fmult = fmult_simulation

        else:
            raise NotImplementedError()

        self.weight = torch.stack([self.weight, self.weight], -1)

        if config.method.dimension == 2:
            self.weight = self.weight.unsqueeze(0).unsqueeze(0)
        else:
            self.weight = self.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.weight = self.weight.cuda()

    def forward(self, x0, y, smps, mask, is_trained=False, y_label=None, smps_label=None, mask_label=None, is_inference=False, is_verbose=False):

        x_hat = [x0]
        for i in range(self.config.method.ured.unfolding_step):
            x_hat.append(self.net(x_hat[i], y, smps, mask, is_trained, is_inference))

        if is_trained:

            x_hat_to_input_k_space = self.fmult(x_hat[-1], smps, mask, is_3D=not (self.config.method.dimension == 2))
            x_hat_to_target_k_space = self.fmult(x_hat[-1], smps_label, mask_label, is_3D=not (self.config.method.dimension == 2))

            if self.config.method.loss_fn.is_weighted:
                x_hat_to_input_k_space *= self.weight
                x_hat_to_target_k_space *= self.weight

                y *= self.weight
                y_label *= self.weight

            loss_to_input, loss_to_target = \
                self.rec_fn(x_hat_to_input_k_space, y), self.rec_fn(x_hat_to_target_k_space, y_label)

            loss_total = self.config.method.loss_fn.coff_k_space_target * loss_to_target + self.config.method.loss_fn.coff_k_space_input * loss_to_input

            return x0, [x_hat[-1]], loss_total, loss_to_input, loss_to_target, None

        else:

            return x0, [x_hat[-1]]

def train(
        config
):
    trainer_brain.train(config, UnfoldedRegularizerByDenoiser(config))
