import torch
from tqdm import tqdm
import numpy as np

from method_brain import trainer_brain
from method_brain import tester_brain
from method_brain.ured_brain import UREDBlock

from method_brain.stimulation.dataset_simulation import fmult as fmult_simulation
from method_brain.stimulation.dataset_simulation import ftran as ftran_simulation

from pkg.torch.callback import pformat_module
import torch.autograd as autograd
from method.shared import architecture_dict, loss_fn_dict
from method_brain.stimulation.dataset_simulation import uniformly_cartesian_mask as mri_uniformly_undersampling_mask_simulation


def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0, is_verbose=False):
    """ Anderson acceleration for fixed point iteration. """

    if len(x0.shape) == 5:
        bsz, d, Z, H, W = x0.shape
        X = torch.zeros(bsz, m, d * Z * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d * Z * H * W, dtype=x0.dtype, device=x0.device)

    elif len(x0.shape) == 4:
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    else:
        raise NotImplementedError()

    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []

    iter_ = range(2, max_iter)
    if is_verbose:
        iter_ = tqdm(iter_, desc='anderson_solver')

    for k in iter_:
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if is_verbose:
            iter_.set_description("forward_res: %.4f" % res[-1])

        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res


def generic_solver(f, x0, tol=1e-4, max_iter=50):
    f0 = f(x0)
    res = []
    fo_List = [f0]

    for k in range(max_iter):
        x = f0
        f0 = f(x)
        fo_List.append(f0)

        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if res[-1] < tol:
            break

    return f0, res


class UnfoldedRegularizerByDenoiser(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net = UREDBlock(config)

        if (config.setting.mode == 'train' or config.setting.mode == 'debug') and config.method.ured.pre_trained_model_path is not None:
            state_dict = torch.load(config.method.ured.pre_trained_model_path)

            print("*********Load pre-trained model from: %s for net in the U-RED*********" % (
                config.method.ured.pre_trained_model_path))
            self.net.load_saved_state_dict(state_dict)
            print("*********End Loading*********")

        print("Parameters of UnfoldedRegularizerByDenoiser: ", pformat_module(self.net)[-1])

        self.forward_res, self.backward_res = 0.0, 0.0

        self.rec_fn = loss_fn_dict()[config.method.loss_fn.rec_fn]()

        if self.config.setting.dataset == 'simulation':
            self.weight = torch.from_numpy(
                mri_uniformly_undersampling_mask_simulation(
                    img_size=(256, 232), fold=self.config.dataset.simulation.acceleration_factor,
                    is_complete=self.config.dataset.simulation.is_complete,
                    is_weight_compensate_incomplete=self.config.dataset.simulation.is_weight_compensate_incomplete,
                    ACS_percentage=config.dataset.simulation.ACS_percentage,
                    is_fixed_ACS=config.dataset.simulation.is_fixed_ACS,
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

    def forward(self, x0, y, smps, mask, is_trained=True, y_label=None, smps_label=None, mask_label=None, is_inference=False, is_verbose=False):

        with torch.no_grad():
            z_fixed, forward_res = anderson_solver(
                lambda x: self.net(x, y, smps, mask, is_trained, is_inference), x0,
                max_iter=self.config.method.ured.forward_num_step - 1,
                tol=self.config.method.ured.tol,
                is_verbose=is_verbose,
            )

            forward_res = torch.from_numpy(np.array(forward_res[-1])).cuda()
            forward_res = torch.unsqueeze(forward_res, 0)

        z = self.net(z_fixed, y, smps, mask, is_trained, is_inference)

        if is_trained and not self.config.method.ured.is_jacobian_free:

            z0 = z.clone().detach().requires_grad_()
            f0 = self.net(z0, y, smps, mask, is_trained)

        def backward_hook(grad):
            g, backward_res = anderson_solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad,
                max_iter=self.config.method.ured.backward_num_step,
                tol=self.config.method.ured.tol)

            return g

        if is_trained and not self.config.method.ured.is_jacobian_free:
            z.register_hook(backward_hook)

        if is_trained:

            if self.config.setting.dataset == 'simulation' and self.config.dataset.simulation.is_supervised:
                x_label = ftran_simulation(y_label, smps, mask_label)
                loss_to_input = self.rec_fn(z, x_label)
                loss_to_target = self.rec_fn(z, x_label)

            else:
                x_hat_to_input_k_space = self.fmult(z, smps, mask, is_3D=not (self.config.method.dimension == 2))
                x_hat_to_target_k_space = self.fmult(z, smps_label, mask_label, is_3D=not (self.config.method.dimension == 2))

                if self.config.method.loss_fn.is_weighted:
                    x_hat_to_input_k_space *= self.weight
                    x_hat_to_target_k_space *= self.weight

                    y *= self.weight
                    y_label *= self.weight

                loss_to_input, loss_to_target = \
                    self.rec_fn(x_hat_to_input_k_space, y), self.rec_fn(x_hat_to_target_k_space, y_label)

            loss_total = self.config.method.loss_fn.coff_k_space_target * loss_to_target + self.config.method.loss_fn.coff_k_space_input * loss_to_input

            return x0, [z], loss_total, loss_to_input, loss_to_target, forward_res

        else:

            return x0, [z]


def train(
        config
):
    trainer_brain.train(config, UnfoldedRegularizerByDenoiser(config))

def test(
        config
):
    tester_brain.test(config, UnfoldedRegularizerByDenoiser(config))