# Add this function into the corresponding file in the CleverHans library
# "cleverhans\torch\utils.py"
"""Utils for PyTorch"""

import numpy as np

import torch


def clip_energy(x, eta, channet, alpha, decay):
    adv_x = x + eta
    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    # clean energy
    chan_x = channet(x)  
    energy_chan_x = torch.max(
        avoid_zero_div, torch.sum(chan_x ** 2, dim=reduc_ind, keepdim=True)
    )

    # difference
    chan_x_adv = channet(adv_x)
    chan_x_differ = chan_x - chan_x_adv
    energy_chan_differ = torch.max(
        avoid_zero_div, torch.sum(chan_x_differ ** 2, dim=reduc_ind, keepdim=True)
    )

    # alpha
    alpha_all = energy_chan_differ / energy_chan_x

    # clip the eta
    while not (alpha_all < alpha).all():
        # decay_factor of all the samples
        decay_factor = 1 * (alpha_all <= alpha) + decay * (alpha_all > alpha)
        # new eta
        eta *= decay_factor
        # nea adv_x
        adv_x = x + eta
        # new energy
        chan_x_adv = channet(adv_x)
        chan_x_differ = chan_x - chan_x_adv
        energy_chan_differ = torch.max(
            avoid_zero_div, torch.sum(chan_x_differ ** 2, dim=reduc_ind, keepdim=True)
        )
        # new alpha
        alpha_all = energy_chan_differ / energy_chan_x

    del x, eta, chan_x_adv, chan_x, chan_x_differ, avoid_zero_div, energy_chan_x, energy_chan_differ
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    return adv_x
