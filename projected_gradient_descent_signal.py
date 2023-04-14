# Add this function into the corresponding file in the CleverHans library
# "cleverhans\torch\attacks\projected_gradient_descent.py"
"""The Projected Gradient Descent attack.(for signal)"""
import numpy as np
import torch

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import clip_energy


def projected_gradient_descent_signal(
        model_fn,
        x,
        eps,
        eps_iter,
        nb_iter,
        norm,
        channet,                             # new param
        alpha=0.1,                           # new param
        decay=0.5,                           # new param
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        sanity_checks=False,
):

    if eps == 0:
        return x

    if eps_iter == 0:
        return x

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta 
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    # if clip_min is not None or clip_max is not None:
    #     adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        # if clip_min is not None or clip_max is not None:
        #     adv_x = torch.clamp(adv_x, clip_min, clip_max)

        # use clip_energy to constrain the perturbation according intermediate power variation
        adv_x = clip_energy(x, eta, channet, alpha, decay)

        i += 1

    del eta, norm, x
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    return adv_x
