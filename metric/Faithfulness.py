from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from metric.perturb_func import *
from metric.similarity_func import *
from captum.metrics import *

torch.manual_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InfidelityScore:
    def __init__(
            self,
            perturb_func: Callable = None,
            device=None,
            n_perturb_samples=25,
            normalize=True
    ):
        self.perturb_func = perturb_func
        self.device = device
        self.n_perturb_samples = n_perturb_samples
        self.normalize = normalize

        if self.perturb_func is None:
            self.perturb_func = perturb_fn1

        if self.device is None:
            self.device = device

    def evaluate_instance(
        self,
        model,
        x: torch.tensor, # size: (batch, channel, width, height)
        label,
        a: torch.tensor, # size: (batch, channel, width, height)
    ):
        score = infidelity(forward_func=model.to(self.device), perturb_func=self.perturb_func,
                           inputs=x.to(self.device), target=label,
                            attributions=a.to(self.device),
                            n_perturb_samples=self.n_perturb_samples, normalize=self.normalize)

        return score

    def evaluate_instance2(
        self,
        model,
        x: torch.tensor, # size: (batch, channel, width, height)
        label,
        a: torch.tensor, # size: (batch, channel, width, height)
    ):
        score = infidelity(forward_func=model, perturb_func=self.perturb_func,
                           inputs=x, target=label,
                            attributions=a,
                            n_perturb_samples=self.n_perturb_samples, normalize=self.normalize)

        return score