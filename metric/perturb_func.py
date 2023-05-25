import torch
import numpy as np
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def perturb_fn(inputs, indices=None, device=device, mean=0.0, std=1.0, faith=False):
    # torch.manual_seed(42)
    # noise = torch.tensor(np.random.normal(0, 1, inputs.shape)).float().to(device)
    noise = torch.normal(mean=mean, std=std, size=inputs.shape).to(device)
    if faith:
        return inputs.to(device) - noise
    else:
        return noise, inputs.to(device) - noise


# testing
def perturb_fn1(inputs):
    # torch.manual_seed(42)
    noise = torch.tensor(np.random.normal(0, 1, inputs.shape)).float().to(device)
    # noise = torch.normal(mean=mean, std=std, size=inputs.shape).to(device)
    return noise, inputs.to(device) - noise

def perturb_fn2(inputs):
    # torch.manual_seed(42)
    noise = torch.tensor(np.random.normal(0, 1, inputs.shape)).float()
    # noise = torch.normal(mean=mean, std=std, size=inputs.shape).to(device)
    return noise, inputs - noise
