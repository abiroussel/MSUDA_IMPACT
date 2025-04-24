from collections import OrderedDict
import torch
import numpy as np

def computeMaxMin(x):
    return np.percentile(x,2), np.percentile(x,98)


def rescale(x, x_min, x_max):
    new_data = (x - x_min) / (x_max - x_min)
    new_data = np.clip(new_data, 0, 1)
    return new_data

def cumulate_EMA(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    return current_weights