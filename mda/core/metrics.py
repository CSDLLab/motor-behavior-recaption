import torch
import torch.nn as nn
import torch.nn.functional as F


def minmax(tensor, dim=2):
    t_min, _ = tensor.min(dim, keepdim=True)
    t_max, _ = tensor.max(dim, keepdim=True)
    return (tensor - t_min) / (t_max - t_min)


def ade(pred_all, target_all, count):
    """Compute the all sequence loss

    :param pred_all: All predictions from model
    :type pred_all: torch.Tensor
    :param target_all: All target for testing
    :type target_all: torch.Tensor
    :param count: Start time step of prediction
    :type count: int
    """
    num_all = len(pred_all)
    sum_all = 0
    for s in range(num_all):
        pred = pred_all[s][:, :count[s], :]
        target = target_all[s][:, :count[s], :]
        N, T = pred.shape[1], pred.shape[0]
        sum_ = torch.norm((pred - target), dim=-1).sum()
        sum_all += sum_/(N*T)
    return sum_all / num_all


def fde(pred_all, target_all, count):
    """Compute the all sequence loss

    :param pred_all: All predictions from model
    :type pred_all: torch.Tensor
    :param target_all: All target for testing
    :type target_all: torch.Tensor
    :param count: Start time step of prediction
    :type count: int
    """
    num_all = len(pred_all)
    sum_all = 0.0
    for s in range(num_all):
        pred = pred_all[s][-1:, :count[s], :]
        target = target_all[s][-1: :count[s], :]
        N = pred.shape[1]
        sum_ = torch.norm((pred - target), dim=-1).sum()
        sum_all += sum_ / N

    return sum_all / num_all


def geometric_diff(a, b):
    """Calculate the shape difference between a and b

    :param a: Shape features with [[length, width, angle, cx, cy], ...]
    :type a: torch.Tensor
    :param b: Shape features with [[length, width, angle, cx, cy], ...]
    :type b: torch.Tensor
    """
    res = a[:, :2] - b[:, :2]
    return res.norm()
