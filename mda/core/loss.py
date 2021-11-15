import torch


def sequence_diff(pred_all, target_all, count=None):
    """Caculate the shape difference between squence

    :param pred_all: All predicited sequences
    :type pred_all: torch.Tensor
    :param target_all: All target sequences
    :type target_all: torch.Tensor
    :param count: List of start time of each sequence
    :type count: List
    """
    N, C, T, V = pred_all.size()
    loss = torch.abs(pred_all - target_all).sum(dim=(1, 3)).mean()
    return loss
