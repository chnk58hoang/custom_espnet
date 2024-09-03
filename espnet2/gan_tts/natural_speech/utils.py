import torch


def get_sequence_mask(x_lengths: torch.Tensor,
                      max_length: int):
    """
    Args:
        x_lengths: tensor(B)
    Return:
        mask: tensor(B, max_length)
    """
    if x_lengths.max() <= max_length:
        max_length = x_lengths.max().item()
    idx = torch.arange(0, max_length, step=1)
    return idx.unsqueeze(0) < x_lengths.unsqueeze(1)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)