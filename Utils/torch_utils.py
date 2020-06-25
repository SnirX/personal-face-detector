import torch


def get_torch_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def requires_and_retains_grad(tensor: torch.Tensor):
    tensor.requires_grad = True
    tensor.retain_grad()
