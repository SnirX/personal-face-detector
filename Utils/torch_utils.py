import torch


def get_torch_device():
    torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
