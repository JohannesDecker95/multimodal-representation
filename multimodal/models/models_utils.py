import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from torch.nn import functional as F
from torch.distributions import Normal


def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def sample_gaussian(m, v, device):

    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    return z


def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def product_of_experts(m_vect, v_vect):

    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    return mu, var


def duplicate(x, rep):
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def depth_deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, 16, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True
        ),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(
            16, out_planes, kernel_size=4, stride=2, padding=1, bias=True
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    image_transform = image * scale
    return image_transform.transpose(1, 3).transpose(2, 3)


def filter_depth(depth_image):
    depth_image = torch.where(
        depth_image > 1e-7, depth_image, torch.zeros_like(depth_image)
    )
    return torch.where(depth_image < 2, depth_image, torch.zeros_like(depth_image))

def remove_zeros(tensor):
    """
    This function takes a torch tensor as input and adds 1e-9 to each element of the tensor until it no longer contains
    any zero values. The modified tensor is then returned.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    tensor (torch.Tensor): The modified tensor with no zero values.
    """
    # Check if the tensor contains any zero values
    while torch.any(tensor == 0):
        # If it does, add 1e-9 to each element of the tensor
        tensor += 1e-9

    # Return the modified tensor
    return tensor