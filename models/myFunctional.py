import torch
import torch.nn.functional as F
from torch.nn.functional import dropout



def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
           + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


def complex_leaky_relu(input, negative_slope):
    return F.leaky_relu(input.real, negative_slope).type(torch.complex64) + \
           1j * F.leaky_relu(input.imag, negative_slope).type(torch.complex64)


def complex_dropout(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    # mask = torch.ones_like(input).type(torch.float32)
    mask = torch.ones(*input.shape, dtype=torch.float32)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(input.dtype)
    return mask.to('cuda') * input


def complex_relu(input):
    return F.relu(input.real).type(torch.complex64) + 1j * F.relu(input.imag).type(torch.complex64)
