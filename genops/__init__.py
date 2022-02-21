from typing import Union, Optional, List

from contextlib import contextmanager
import re
import torch
import numpy as np


"""
    genops

    1.  Use .zeros([1, 2]) instead of .zeros(1, 2)
        BOTH numpy & torch support tuple/list
        ONLY torch supports unpacking operator(*args)

    2.  Use axis=? instead of dim=?
        BOTH numpy & torch support axis=?
        ONLY torch supports dim=?

    3.  Use einops for reduce(max, min, mean)/transpose/[un]squeeze
    
"""

DATA = Union[torch.Tensor, np.ndarray]
DATALIST = Union[List[torch.Tensor], List[np.ndarray], List[List]]
SHAPE = List[int]


TORCH = "TORCH"
NUMPY = "NUMPY"

class Backend:
    FRAMEWORK = TORCH
    TORCH_DEVICE = None
    # TORCH_DTYPE = None


def set_backend(backend, device=None):
    if isinstance(backend, str):
        assert backend in (TORCH, NUMPY)
        Backend.FRAMEWORK = backend
        Backend.TORCH_DEVICE = device
    elif isinstance(backend, torch.Tensor):
        Backend.FRAMEWORK = TORCH
        Backend.TORCH_DEVICE = backend.device
        # Backend.TORCH_DTYPE = backend.dtype
    elif isinstance(backend, np.ndarray):
        Backend.FRAMEWORK = NUMPY


def set_seed(seed):
    if Backend.FRAMEWORK == TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    elif Backend.FRAMEWORK == NUMPY:
        np.random.seed(seed)


def same_arg_and_same_behaviour(function):
    def inner(*args, **kwargs):
        # print(function.__name__)
        if Backend.FRAMEWORK == TORCH:
            kwargs["device"] = Backend.TORCH_DEVICE
            return getattr(torch, function.__name__)(*args, **kwargs)
        elif Backend.FRAMEWORK == NUMPY:
            return getattr(np, function.__name__)(*args, **kwargs)
    return inner


def set_printoptions(precision=None, threshold=None, sci_mode=None):
    if Backend.FRAMEWORK == TORCH:
        torch.set_printoptions(precision=precision,
                               threshold=threshold, sci_mode=sci_mode)
    elif Backend.FRAMEWORK == NUMPY:
        np.set_printoptions(precision=precision,
                            threshold=threshold, suppress=not sci_mode)

#############################################
# Functions for creating array/tensor
#############################################


def create(data: DATA):
    if Backend.FRAMEWORK == TORCH:
        return torch.tensor(data)
    elif Backend.FRAMEWORK == NUMPY:
        return np.array(data)

@same_arg_and_same_behaviour
def zeros(shape: SHAPE):
    pass


@same_arg_and_same_behaviour
def ones(shape: SHAPE):
    pass


@same_arg_and_same_behaviour
def arange(*args, **kwargs):
    pass


def rand(shape):
    """
        Return: Uniform(0, 1)
    """
    if Backend.FRAMEWORK == TORCH:
        return torch.rand(shape, device=Backend.TORCH_DEVICE)
    elif Backend.FRAMEWORK == NUMPY:
        return np.random.random(shape)


def normal(shape: SHAPE):
    """
        Return: Normal(0, 1)
    """
    if Backend.FRAMEWORK == TORCH:
        return torch.normal(mean=0.0, std=1.0, size=shape, device=Backend.TORCH_DEVICE)
    elif Backend.FRAMEWORK == NUMPY:
        return np.random.normal(loc=0.0, scale=1.0, size=shape)


#############################################
# Functions for conversion
#############################################
def convert(data: DATA):
    if Backend.FRAMEWORK == TORCH and isinstance(data, torch.Tensor):
        return data
    elif Backend.FRAMEWORK == NUMPY and isinstance(data, np.ndarray):
        return data
    elif Backend.FRAMEWORK == TORCH and isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif Backend.FRAMEWORK == NUMPY and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


#############################################
# Functions for operating an array/tensor
#############################################
# def argmax(tensor: DATA, axis: Optional[int] = None, keepdim: bool = False):
#     if Backend.FRAMEWORK == TORCH:
#         return torch.argmax(tensor, axis=axis, keepdim=keepdim)
#     elif Backend.FRAMEWORK == NUMPY:
#         return np.argmax(tensor, axis=axis, keepdims=keepdim)
def argmax(tensor: DATA, axis: Optional[int] = None):
    """
        [not sure about the precise version]
        numpy: keepdims not supported before 1.19.2
        torch: axis not supported before 1.10.0
    """
    if Backend.FRAMEWORK == TORCH:
        return torch.argmax(tensor, dim=axis)
    elif Backend.FRAMEWORK == NUMPY:
        return np.argmax(tensor, axis=axis)


#############################################
# Functions for operating arraies/tensors
#############################################
def cat(tensors: DATALIST, axis: int):
    if Backend.FRAMEWORK == TORCH:
        return torch.cat(tensors, axis=axis)
    elif Backend.FRAMEWORK == NUMPY:
        return np.concatenate(tensors, axis=axis)


@same_arg_and_same_behaviour
def stack(*args, **kwargs):
    pass


#############################################
# Functions for einsum
#############################################
def einsum(equation, *operands):
    """
        param:
        ---
        >>> einsumx("bsz seq_len hidden , hidden o -> bsz seqlen o", *operands)
        >>> # same as
        >>> torch.einsum("zyx,xo->zyo", *operands)
    """
    equation = re.sub(",", " , ", equation.strip())
    equation = re.sub("->", " -> ", equation.strip())
    equation = re.sub(r"\s+", " ", equation.strip())
    # print(equation)
    avail_chars = list("abcdefghijklmnopqrstuvwxyz")
    for ele in equation.split(" "):
        if ele in avail_chars:
            avail_chars.remove(ele)
    maps = {}
    conversion = []
    for ele in equation.split(" "):
        if ele in (",", "->"):
            conversion.append(ele)
        elif len(ele) == 1 and ele in "abcdefghijklmnopqrstuvwxyz":
            conversion.append(ele)
        else:
            if ele not in maps:
                maps[ele] = avail_chars.pop()
            conversion.append(maps[ele])
    conversion = "".join(conversion)
    # print(conversion)
    try:
        if Backend.FRAMEWORK == TORCH:
            ret = torch.einsum(conversion, *operands)
        else:
            ret = np.einsum(conversion, *operands)
    except Exception as e:
        print(f"Error occurs! In this function, '{equation}' is converted to '{conversion}'")
        raise e
    return ret

