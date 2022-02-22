from typing import Union, Optional, List

import re
import torch
import numpy as np
from scipy import sparse

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

BASIC_DATA_LIST = List[Union[List, int, float, bool]]
TENSOR = Union[torch.Tensor, np.ndarray]
TENSOR_LIKE = Union[TENSOR, BASIC_DATA_LIST]
TENSOR_LIST = Union[List[torch.Tensor], List[np.ndarray], List[List]]
TENSOR_SHAPE = List[int]

TORCH = "TORCH"
NUMPY = "NUMPY"


class Backend:
    FRAMEWORK = TORCH
    TORCH_DEVICE = None
    # TORCH_DTYPE = None


def set_backend(backend, device=None):
    assert backend in (TORCH, NUMPY)
    Backend.FRAMEWORK = backend
    Backend.TORCH_DEVICE = device


def set_backend_as(backend: TENSOR):
    assert isinstance(backend, (torch.Tensor, np.ndarray)), type(backend)
    if isinstance(backend, torch.Tensor):
        Backend.FRAMEWORK = TORCH
        Backend.TORCH_DEVICE = backend.device
        # Backend.TORCH_DTYPE = backend.dtype
    elif isinstance(backend, np.ndarray):
        Backend.FRAMEWORK = NUMPY


def set_seed(seed):
    if is_torch():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    elif is_numpy():
        np.random.seed(seed)


def is_torch():
    return Backend.FRAMEWORK == TORCH


def is_numpy():
    return Backend.FRAMEWORK == NUMPY


def same_arg_and_behaviour(need_device=False):
    def _wrapper(function):
        def inner(*args, **kwargs):
            # print(function.__name__)
            if is_torch():
                if need_device:
                    kwargs["device"] = Backend.TORCH_DEVICE
                return getattr(torch, function.__name__)(*args, **kwargs)
            elif is_numpy():
                return getattr(np, function.__name__)(*args, **kwargs)

        return inner

    return _wrapper


def set_printoptions(precision=None, threshold=None, sci_mode=None):
    if is_torch():
        torch.set_printoptions(precision=precision, threshold=threshold, sci_mode=sci_mode)
    elif is_numpy():
        np.set_printoptions(precision=precision, threshold=threshold, suppress=not sci_mode)


#############################################
# Functions for creating array/tensor
#############################################


def tensor(data: BASIC_DATA_LIST):
    """
        Only receive basic number type!
    """
    if is_torch():
        return torch.tensor(data)
    elif is_numpy():
        return np.array(data)


@same_arg_and_behaviour(need_device=True)
def zeros(shape: TENSOR_SHAPE):
    pass


@same_arg_and_behaviour(need_device=True)
def ones(shape: TENSOR_SHAPE):
    pass


@same_arg_and_behaviour(need_device=True)
def arange(*args, **kwargs):
    pass


def rand(shape):
    """
        Return: Uniform(0, 1)
    """
    if is_torch():
        return torch.rand(shape, device=Backend.TORCH_DEVICE)
    elif is_numpy():
        return np.random.random(shape)


def normal(shape: TENSOR_SHAPE):
    """
        Return: Normal(0, 1)
    """
    if is_torch():
        return torch.normal(mean=0.0, std=1.0, size=shape, device=Backend.TORCH_DEVICE)
    elif is_numpy():
        return np.random.normal(loc=0.0, scale=1.0, size=shape)


def sparse_coo_tensor(*, indices: TENSOR, values: TENSOR, shape: TENSOR_SHAPE):
    if is_torch():
        return torch.sparse_coo_tensor(indices=indices, values=values, size=shape, device=Backend.TORCH_DEVICE)
    elif is_numpy():
        return sparse.coo_matrix((values, indices), shape=shape)


#############################################
# Functions for conversion
#############################################
def convert(data: TENSOR):
    if is_torch() and isinstance(data, torch.Tensor):
        return data
    elif is_numpy() and isinstance(data, np.ndarray):
        return data
    elif is_torch() and isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif is_numpy() and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


#############################################
# Functions for operating an array/tensor
#############################################
@same_arg_and_behaviour()
def abs(tensor: TENSOR):
    pass


@same_arg_and_behaviour()
def sum(tensor: TENSOR):
    pass


# def argmax(tensor: DATA, axis: Optional[int] = None, keepdim: bool = False):
#     if is_torch():
#         return torch.argmax(tensor, axis=axis, keepdim=keepdim)
#     elif is_numpy():
#         return np.argmax(tensor, axis=axis, keepdims=keepdim)
def argmax(tensor: TENSOR, axis: Optional[int] = None):
    """
        [not sure about the precise version]
        numpy: keepdims not supported before 1.19.2
        torch: axis not supported before 1.10.0
    """
    if is_torch():
        return torch.argmax(tensor, dim=axis)
    elif is_numpy():
        return np.argmax(tensor, axis=axis)


#############################################
# Functions for operating arraies/tensors
#############################################
def cat(tensors: TENSOR_LIST, axis: int):
    if is_torch():
        return torch.cat(tensors, axis=axis)
    elif is_numpy():
        return np.concatenate(tensors, axis=axis)


@same_arg_and_behaviour()
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
        if is_torch():
            ret = torch.einsum(conversion, *operands)
        else:
            ret = np.einsum(conversion, *operands)
    except Exception as e:
        print(f"Error occurs! In this function, '{equation}' is converted to '{conversion}'")
        raise e
    return ret
