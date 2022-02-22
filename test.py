import torch
import numpy as np
import genops

for backend in (genops.NUMPY, genops.TORCH):
    print("====================")
    genops.set_backend(backend)
    genops.set_printoptions(precision=3)
    genops.set_seed(1926)
    print(f"use backend {genops.Backend.FRAMEWORK}")
    print("* zeros: \n", genops.zeros([1, 2]))
    print("* ones: \n", genops.ones([2, 2]))
    print("* uniform: \n", genops.rand([1, 3]))
    print("* normal: \n", genops.normal([1, 3]))
    print("* arange: \n", genops.arange(1, 5, 3))
    print("* argmax: \n", genops.argmax(genops.rand([2, 4]), axis=1))
    print("* cat: \n", genops.cat([genops.rand([2, 4]), genops.rand([2, 2])], axis=1))
    print("* stack: \n", genops.cat([genops.rand([2, 4]), genops.rand([2, 4])], axis=0))


genops.set_backend(genops.NUMPY)
a = genops.rand([2, 2, 1, 3])
b = genops.rand([3, 3])
c = genops.rand([2, 2, 3, 3])
genops.set_backend(genops.NUMPY)
print("* numpy einsum:")
print(genops.einsum("B T K1 D1, D1 D2, B T K2 D2->B T K1 K2", a, b, c))
genops.set_backend(genops.TORCH)
print("* torch einsum:")
a, b, c = genops.convert(a), genops.convert(b), genops.convert(c)
print(genops.einsum("B T K1 D1, D1 D2, B T K2 D2->B T K1 K2", a, b, c))


if torch.cuda.is_available():
    genops.set_backend(torch.tensor(0.0).cuda())
    a = genops.rand([20, 100, 32, 5])
    b = genops.rand([5, 5])
    c = genops.rand([20, 100, 64, 5])
    print(genops.einsum("B T K1 D1, D1 D2, B T K2 D2->B T K1 K2", a, b, c).device)
