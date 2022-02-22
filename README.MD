# GenOps

`GenOps` provides **Genral tensor Operations** for `numpy` and `torch`.

## Switching Backend Framework

`GenOps` is **torch-first**, just write your code following `torch`'s API, it can be easily adapted to `numpy`. `GenOps` **does not** re-invent new APIs to integrate other APIs.

```python
genops.set_backend(genops.TORCH)
genops.rand([1, 2])   # tensor([[0.0294, 0.9194]])

genops.set_backend(genops.NUMPY)
genops.rand([1, 2])   # array([[0.85400985, 0.17538447]])

genops.set_backend_as(some_tensor_on_CUDA_0)
genops.rand([1, 2])   # tensor([[~]], device='cuda:0')  

genops.set_backend(genops.TORCH)
genops.convert(some_numpy_array)   # tensor(~)
```


## User Friendly `einsum`

`GenOps` provide a nice operator which supports better usage of `einsum` than `torch` and `numpy`.

```python
genops.einsum("BSZ SEQ1 DIM1, DIM1 DIM2, BSZ SEQ2 DIM2->BSZ SEQ1 SEQ2", a, b, c)
torch.einsum("bsd,de,bte->bst", a, b)
```

## More General Operations

`GenOps` **does not** provide APIs which can be supported by `einops` (https://einops.rocks/), a framework-agnoistic library which makes my skin smooth.
```python
import einops
# rearrange elements according to the pattern
output_tensor = einops.rearrange(input_tensor, 't b c -> b c t')
# combine rearrangement and reduction
output_tensor = einops.reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
```

