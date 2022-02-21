# GenOps

`GenOps` provide **Genral tensor Operations** for `numpy` and `torch`.

`GenOps` is **torch-first**, just write your code following `torch`'s API, it can be easily adapted to `numpy`. `GenOps` **does not** re-invent new APIs to integrate other APIs.

```python
genops.set_backend(genops.TORCH)

# genops.set_backend(genops.NUMPY)
#    all generated tensor are np.ndarray!

# genops.set_backend(obj)
#    if obj is a torch.Tensor/np.ndarray, 
#       all generated tensor will have the same type.
#    even if obj is a torch.CudaTensor, 
#       all generated tensor will be on the same GPU!
genops.rand([1, 2])
```


`GenOps` provide a nice operator which supports better usage of `einsum` than `torch` and `numpy`.

```python
genops.einsum("BSZ SEQ1 DIM1, DIM1 DIM2, BSZ SEQ2 DIM2->BSZ SEQ1 SEQ2", a, b, c)
torch.einsum("bsd,de,bte->bst", a, b)
```



`GenOps` **does not** provide APIs which can be supported by `einops` (https://einops.rocks/), which is a framework-agnoistic library which make my skin smooth.
```python
import einops
# rearrange elements according to the pattern
output_tensor = einops.rearrange(input_tensor, 't b c -> b c t')
# combine rearrangement and reduction
output_tensor = einops.reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
```

