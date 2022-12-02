# GPU computations

GPU computations in DFTK is very much a work in progress. The goal is to build on
Julia's multiple dispatch to have the same code base for CPU and GPU. We hope we
will be able to reach decent performances without having to write any kernels at all,
only by relying on the high level functionalities implemented in the GPU packages.

To go even further with this idea of unified code, we would also like to be able to
support any type of GPU architecture: unlike other packages, we do not want to
hard-code the use of a specific architecture, say a NVIDIA CUDA GPU. In order to do
this, DFTK uses [GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl), which is the
counterpart of `AbstractArray` but for GPU arrays.

Currently the three main GPU architectures are [CUDA](https://github.com/JuliaGPU/CUDA.jl),
[ROCm](https://github.com/JuliaGPU/AMDGPU.jl) and [OneAPI](https://github.com/JuliaGPU/oneAPI.jl).

## Current implementation

For now, GPU computations are done by specializing the `architecture` keyword argument
when creating the basis. `architecture` should be an initialized instance of
the (non-exported) `CPU` and `GPU` structures. `CPU` does not require any argument,
but `GPU` requires the type of array which will be used for GPU computations.

```@example gpu_computations
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.CPU())
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.GPU(CuArray))
```
Those `CPU` and `GPU` structures we inspired by the structures of same name in
[Oceananigans](https://github.com/CliMA/Oceananigans.jl). **However, it is very**
**likely that this API will change.** We hope that a unified convention will emerge,
instead of having every package define their own way of dealing with computations on
multiple architectures.

Not all terms can be used when doing GPU computations: currently, the `Anyonic`,
`Magnetic`, `TermPairwisePotential` and `Xc` terms are not supported. Symmetries
should also be disabled. Nothing has yet been made to enable GPU support for
automatic differentiation.

## Pitfalls
There are a few things to keep in mind when doing GPU programming in DFTK.
- Transfers to and from a device are a bit tricky because they are done by
converting the array to an other type (ex: `CuArray(A)`, `Array(A)`...). This
means that the array type used for computations should be stored somewhere:
currently, this is done by the `CPU` and `GPU` structures and the helper functions
`to_device` and `to_cpu`. `similar` could also be used, but then a reference array
(one which already lives on the device) needs to be available at any time.
- Functions' arguments should always be isbits (immutable and contains no
references to other values). This means that the model should never be given
to a function which will get executed on the GPU or it will fail: instead, only the
relevant field should be passed.
- List comprehensions should be avoided, as they always return a CPU `Array`.
Instead, we should use `map` which returns an array of the same type as the input
one.
- Sometimes, creating a new array or making a copy can be necessary to achieve good
performance. For example, iterating through the columns of a matrix to compute their
norms is not efficient, as a new kernel is launched for every column. Instead, it
is better to build the vector containing these norms, as it is a vectorized
operation and will be much faster on the GPU.
