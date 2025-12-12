# GPU computations

Performing GPU computations in DFTK is still work in progress. The goal is to build
on Julia's multiple dispatch to have the same code base for CPU and GPU. Our current
approach is to aim at decent performance without writing any custom kernels at all,
relying only on the high level functionalities implemented in the GPU packages.

To go even further with this idea of unified code, we would also like to be able to
support any type of GPU architecture: we do not want to hard-code the use of a
specific architecture, say a NVIDIA CUDA GPU. DFTK does not rely on an
architecture-specific package ([CUDA](https://github.com/JuliaGPU/CUDA.jl),
[ROCm](https://github.com/JuliaGPU/AMDGPU.jl),
[OneAPI](https://github.com/JuliaGPU/oneAPI.jl)...) but rather uses
[GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl), which is the counterpart of
`AbstractArray` but for GPU arrays.

## Current implementation

For now, GPU computations are done by specializing the `architecture` keyword argument
when creating the [`PlaneWaveBasis`](@ref). `architecture` should be an initialized instance of
the (non-exported) [`CPU`](@ref) and [`GPU`](@ref) structures. [`CPU`](@ref) does not require any argument,
but [`GPU`](@ref) requires the type of array which will be used for GPU computations.

```julia
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.CPU())
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.GPU(CuArray))
```
!!! note "GPU API is experimental"
    It is very likely that this API will change, based on the evolution of the
    Julia ecosystem concerning distributed architectures.

Not all terms can be used when doing GPU computations. As of January 2023 this
concerns [`Anyonic`](@ref), [`Magnetic`](@ref) and [`TermPairwisePotential`](@ref). Similarly GPU features are
not yet exhaustively tested, and it is likely that some aspects of the code such as
automatic differentiation or stresses will not work.
