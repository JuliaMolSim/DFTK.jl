# GPU computations

As of December 2025, DFTK porting to GPU is stable. SCF and forces are fully supported
for standard Libxc functionals with norm-conserving pseudopotentials. Stress tensor
calculations and automatic differentiation on the GPU is work in progress. DFTK can be
run on NVIDIA and AMD devices.

Our approach to GPU computation relies on Julia's multiple dispatch mechanism, such that
the same code base can be used on both CPU and GPU, regardless of the vendor. GPU sepcific
code is only written when necessary, but always at a high level. There are no explicit
CUDA/HIP kernels in DFTK. This high level approach is made possible by the following
packages: [CUDA](https://github.com/JuliaGPU/CUDA.jl), [ROCm](https://github.com/JuliaGPU/AMDGPU.jl),
and [GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl).

GPU acceleration of SCF has been measured to reach ~250x on NVIDIA GH200, and ~50x on AMD MI250.

## Usage

GPU computations are done by specializing the `architecture` keyword argument
when creating the [`PlaneWaveBasis`](@ref). `architecture` should be an initialized instance of
the (non-exported) `CPU` and `GPU` structures. `CPU` does not require any argument,
but `GPU` requires the type of array which will be used for GPU computations.

```julia
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.CPU())
```
```julia
using CUDA
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.GPU(CuArray))
```
```julia
using AMDGPU
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.GPU(ROCArray))
```

## Pitfalls
There are a few things to keep in mind when doing GPU programming in DFTK.
- Transfers to and from a device can be done simply by converting an array to
another type. However, hard-coding the new array type (such as writing
`CuArray(A)` to move `A` to a CUDA GPU) is not cross-architecture, and can
be confusing for developers working only on the CPU code. These data transfers
should be done using the helper functions [`DFTK.to_device`](@ref) and [`DFTK.to_cpu`](@ref) which
provide a level of abstraction while also allowing multiple architectures
to be used.
```julia
cuda_gpu = DFTK.GPU(CuArray)
cpu_architecture = DFTK.CPU()
A = rand(10)  # A is on the CPU
B = DFTK.to_device(cuda_gpu, A)  # B is a copy of A on the CUDA GPU
B .+= 1.
C = DFTK.to_cpu(B)  # C is a copy of B on the CPU
D = DFTK.to_device(cpu_architecture, B)  # Equivalent to the previous line, but
                                         # should be avoided as it is less clear
```
*Notes:* `similar` can also be used, but then a reference array
(one which already lives on the device) needs to be available at call time. DFTK is mostly
GPU resident, and very few explicit data transfers are required in practice. Generally, this
is done when an array must be accessed in an element-wise fashion (forbidden on `GPUArrays`).
- Functions which will get executed on the GPU should always have arguments
which are `isbits` (immutable and contains no references to other values).
When using `map`, also make sure that every structure used is also `isbits`.
For example, the following map will fail, as `model` contains strings and
arrays which are not `isbits`.
```julia
function map_lattice(model::Model, Gs::AbstractArray{Vec3})
    # model is not isbits
    map(Gs) do Gi
        model.lattice * Gi
    end
end
```
However, the following map will run on a GPU, as the lattice is a static matrix.
```julia
function map_lattice(model::Model, Gs::AbstractArray{Vec3})
    lattice = model.lattice # lattice is isbits
    map(Gs) do Gi
        lattice * Gi
    end
end
```
- All performance-critical for loops should be replaced by map or map! calls, since
the former won't run on the GPU. In case of nested loops, the outermost one should be a map,
such that each GPU thread can execute the inner loop (ideally the smaller one).
- List comprehensions should be avoided, as they always return a CPU `Array`.
Instead, we should use `map` which returns an array of the same type as the input
one.
- Sometimes, creating a new array or making a copy can be necessary to achieve good
performance. For example, iterating through the columns of a matrix to compute their
norms is not efficient, as a new kernel is launched for every column. Instead, it
is better to build the vector containing these norms, as it is a vectorized
operation and will be much faster on the GPU.
- Array broadcasting operations are compiled to GPU kernels. When possible, write
multiple operations on the same line, such that a single fused kernel is created. 
Few large GPU kernels are always more efficient than many small ones.
- Allocation of large GPUArrays can be costly, especially when happening repeatedly
in a loop. When possible, pre-allocate such arrays and use in place operations (map!, mul!, etc.)
- Some operations are GPU-legal, but extremely slow. These can only be spotted with careful profiling.

## Known issues
- Some operations on specific GPUArrays types are not properly taken care of with either CUDA
or AMDGPU. When spotted, workarounds are implemented in the relevent DFTK extensions (`DFTKCUDAExt.jl`, `DFTKAMDGPUExt.jl`).
- Diagonalization of complex Hermitian matrices is very slow on AMDGPU. This becomes the main bottleneck of calculations where it should be negligible. Hopefully, this will be fixed with
ROCm v7.
- In principle, GPU-aware MPI works out of the box. However, depending on the MPI provider, a whole
host of environment variables might be necessary.
