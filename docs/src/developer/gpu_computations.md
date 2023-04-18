# GPU computations

Performing GPU computations in DFTK is still work in progress. The goal is to build
on Julia's multiple dispatch to have the same code base for CPU and GPU. Our current
approach is to aim at decent performance without writing any custom kernels at all,
relying only on the high level functionalities implemented in the GPU packages.

To go even further with this idea of unified code, we would also like to be able to
support any type of GPU architecture: we do not want to hard-code the use of a
specific architecture, say a NVIDIA CUDA GPU. DFTK does not realy on an
architecture-specific package ([CUDA](https://github.com/JuliaGPU/CUDA.jl),
[ROCm](https://github.com/JuliaGPU/AMDGPU.jl),
[OneAPI](https://github.com/JuliaGPU/oneAPI.jl)...) but rather uses
[GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl), which is the counterpart of
`AbstractArray` but for GPU arrays.

## Current implementation

For now, GPU computations are done by specializing the `architecture` keyword argument
when creating the basis. `architecture` should be an initialized instance of
the (non-exported) `CPU` and `GPU` structures. `CPU` does not require any argument,
but `GPU` requires the type of array which will be used for GPU computations.

```julia
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.CPU())
PlaneWaveBasis(model; Ecut, kgrid, architecture = DFTK.GPU(CuArray))
```
!!! note "GPU API is experimental"
    It is very likely that this API will change, based on the evolution of the
    Julia ecosystem concerning distributed architectures.

Not all terms can be used when doing GPU computations. As of January 2023 this
concerns `Anyonic`, and `TermPairwisePotential`. Similarly GPU features are
not yet exhaustively tested, and it is likely that some aspects of the code such as
automatic differentiation or stresses will not work.

## Pitfalls
There are a few things to keep in mind when doing GPU programming in DFTK.
- Transfers to and from a device can be done simply by converting an array to
an other type. However, hard-coding the new array type (such as writing
`CuArray(A)` to move `A` to a CUDA GPU) is not cross-architecture, and can
be confusing for developers working only on the CPU code. These data transfers
should be done using the helper functions `to_device` and `to_cpu` which
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
*Note:* `similar` could also be used, but then a reference array
(one which already lives on the device) needs to be available at call time.
This was done previously, with helper functions to easily build new arrays
on a given architecture: see for example
[`zeros_like`](https://github.com/JuliaMolSim/DFTK.jl/pull/711/commits/ce5da66009440bd8552429eb8cfe96944da16564).
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
        model.lattice * Gi
    end
end
```
- List comprehensions should be avoided, as they always return a CPU `Array`.
Instead, we should use `map` which returns an array of the same type as the input
one.
- Sometimes, creating a new array or making a copy can be necessary to achieve good
performance. For example, iterating through the columns of a matrix to compute their
norms is not efficient, as a new kernel is launched for every column. Instead, it
is better to build the vector containing these norms, as it is a vectorized
operation and will be much faster on the GPU.

## Anti-patterns

- Avoid hard coding `Array` in fields of a struct, as it prevents the use of GPU. In fact, this results in the underlying data being automatically moved to the CPU, which might not be ideal.
- Avoid using `zeros`, `ones` as they simply the output would be an `Array` that is on CPU. Instead, use `zeros_like` constructor, or `similar`.

For example,

```julia
rho_temp = zeros(T, basis.fft_size...)
```

should be written as:

```julia
rho_temp = zeros_like(basis.G_vectors, T, basis.fft_size...)
```


## Debugging

First, make sure the scalar indexing is disabled
as otherwise fallback routines on the CPU are allowed. 
While  these are helpful for interactive debugging, these always imply a synchronisation on the device and a data transfer to main memory. 
In practice runs this then results in an unnecessary overhead and the calculation appearing to be struck.

Note that by default scalar indexing is allowed on REPL and results in a warning being printed. 
However, this does not tell you where the offending lines are. 

To completely disable scalar indexing, just run

```julia
CUDA.allowscalar(false)
```

which will cause the program to error when scalar indexing is used.
The offending code should be rewritten using array operations. 
If it is not possible or non-trivial to do so, it is usually best to have the calculation to run on the CPU:

```julia
a_cpu = to_cpu(a)

y_cpu = map(a_cpu) do x
...
end

y = to_device(y_cpu)
```

If the calculation fails with a `GPU` architecture, the stack trace can often provide useful information.
Pay attention to the function call that is in DFTK.jl in the trace, and also what type of its arguments are.
Typically, the error is caused by mixing arrays on GPU (e.g. `CuArray`) with arrays on CPU (e.g. `Array`).

