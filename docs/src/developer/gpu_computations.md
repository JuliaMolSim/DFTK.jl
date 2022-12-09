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

Not all terms can be used when doing GPU computations: currently, the `Anyonic`,
`Magnetic`, `TermPairwisePotential` and `Xc` terms are not supported. Some mixings,
such as `χ0Mixing` are also not yet supported. GPU features are not yet exhaustively
 tested, and it is likely that some aspects of the code such as automatic
 differentiation or stresses will not work.

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

## A quick benchmark
In this section, we will be running CPU and GPU computations on a medium-sized system to compare performance. We will be computing the SCF of a silicon supercell with parameters `(4,5,2)`, using Float32.

```julia
lattice   = [0.0  5.131570667152971 5.131570667152971;
            5.131570667152971 0.0 5.131570667152971;
            5.131570667152971 5.131570667152971  0.0]
Si        = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8];
supercell = ase_atoms(lattice, atoms, positions) * (4, 5, 2)
lattice   = Float32.(load_lattice(supercell))
positions = load_positions(supercell)
atoms     = fill(Si, length(positions))
model     = model_DFT(lattice, atoms, positions, [];
                      temperature=1e-3)
basis     = PlaneWaveBasis(model;
                          Ecut=30, kgrid=(1,1,1),
                          architecture=DFTK.GPU(CuArray)) # Or DFTK.CPU()
scfres    = self_consistent_field(basis; tol=1e-3,
                                  solver=scf_anderson_solver(),
                                  mixing = KerkerMixing());
```
Here is the timer produced by DFTK when doing GPU computations.
```
 Section                             ncalls     time    %tot     avg     alloc    %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────────────
 self_consistent_field                    1    23.0s   94.9%   23.0s    464MiB   75.2%   464MiB
   LOBPCG                                 6    21.2s   87.4%   3.53s    166MiB   26.9%  27.6MiB
     DftHamiltonian multiplication       40    5.84s   24.1%   146ms   77.4MiB   12.6%  1.93MiB
       local+kinetic                  4.87k    5.79s   23.9%  1.19ms   72.6MiB   11.8%  15.3KiB
         fft                          4.87k    5.04s   20.8%  1.04ms   29.0MiB    4.7%  6.10KiB
         ifft                         4.87k    577ms    2.4%   119μs   14.6MiB    2.4%  3.06KiB
       nonlocal                          40   42.1ms    0.2%  1.05ms    510KiB    0.1%  12.8KiB
     ortho! X vs Y                       62    4.59s   19.0%  74.1ms   47.2MiB    7.7%   780KiB
       drop!                            142    3.24s   13.4%  22.8ms    827KiB    0.1%  5.82KiB
       ortho!                           142    572ms    2.4%  4.03ms   29.6MiB    4.8%   213KiB
     rayleigh_ritz                       34    990ms    4.1%  29.1ms   1.14MiB    0.2%  34.4KiB
     preconditioning                     40    202ms    0.8%  5.06ms   30.5MiB    4.9%   781KiB
     ortho!                               6   59.1ms    0.2%  9.86ms   1.90MiB    0.3%   325KiB
   compute_density                        6    1.16s    4.8%   193ms   42.5MiB    6.9%  7.08MiB
     symmetrize_ρ                         6   10.8ms    0.0%  1.81ms   95.8KiB    0.0%  16.0KiB
       accumulate_over_symmetries!        6   80.3μs    0.0%  13.4μs   12.6KiB    0.0%  2.09KiB
   energy_hamiltonian                    13    211ms    0.9%  16.2ms   5.74MiB    0.9%   452KiB
     ene_ops                             13    208ms    0.9%  16.0ms   5.55MiB    0.9%   437KiB
       ene_ops: nonlocal                 13   90.8ms    0.4%  6.99ms    215KiB    0.0%  16.5KiB
       ene_ops: kinetic                  13   57.2ms    0.2%  4.40ms   4.48MiB    0.7%   353KiB
       ene_ops: hartree                  13   51.0ms    0.2%  3.93ms    400KiB    0.1%  30.8KiB
       ene_ops: local                    13   8.55ms    0.0%   658μs    170KiB    0.0%  13.1KiB
   KerkerMixing                           6    142ms    0.6%  23.6ms    113MiB   18.3%  18.8MiB
     enforce_real!                        6    132ms    0.5%  22.0ms    113MiB   18.3%  18.8MiB
   ortho_qr                               1   46.6ms    0.2%  46.6ms   10.7KiB    0.0%  10.7KiB
 guess_density                            1    1.23s    5.1%   1.23s    153MiB   24.8%   153MiB
 ──────────────────────────────────────────────────────────────────────────────────────────────
```

Here is the timer produced by DFTK when doing computations on a CPU with 8 threads. MKL was also used to increase performance.
```
 Section                             ncalls     time    %tot     avg     alloc    %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────────────
 self_consistent_field                    1    65.3s   95.8%   65.3s   24.3GiB   99.6%  24.3GiB
   LOBPCG                                 4    56.6s   83.0%   14.1s   18.5GiB   75.9%  4.63GiB
     DftHamiltonian multiplication       27    35.4s   51.9%   1.31s   2.61GiB   10.7%  99.1MiB
       local+kinetic                  3.67k    29.0s   44.9%  63.2ms    101MiB    0.4%  28.3KiB
         ifft                         3.67k    12.4s   19.0%  27.1ms   36.9MiB    0.1%  10.3KiB
         fft                          3.67k    11.9s   18.3%  25.9ms   38.9MiB    0.2%  10.9KiB
       nonlocal                          27    3.90s    5.7%   145ms   2.35GiB    9.6%  89.2MiB
     ortho! X vs Y                       42    7.48s   11.0%   178ms   2.68GiB   11.0%  65.3MiB
       ortho!                            97    2.45s    3.6%  25.2ms    101MiB    0.4%  1.04MiB
     rayleigh_ritz                       23    4.06s    5.9%   176ms    154MiB    0.6%  6.68MiB
     preconditioning                     27    726ms    1.1%  26.9ms   2.15MiB    0.0%  81.5KiB
     ortho!                               4    236ms    0.3%  58.9ms   5.78MiB    0.0%  1.44MiB
   compute_density                        4    5.38s    7.9%   1.34s   1.57GiB    6.4%   403MiB
     symmetrize_ρ                         4    246ms    0.4%  61.4ms    450MiB    1.8%   113MiB
       accumulate_over_symmetries!        4   12.5ms    0.0%  3.12ms     0.00B    0.0%    0.00B
   energy_hamiltonian                     9    1.96s    2.9%   217ms   2.49GiB   10.2%   283MiB
     ene_ops                              9    1.62s    2.4%   180ms   1.09GiB    4.5%   124MiB
       ene_ops: nonlocal                  9    755ms    1.1%  83.9ms   19.0MiB    0.1%  2.11MiB
       ene_ops: hartree                   9    589ms    0.9%  65.4ms    928MiB    3.7%   103MiB
       ene_ops: kinetic                   9    234ms    0.3%  26.0ms    309KiB    0.0%  34.3KiB
       ene_ops: local                     9   41.9ms    0.1%  4.65ms    169MiB    0.7%  18.8MiB
   KerkerMixing                           4    444ms    0.7%   111ms    750MiB    3.0%   188MiB
     enforce_real!                        4   69.5ms    0.1%  17.4ms   31.8KiB    0.0%  7.94KiB
   ortho_qr                               1    334ms    0.5%   334ms    377MiB    1.5%   377MiB
 guess_density                            1    2.89s    4.2%   2.89s    112MiB    0.4%   112MiB
 ──────────────────────────────────────────────────────────────────────────────────────────────
```
