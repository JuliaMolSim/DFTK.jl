# Timings and parallelization

This section summarizes the options DFTK offers
to monitor and influence performance of the code.

```@setup parallelization
using DFTK
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [2, 2, 2]
Ecut = 5
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

DFTK.reset_timer!(DFTK.timer)
scfres = self_consistent_field(basis, tol=1e-8)
```

## Timing measurements

By default DFTK uses [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)
to record timings, memory allocations and the number of calls
for selected routines inside the code. These numbers are accessible
in the object `DFTK.timer`. Since the timings are automatically accumulated
inside this datastructure, any timing measurement should first reset
this timer before running the calculation of interest.

For example to measure the timing of an SCF:
```@example parallelization
DFTK.reset_timer!(DFTK.timer)
scfres = self_consistent_field(basis, tol=1e-8)

DFTK.timer
```
The output produced when printing or displaying the `DFTK.timer`
now shows a nice table summarising total time and allocations as well
as a breakdown over individual routines.


!!! note "Timing measurements and stack traces"
    Timing measurements have the unfortunate disadvantage that they
    alter the way stack traces look making it sometimes harder to find
    errors when debugging.
    For this reason timing measurements can be disabled completely
    (i.e. not even compiled into the code) by setting the environment variable
    `DFTK_TIMING` to `"0"` or `"false"`.
    For this to take effect recompiling all DFTK (including the precompile cache)
    is needed.


!!! note "Timing measurements and threading"
    Unfortunately measuring timings in `TimerOutputs` is not yet thread-safe.
    Therefore taking timings of threaded parts of the code will be disabled
    unless you set `DFTK_TIMING` to `"all"`. In this case you must not use
    Julia threading (see section below) or otherwise undefined behaviour results.

## Threading
At the moment DFTK employs shared-memory parallelism
using multiple levels of threading
which distribute the workload
over different ``k``-Points, bands or within
an FFT or BLAS call between processors.
At its current stage our approach to threading is quite
simple and pragmatic,
such that we do not yet achieve a great scaling.
This should be improved in future versions of the code.

Finding a good sweet spot between the number of threads to use
and the extra performance gained by each additional working core
is not always easy,
since starting, terminating and synchronising threads takes time as well.
Most importantly the best settings wrt. threading
depend on both hardware and the problem
(e.g. number of bands, ``k``-Points, FFT grid size).

For the moment DFTK does not offer an automated selection mechanism
of thread and parallelization threading and just uses the Julia defaults.
Since these are rarely good,
users are advised to use the timing capabilities of DFTK
to experiment with threading
for their particular use case before running larger calculations.

### FFTW threads
For typical small to medium-size calculations in DFTK the largest part of time is spent
doing discrete Fourier transforms (about 80 to 90%).
For this reason parallelising FFTs can have a large effect on the runtime
of larger calculation in DFTK.
Unfortunately for scaling of FFT threading for smaller problem sizes
and large numbers of threads is not great,
such that by default threading in FFTW is even disabled.

The **recommended setting** for FFT threading with DFTK
is therefore to only use moderate number of FFT threads,
something like ``2`` or ``4`` and for smaller calculations
disable FFT threading completely.
To **enable parallelization of FFTs** (which is by default disabled),
use
```
using FFTW
FFTW.set_num_threads(N)
```
where `N` is the number of threads you desire.


### BLAS threads
All BLAS calls in Julia go through a parallelized OpenBlas
or MKL (with [MKL.jl](https://github.com/JuliaComputing/MKL.jl).
Generally threading in BLAS calls is far from optimal and
the default settings can be pretty bad.
For example for CPUs with hyper threading enabled,
the default number of threads seems to equal the number of *virtual* cores.
Still, BLAS calls typically take second place
in terms of the share of runtime they make up (between 10% and 20%).
Of note many of these do not take place on matrices of the size
of the full FFT grid, but rather only in a subspace
(e.g. orthogonalization, Rayleigh-Ritz, ...)
such that parallelization is either anyway disabled by the BLAS library
or not very effective.

The **recommendation** is therefore to use the same number of threads
as for the FFT threads.
You can set the number of BLAS threads by
```
using LinearAlgebra
BLAS.set_num_threads(N)
```
where `N` is the number of threads you desire.
To **check the number of BLAS threads** currently used, you can use
```
Int(ccall((BLAS.@blasfunc(openblas_get_num_threads), BLAS.libblas), Cint, ()))
```
or (from Julia 1.6) simply `BLAS.get_num_threads()`.

### Julia threads
On top of FFT and BLAS threading DFTK uses Julia threads (`Thread.@threads`)
in a couple of places to parallelize over `k`-Points (density computation)
or bands (Hamiltonian application).
The number of threads used for these aspects is controlled
by the *environment variable* `JULIA_NUM_THREADS`.
To influence the number of Julia threads used, set this variable *before*
starting the Julia process.

Notice, that Julia threading is applied on top of FFTW and BLAS threading
in the sense that the regions parallelized by Julia threads
again use parallelized FFT and BLAS calls,
such that the effects are not orthogonal.
Compared to FFT and BLAS threading the parallelization implied by using Julia
threads tends to scale better,
but its effectiveness is limited by the number of bands and
the number of irreducible `k`-Points used in the calculation.
Therefore this good scaling quickly diminishes for small to medium systems.

The **recommended setting** is to stick to `2` Julia threads
and to use `4` or more Julia threads only for
large systems and/or many `k`-Points.
To **check the number of Julia threads** use `Threads.nthreads()`.

### Summary of recommended settings

| Calculation size | `JULIA_NUM_THREADS` | `FFTW.set_num_threads(N)` | `BLAS.set_num_threads(N)` |
| ---------------- | ----- | ----- | ----- |
| tiny             |    1  |    1  |     1 |
| small            |    2  |    1  |     1 |
| medium           |    2  |    2  |     2 |
| large            |    2  |    4  |     4 |

Note, that this picture is likely to change in future versions
of DFTK and Julia as improvements to threading and parallelization
are made in the language or the code.

## MPI
DFTK uses MPI to distribute on kpoints only at the moment. This should be the
most performant method of parallelization: if you have kpoints, start by disabling
all threading and use this. Simply follow the instructions on
[MPI.jl](https://github.com/JuliaParallel/MPI.jl) and run DFTK under MPI:
```
mpiexecjl -np 16 julia myscript.jl
```
Notice that we use mpiexecjl (see MPI.jl docs) to automatically select the `mpiexec`
compatible with the MPI version used by Julia.

Issues and workarounds:
- Printing is garbled, as usual with MPI. You can use
  ```julia
  DFTK.mpi_master() || (redirect_stdout(); redirect_stderr())
  ```
  at the top of your script to disable printing on all processes but one.
- This feature is still experimental and some routines (e.g. band
  structure and direct minimization) are not compatible with this yet.
