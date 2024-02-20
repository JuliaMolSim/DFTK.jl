# Using DFTK on compute clusters

This chapter summarises a few tips and tricks for running DFTK on compute clusters.
It assumes you have already installed Julia on the machine in question
(see [Julia downloads](https://julialang.org/downloads/)
and [Julia installation instructions](https://julialang.org/downloads/platform/)).
For a general presentation of using Julia on HPC systems from both the user
and the admin perspective,
see [JuliaOnHPCClusters](https://juliahpc.github.io/JuliaOnHPCClusters/).

In this presentation we use [EPFL's scitas clusters](https://scitas-doc.epfl.ch/)
as the example to explain the basic principles.

## Julia depot path
By default on Linux-based systems
Julia puts all installed packages, including the binary packages
into the path `$HOME/.julia`, which can easily take a few tens of GB.
On many compute clusters the `/home` partition is
on a shared filesystem (thus access is slower) and has a tight space quota.
Usually it is therefore more advantageous to put Julia packages
on a less persistent, *faster* filesystem.
On many systems (such as [EPFL scitas](https://scitas-doc.epfl.ch/user-guide/using-clusters/file-system/))
this is the `/scratch` partition.
In your `~/.bashrc` or otherwise you should thus redirect
the `JULIA_DEPOT_PATH` to be a subdirectory of `/scratch`.

**EPFL scitas.**
On scitas the right thing to do is to insert
```
export JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH:/scratch/$USER/.julia"
```
into your `~/.bashrc`.

## Installing DFTK into a local Julia environment
When employing a compute cluster, it is often desirable to integrate
with the matching cluster-specific libraries,
such as vendor-specific versions of BLAS, LAPACK etc.
This is easiest achieved by installing `DFTK`
not into the global Julia environment,
but instead bundle the installation and the cluster-specific configuration
in a local, cluster-specific environment.

On top of the discussion of the main [Installation](@ref) instructions,
this requires one additional step, namely the use of a custom Julia package environment.

**Setting up a package environment.**
In the Julia REPL, create a new environment (essentially a new Julia project)
using the following commands:
```
import Pkg
Pkg.activate("path/to/new/environment")
```
Replace `path/to/new/environment` with the directory where you wish to create the new environment.
This will generate a folder containing `Project.toml` and `Manifest.toml` files.
Both together provide a reproducible image of the packages you installed in your project.
Once the `activate` call has been issued,
you can than install packages as usual. E.g. use `Pkg.add("DFTK")`
to install DFTK. The difference is that instead of tracking this in your *global*
environment, the installations will be tracked
in the local `Project.toml` and `Manifest.toml`
files of the `path/to/new/environment` folder.

To start a Julia shell directly with such this environment activated,
run it as `julia --project=path/to/new/environment`.

**Updating an environment.**
Start Julia and activate the environment. Then run `Pkg.update()`, i.e.
```
import Pkg
Pkg.activate("path/to/new/environment")
Pkg.update()
```

For more information on Julia environments, see the respective documentation
on [Code loading](https://docs.julialang.org/en/v1/manual/code-loading/)
and on [Managing Environments](https://pkgdocs.julialang.org/v1/environments/).

## Setting up local preferences
On cluster machines often highly optimised versions of BLAS, LAPACK, FFTW, MPI
or other basic packages are provided by the cluster vendor or operator.
These can be used
with DFTK by configuring an appropriate `LocalPreferences.toml` file, which
tells Julia where the cluster-specific libraries are located. The following
sections explain how to generate such a `LocalPreferences.toml` specific
for your cluster. Once this file has been generated and sits next to a
`Project.toml` in a Julia environment,
it ensures that the cluster-specific libraries are used instead
of the default ones.
Therefore this  setup only needs to be done once per project.

A useful way to check whether the setup has been successful and
`DFTK` indeed employs the desired cluster-specific libraries
provides the
```julia
DFTK.versioninfo()
```
command. It produces an output such as
```
DFTK Version      0.6.16
Julia Version     1.10.0
FFTW.jl provider  fftw v3.3.10

BLAS.get_config()
  LinearAlgebra.BLAS.LBTConfig
  Libraries:
  └ [ILP64] libopenblas64_.so

MPI.versioninfo()
  MPIPreferences:
    binary:  MPICH_jll
    abi:     MPICH

  Package versions
    MPI.jl:             0.20.19
    MPIPreferences.jl:  0.1.10
    MPICH_jll:          4.1.2+1

  Library information:
    libmpi:  /home/mfh/.julia/artifacts/0ed4137b58af5c5e3797cb0c400e60ed7c308bae/lib/libmpi.so
    libmpi dlpath:  /home/mfh/.julia/artifacts/0ed4137b58af5c5e3797cb0c400e60ed7c308bae/lib/libmpi.so

[...]
```
which thus specifies in one overview the details of the employed
BLAS, LAPACK, FFTW, MPI, etc. libraries.

### Switching to MKL for BLAS and FFT
The [`MKL`](https://github.com/JuliaLinearAlgebra/MKL.jl)
Julia package provides the Intel MKL library as a BLAS backend
in Julia. To use fully use it you need to do two things:
1. Add a `using MKL` to your scripts.
2. Configure a `LocalPreferences.jl` to employ the MKL also for FFT operations:
   to do so run the following Julia script:
   ```julia
   using MKL
   using FFTW
   FFTW.set_provider!("mkl")
   ```

### Switching to the system-provided MPI library
To use a system-provided MPI library, load the required modules.
On scitas that is
```sh
module load gcc
module load openmpi
```
Afterwards follow the [MPI system binary instructions](https://juliaparallel.org/MPI.jl/stable/configuration/#configure_system_binary)
and execute
```julia
using MPIPreferences
MPIPreferences.use_system_binary()
```

## Using the `--heap-size-hint` flag
Julia uses dynamic memory management,
which means that unused memory may not be directly released back to the operating system.
Much rather a garbage collection takes care of doing such cleanup periodically.
If you are in a memory-bound situation,
it can thus be helpful to employ the `--heap-size-hint` flag,
which provides Julia with a hint of the maximal memory Julia may use.
For example the call
```
julia --heap-size-hint 40G
```
tells Julia to optimise the garbage collection,
such that the overall memory consumption remains below `40G`.
Note, however that this is just a *hint*,
i.e. no hard limits are enforced.
Furthermore using too small a heap size hint can have a negative impact on performance.

## Running slurm jobs
This example shows how to run a DFTK calculation on a slurm-based system
such as scitas. We use the MKL for FFTW and BLAS and the system-provided MPI.
This setup will create five files `LocalPreferences.toml`, `Project.toml`,
`dftk.jl`, `silicon.extxyz` and `job.sh`.

At the time of writing (Dec 2023)
following the setup indicated above leads to this `LocalPreferences.toml` file:
```toml
[FFTW]
provider = "mkl"

[MPIPreferences]
__clear__ = ["preloads_env_switch"]
_format = "1.0"
abi = "OpenMPI"
binary = "system"
cclibs = []
libmpi = "libmpi"
mpiexec = "mpiexec"
preloads = []
```
We place this into a folder next to a `Project.toml` to define our project:
```toml
[deps]
AtomsIO = "1692102d-eeb4-4df9-807b-c9517f998d44"
DFTK = "acf6eb54-70d9-11e9-0013-234b7a5f5337"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
MKL = "33e6dc65-8f57-5167-99aa-e5a354878fb2"
MPIPreferences = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
```
We additionally create a small file `dftk.jl` to run an MPI-parallelised
calculation from a passed structure
```julia
using MKL
using DFTK
using AtomsIO

disable_threading()  # Threading and MPI not compatible

function main(structure, pseudos; Ecut, kspacing)
    if mpi_master()
        println("DEPOT_PATH=$DEPOT_PATH")
        println(DFTK.versioninfo())
        println()
    end

    system = attach_psp(load_system(structure); pseudos...)
    model  = model_PBE(system; temperature=1e-3, smearing=Smearing.MarzariVanderbilt())

    kgrid = kgrid_from_minimal_spacing(model, kspacing)
    basis = PlaneWaveBasis(model; Ecut, kgrid)

    if mpi_master()
        println()
        show(stdout, MIME("text/plain"), basis)
        println()
        flush(stdout)
    end

    DFTK.reset_timer!(DFTK.timer)
    scfres = self_consistent_field(basis)
    println(DFTK.timer)

    if mpi_master()
        show(stdout, MIME("text/plain"), scfres.energies)
    end
end
```
and we dump the structure file `silicon.extxyz` with content
```
2
pbc=[T, T, T] Lattice="0.00000000 2.71467909 2.71467909 2.71467909 0.00000000 2.71467909 2.71467909 2.71467909 0.00000000" Properties=species:S:1:pos:R:3
Si         0.67866977       0.67866977       0.67866977
Si        -0.67866977      -0.67866977      -0.67866977
```
Finally the jobscript `job.sh` for a slurm job looks like this:
```sh
#!/bin/bash
# This is the block interpreted by slurm and used as a Job submission script.
# The actual julia payload is in dftk.jl
# In this case it sets the parameters for running with 2 MPI processes using a maximal
# wall time of 2 hours.

#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1

# Now comes the setup of the environment on the cluster node (the "jobscript")

# IMPORTANT: Set Julia's depot path to be a local scratch
# direction (not your home !).
export JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH:/scratch/$USER/.julia"

# Load modules to setup julia (this is specific to EPFL scitas systems)
module purge
module load gcc
module load openmpi
module load julia

# Run the actual payload of Julia using 1 thread. Use the name of this
# file as an include to make everything self-contained.
srun julia -t 1 --project -e '
    include("dftk.jl")
    pseudos = (; Si="hgh/pbe/si-q4.hgh" )
    main("silicon.extxyz",
         pseudos;
         Ecut=10,
         kspacing=0.3,
 )
'
```

## Using DFTK via the Aiida workflow engine
A preliminary integration of DFTK with the [Aiida](https://www.aiida.net/)
high-throughput workflow engine is available
via the [aiida-dftk](https://github.com/aiidaplugins/aiida-dftk) plugin.
This can be a useful alternative
if many similar DFTK calculations should be run.
