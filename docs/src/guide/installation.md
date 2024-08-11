# Installation

In case you don't have a working Julia installation yet, first
[download the Julia binaries](https://julialang.org/downloads/)
and follow the [Julia installation instructions](https://julialang.org/downloads/platform/).
At least **Julia 1.10** is required for DFTK.

Afterwards you can install DFTK
[like any other package](https://julialang.github.io/Pkg.jl/v1/getting-started/)
in Julia. For example run in your Julia REPL terminal:
```julia
import Pkg
Pkg.add("DFTK")
```
which will install the latest DFTK release.
DFTK is continuously tested on Debian, Ubuntu, mac OS and Windows and should work on
these operating systems out of the box.
With this you are all set to run the code in the [Tutorial](@ref) or the
[`examples` directory](https://dftk.org/tree/master/examples).

For obtaining a good user experience as well as peak performance
some optional steps see the next sections on *Recommended packages*
and *Selecting the employed linear algebra and FFT backend*.
See also the details on [Using DFTK on compute clusters](@ref)
if you are not installing DFTK on a laptop or workstation.

!!! tip "DFTK version compatibility"
    We follow the usual [semantic versioning](https://semver.org/) conventions of Julia.
    Therefore all DFTK versions with the same minor (e.g. all `0.6.x`) should be
    API compatible, while different minors (e.g. `0.7.y`) might have breaking changes.
    These will also be announced in the [release notes](https://github.com/JuliaMolSim/DFTK.jl/releases).

## Optional: Recommended packages
While not strictly required to use DFTK it is usually convenient for computational
workflows (setting up initial structures, transferring data to ML training, etc.)
to install a few standard packages from the [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl).
Examples are
- [AtomsIO](https://github.com/mfherbst/AtomsIO.jl) and
  [AtomsIOPython](https://github.com/mfherbst/AtomsIO.jl),
  which allow you to read (and write) a large range of standard file formats
  for atomistic structures. In particular AtomsIO is lightweight and highly recommended.
- [AtomsBuilder](https://github.com/JuliaMolSim/AtomsBuilder.jl),
  which provides a few convenience functions for setting up initial structures,
  see [AtomsBase integration](@ref) for more details.

You can install these packages using
```julia
import Pkg
Pkg.add(["AtomsIO", "AtomsIOPython", "AtomsBuilder"])
```

!!! tip "Python dependencies in Julia"
    There are two main packages to use Python dependencies from Julia,
    namely [PythonCall](https://cjdoris.github.io/PythonCall.jl)
    and [PyCall](https://github.com/JuliaPy/PyCall.jl).
    These packages can be used side by side,
    but [some care is needed](https://cjdoris.github.io/PythonCall.jl/stable/pycall/).
    By installing AtomsIOPython and ASEconvert you indirectly install PythonCall
    which these two packages use to manage their third-party Python dependencies.
    This might cause complications if you plan on using PyCall-based packages
    (such as [PyPlot](https://github.com/JuliaPy/PyPlot.jl))
    In contrast AtomsIO is free of any Python dependencies and can be safely
    installed in any case.

## Optional: Selecting the employed linear algebra and FFT backend
The default Julia setup uses the BLAS, LAPACK, MPI and FFT libraries shipped as part
of the Julia package ecosystem.
The default setup works, but to obtain peak performance
for your hardware additional steps may be necessary, e.g. to employ the vendor-specific
BLAS or FFT libraries. See the documentation of the
[`MKL`](https://github.com/JuliaLinearAlgebra/MKL.jl),
[`FFTW`](https://juliamath.github.io/FFTW.jl/stable/),
[`MPI`](https://juliaparallel.org/MPI.jl/stable/configuration/#configure_system_binary)
and
[`libblastrampoline`](https://github.com/JuliaLinearAlgebra/libblastrampoline)
packages for details on switching the underlying backend library.
If you want to obtain a summary of the backend libraries currently employed
by DFTK run the `DFTK.versioninfo()` command.
See also [Using DFTK on compute clusters](@ref), where some of this is explained
in more details.

## Installation for DFTK development
If you want to contribute to DFTK, see the [Developer setup](@ref)
for some additional recommendations on how to setup Julia and DFTK.
