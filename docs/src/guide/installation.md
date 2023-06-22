# Installation

In case you don't have a working Julia installation yet, first
[download the Julia binaries](https://julialang.org/downloads/)
and follow the [Julia installation instructions](https://julialang.org/downloads/platform/).
At least **Julia 1.6** is required for DFTK.

Afterwards you can install DFTK
[like any other package](https://julialang.github.io/Pkg.jl/v1/getting-started/)
in Julia. For example run in your Julia REPL terminal:
```julia
import Pkg
Pkg.add("DFTK")
```
which will install the latest DFTK release.
Alternatively (if you like to be fully up to date) install the master branch:
```julia
import Pkg
Pkg.add(name="DFTK", rev="master")
```

DFTK is continuously tested on Debian, Ubuntu, mac OS and Windows and should work on
these operating systems out of the box.

That's it. With this you are all set to
run the code in the [Tutorial](@ref) or the
[`examples` directory](https://dftk.org/tree/master/examples).

!!! tip "DFTK version compatibility"
    We follow the usual [semantic versioning](https://semver.org/) conventions of Julia.
    Therefore all DFTK versions with the same minor (e.g. all `0.6.x`) should be
    API compatible, while different minors (e.g. `0.7.y`) might have breaking changes.
    These will also be announced in the [release notes](https://github.com/JuliaMolSim/DFTK.jl/releases).

## Recommended optional packages
While not strictly speaking required to use DFTK it is usually convenient to install
a couple of standard packages from the [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl)
ecosystem to make working with DFT more convenient. Examples are

- [AtomsIO](https://github.com/mfherbst/AtomsIO.jl) and
  [AtomsIOPython](https://github.com/mfherbst/AtomsIO.jl),
  which allow you to read (and write) a large range of standard file formats
  for atomistic structures. In particular AtomsIO is lightweight and highly recommended.
- [ASEconvert](https://github.com/mfherbst/ASEconvert.jl),
  which integrates DFTK with a number of convenience features of the
  ASE, the [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html).
  See [Creating and modelling metallic supercells](@ref) for an example where
  ASE is used within a DFTK workflow.

You can install these packages using
```julia
import Pkg
Pkg.add(["AtomsIO", "AtomsIOPython", "ASEconvert"])
```

!!! tip "Python dependencies in Julia"
    There are two main packages to use Python dependencies from Julia,
    namely [PythonCall](https://cjdoris.github.io/PythonCall.jl)
    and [PyCall](https://github.com/JuliaPy/PyCall.jl).
    These packages can be used side by side,
    but [some care is needed](https://cjdoris.github.io/PythonCall.jl/stable/pycall/).
    By installing AtomsIOPython and ASEconvert you indirectly install PythonCall
    which these two packages use to manage their third-party Python dependencies.
    This might cause complications if you plan on  using PyCall-based packages
    (such as [PyPlot](https://github.com/JuliaPy/PyPlot.jl))
    In contrast AtomsIO is free of any Python dependencies and can be safely installed in any case.

## Developer setup
See [Developer setup](@ref) for recommendations if you
want to develop and contribute to DFTK
