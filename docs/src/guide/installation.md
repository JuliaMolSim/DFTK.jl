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

## Recommended optional python dependencies
While not strictly speaking required to use DFTK, a large number of convenience
features (such as reading standard [Input and output formats](@ref))
are based on ASE, the [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html).
The installation of the ASE python package is therefore strongly recommended.
To do so first find out which version of python is used by the
[PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package.
You can do this for example with the Julia commands
```julia
using PyCall
PyCall.python
```
Then use the corresponding package manager (usually `apt`, `pip`, `pip3` or `conda`)
to install ASE, for example
```
pip install ase
```
or
```
conda install -c conda-forge ase
```
See also the [installation instructions](https://wiki.fysik.dtu.dk/ase/install.html)
in the ASE documentation.

## Developer setup
If you want to start developing DFTK Julia has the option to
automatically keep track of the changes of the sources during development.
This means, for example, that
[`Revise`](https://github.com/timholy/Revise.jl) will automatically be aware
of the changes you make to the DFTK sources and automatically
reload your changes inside an active Julia session.
To achieve such a setup you have two recommended options:

1. Add a development version of DFTK to the global Julia environment:
   ```julia
   import Pkg
   Pkg.develop("DFTK")
   ```
   This clones DFTK to the path `~/.julia/dev/DFTK"` (on Linux).
   Note that with this method you cannot install both the stable
   and the development version of DFTK into your global environment.

2. Clone [DFTK](https://dftk.org) into a location of your choice
   ```bash
   $ git clone https://github.com/JuliaMolSim/DFTK.jl /some/path/
   ```
   Whenever you want to use exactly this development version of DFTK
   in a [Julia environment](https://julialang.github.io/Pkg.jl/v1/environments/)
   (e.g. the global one) add it as a `develop` package:
   ```julia
   import Pkg
   Pkg.develop("/some/path/")
   ```
   To run a script or start a Julia REPL using exactly this source tree
   as the DFTK version, use the `--project` flag of Julia,
   see [this documentation](https://julialang.github.io/Pkg.jl/v1/environments/)
   for details. For example to start a Julia REPL with this version of DFTK use
   ```bash
   $ julia --project=/some/path/
   ```
   The advantage of this method is that you can easily have multiple
   clones of DFTK with potentially different modifications made.
