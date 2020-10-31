# Installation

In case you have not yet installed Julia, first do so by
[downloading the Julia binaries](https://julialang.org/downloads/)
and following the [Julia installation instructions](https://julialang.org/downloads/platform/).
At least **Julia 1.4** is required for DFTK.

Afterwards you can install DFTK
[like any other package](https://julialang.github.io/Pkg.jl/v1/getting-started/)
in Julia. For example run in your Julia REPL terminal:
```julia
using Pkg
pkg"add DFTK"
```
which will install the latest DFTK release.
Alternatively (if you like to be fully up to date) install the master branch:
```
using Pkg
pkg"add DFTK#master"
```

## Python dependencies
Some parts of the code require a working Python installation with the
[`pymatgen`](https://pymatgen.org/) module.
Check out which version of python is used by the
[PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package.
You can do this for example with the Julia commands
```julia
using PyCall
PyCall.python
```
Then use the corresponding package manager (usually `apt`, `pip`, `pip3` or `conda`)
to install aforementioned libraries, for example
```
pip install pymatgen
```
or
```
conda install -c conda-forge pymatgen
```
Afterwards you're all set and should be able to
run the code in the [`examples` directory](https://dftk.org/tree/master/examples).
