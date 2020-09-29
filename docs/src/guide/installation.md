# Installation

Similar to [installing any other registered package](https://julialang.github.io/Pkg.jl/v1/getting-started/)
in Julia just run from a Julia REPL:
```
] add DFTK
```
or if you like to be fully up to date:
```
] add DFTK#master
```
At least **Julia 1.4** is required.

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
