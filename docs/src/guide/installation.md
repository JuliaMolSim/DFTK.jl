# Installation

DFTK is not yet registered in the [General](https://github.com/JuliaRegistries/General)
registry of Julia.
Instead you can obtain it from
the [MolSim](https://github.com/JuliaMolSim/MolSim.git) registry,
which contains a bunch of packages related to performing molecular simulations in Julia.
At least **Julia 1.3** is required.

First add `MolSim` to your installed registries. For this use
```
] registry add https://github.com/JuliaMolSim/MolSim.git
```
from a Julia command line.
Afterwards you can install DFTK like any other package in Julia:
```
] add DFTK
```
or if you like to be up to date:
```
] add DFTK#master
```

## Python dependencies
Some parts of the code require a working Python installation with the libraries
[`pymatgen`](https://pymatgen.org/) and [`spglib`](https://atztogo.github.io/spglib/).
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
pip install spglib pymatgen
```
or
```
conda install -c conda-forge spglib pymatgen
```
Afterwards you're all set and should be able to
run the code in the can be found in the [`examples` directory](https://dftk.org/tree/master/examples).
