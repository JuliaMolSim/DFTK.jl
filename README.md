# DFTK.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mfherbst.github.io/DFTK.jl/dev)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/mfherbst/DFTK.jl/blob/master/LICENSE)
[![Build Status on Linux](https://travis-ci.org/mfherbst/DFTK.jl.svg?branch=master)](https://travis-ci.org/mfherbst/DFTK.jl)
[![Coverage Status](https://coveralls.io/repos/mfherbst/DFTK.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mfherbst/DFTK.jl?branch=master)

DFTK, short for the **density-functional toolkit** is a library of
Julia routines for experimentation with plane-wave-based
density-functional theory (DFT), as implemented in much larger
production codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/). The main
aim at the moment is to provide a platform to facilitate numerical
analysis of algorithms and techniques related to DFT. For this we want
to leverage as much of the existing developments in plane-wave DFT and
the related ecosystems of Julia python or C codes as possible.

The library is at a very early stage of development and the supported feature set
is thus limited. Current features include:
- Lattice construction and problem setup based on [pymatgen](https://pymatgen.org/)
- Plane-wave discretisations building on top of
  [FFTW.jl](https://github.com/JuliaMath/FFTW.jl).
- SCF routine based on [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)
  and [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).
- LDA and GGA functionals from [Libxc.jl](https://github.com/unkcpz/Libxc.jl).
- Insulators and metals (Fermi-Dirac or Methfessel-Paxton smearing)
- Band structure generation
- Support for both `Float64` (double precision) and `Float32` (single precision)
  throughout the library (Only for selected DFT functionals at the moment).

**Note:** This code has only been compared against standard packages
for a small number of test cases and might still contain bugs.
Use for production calculations is not yet recommended.

## Installation
The package is not yet registered, so you need to install it from the github url: Use
```
] add https://github.com/mfherbst/DFTK.jl.
```
from a Julia command line (version at least 1.1).

Some parts of the code require a working Python installation with the libraries
`scipy` and `spglib`. The examples require `matplotlib` and `pymatgen` as well.
Check out which version of python is used by the
[PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package, and use the
corresponding package manager (usually `apt`, `pip` or `conda`) to install
these libraries, for example
```
pip install scipy spglib matplotlib pymatgen
```
You can then run the code in the `examples/` directory.


## Perspective
Despite the current focus on numerics, the intention is to keep the
project rather general, so that this platform could be useful for
other research in the future.

## Contact
Feel free to contact us (@mfherbst and @antoine-levitt) directly,
open issues or submit pull requests. Any contribution or discussion is welcome!
