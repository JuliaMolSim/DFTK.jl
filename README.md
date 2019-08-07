# DFTK.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mfherbst.github.io/DFTK.jl/dev)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/mfherbst/DFTK.jl/blob/master/LICENSE)
[![Build Status on Linux](https://travis-ci.org/mfherbst/DFTK.jl.svg?branch=master)](https://travis-ci.org/mfherbst/DFTK.jl)
[![Coverage Status](https://coveralls.io/repos/mfherbst/DFTK.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mfherbst/DFTK.jl?branch=master)

DFTK, short for the **density-functional toolkit** is a library of
Julia routines for experimentation with plane-wave-based
density-functional theory (DFT), as implemented in much larger
production codes such as Abinit, Quantum Espresso and VASP. The main
aim at the moment is to provide a platform to facilitate numerical
analysis of algorithms and techniques related to DFT. For this we want
to leverage as much of the existing developments in plane-wave DFT and
the related ecosystems of Julia python or C codes as possible.

The library is at a very early stage of development and the supported feature set
is thus limited. Current features include:
- Lattice construction and problem setup based on [pymatgen](https://pymatgen.org/)
- Plane-wave discretisations building on top of
  [FFTW](https://github.com/JuliaMath/FFTW.jl).
- SCF routine based on [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl)
  and [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).
- LDA and GGA functionals from [Libxc](https://github.com/unkcpz/Libxc.jl).
- Band structure generation
- Support for both `Float64` (double precision) and `Float32` (single precision)
  throughout the library (Only for selected DFT functionals at the moment).

**Note:** The code has not been properly verified against a standard DFT package
and might contain bugs. Use for production calculations is not yet recommended.

## Installation
The package is not yet registered, so you need to install it from the github url: use
```
] add https://github.com/mfherbst/DFTK.jl.
```
from a Julia command line (version at least 1.1). You can then run the code in the `examples/` directory.

Some parts of the code require a working Python installation with the libraries `scipy` and `spglib`. The examples require `matplotlib` and `pymatgen`. Check out which version of python is used by the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package, and use the correponding package manager (usually `apt`, `pip` or `conda`) to install these libraries.

## Perspective
Despite the current focus on numerics, the intention is to keep the
project rather general, so that this platform could be useful for
other research in the future.

## Contact
Feel free to contact us (Michael Herbst and Antoine Levitt) directly,
to open issues and to submit pull requests. Any contribution or
discussion is welcome!
