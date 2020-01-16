<img src="https://raw.githubusercontent.com/JuliaMolSim/DFTK.jl/master/docs/logo/DFTK_750x250.png" alt="dftk logo" height="100px" />

# Density-functional toolkit

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamolsim.github.io/DFTK.jl/dev)
[![license](https://img.shields.io/github/license/JuliaMolSim/DFTK.jl.svg?maxAge=2592000)](https://github.com/JuliaMolSim/DFTK.jl/blob/master/LICENSE)
[![Build Status on Linux](https://travis-ci.org/JuliaMolSim/DFTK.jl.svg?branch=master)](https://travis-ci.org/JuliaMolSim/DFTK.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaMolSim/DFTK.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaMolSim/DFTK.jl?branch=master)
[![DOI](https://zenodo.org/badge/181734238.svg)](https://zenodo.org/badge/latestdoi/181734238)

The density-functional toolkit, or short **DFTK** is a library of
Julia routines for experimentation with plane-wave-based
density-functional theory (DFT), as implemented in much larger
production codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/). The main
aim at the moment is to provide a platform to facilitate numerical
analysis of algorithms and techniques related to DFT. For this we want
to leverage as much of the existing developments in plane-wave DFT and
the related ecosystems of Julia python or C codes as possible.

The library is at an early stage and the supported feature set
is thus still limited. Current features include:
- Lattice construction and problem setup based on [pymatgen](https://pymatgen.org/)
- Plane-wave discretisations building on top of
  [FFTW.jl](https://github.com/JuliaMath/FFTW.jl).
- All LDA and GGA functionals from [Libxc.jl](https://github.com/unkcpz/Libxc.jl).
- Insulators and metals (Fermi-Dirac or Methfessel-Paxton smearing)
- GTH or HGH pseudopotentials
- Exploitation of Brillouin zone symmetry for k-Point sampling
- Multiple self-consistent field approaches (Kerker mixing, Anderson mixing (DIIS),
  [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl), damping)
- Direct minimization
- Band structure generation
- Computation of density of states (DOS) and local density of states (LDOS)
- Full access to intermediate quantities (density, Bloch wave)
- Support for arbitrary floating point types, including `Float32` (single precision)
  or `Double64` (from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl).
  (Only for selected DFT functionals at the moment).

**Note:** This code has only been compared against standard packages
for a small number of test cases and might still contain bugs.
Use for production calculations is not yet recommended.

## Installation
The package is not yet registered in the [General](https://github.com/JuliaRegistries/General)
registry of Julia. Instead you can obtain it from
the [MolSim](https://github.com/JuliaMolSim/MolSim.git) registry,
which contains a bunch of packages related to performing molecular simulations in Julia.
Note that at least **Julia 1.2** is required.

First add `MolSim` to your installed registries. For this use
```
] registry add https://github.com/JuliaMolSim/MolSim.git
```
for a Julia command line (version at least 1.1).
Afterwards you can install DFTK like any other package in Julia:
```
] add DFTK
```

Some parts of the code require a working Python installation with the libraries
`scipy`, `pymatgen` and `spglib`. The examples require `matplotlib` as well.
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
project rather general, so that this platform is useful for
general research in materials science.

## Citation
[![DOI](https://zenodo.org/badge/181734238.svg)](https://zenodo.org/badge/latestdoi/181734238)

## Contact
Feel free to contact us (@mfherbst and @antoine-levitt) directly,
open issues or submit pull requests. Any contribution or discussion is welcome!
