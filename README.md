<img src="https://raw.githubusercontent.com/JuliaMolSim/DFTK.jl/master/docs/logo/DFTK_750x250.png" alt="dftk logo" height="100px" />

# Density-functional toolkit

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamolsim.github.io/DFTK.jl/dev)
[![gitter](https://badges.gitter.im/DFTK-jl/community.svg)](https://gitter.im/DFTK-jl/community)
[![license](https://img.shields.io/github/license/JuliaMolSim/DFTK.jl.svg?maxAge=2592000)](https://github.com/JuliaMolSim/DFTK.jl/blob/master/LICENSE)
[![Build Status on Linux](https://travis-ci.org/JuliaMolSim/DFTK.jl.svg?branch=master)](https://travis-ci.org/JuliaMolSim/DFTK.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaMolSim/DFTK.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaMolSim/DFTK.jl?branch=master)
[![DOI](https://zenodo.org/badge/181734238.svg)](https://zenodo.org/badge/latestdoi/181734238)


The density-functional toolkit, **DFTK** for short, is a library of
Julia routines for experimentation with plane-wave-based
density-functional theory (DFT), as implemented in much larger
production codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/). The main
aim at the moment is to provide a platform to facilitate numerical
analysis of algorithms and techniques related to DFT. For this we want
to leverage as much of the existing developments in plane-wave DFT and
the related ecosystems of Julia python or C codes as possible.

## Features
The library is at an early stage but we do already support a sizeable
set of features. An overview:

* Methods and models:
	- Kohn-Sham-like models, with an emphasis on flexibility: compose your own model,
	  from Cohen-Bergstresser linear eigenvalue equations to Gross-Pitaevskii equations
	  and sophisticated LDA/GGA functionals (any functional from the
	  [libxc](https://tddft.org/programs/libxc/) library)
	- Analytic potentials or Godecker norm-conserving pseudopotentials (GTH, HGH)
	- Brillouin zone symmetry for k-Point sampling using [spglib](https://atztogo.github.io/spglib/)
	- Smearing functions for metals
	- Self-consistent field approaches: Damping, Kerker mixing, Anderson/Pulay/DIIS mixing,
	  interface to [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)
	- Direct minimization using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
	- Multi-level threading (kpoints, eigenvectors, FFTs, linear algebra)
    - 1D / 2D / 3D systems
    - Magnetic fields
* Ground-state properties and post-processing:
	- Total energy
	- Forces
	- Density of states (DOS), local density of states (LDOS)
	- Band structures
	- Easy access to all intermediate quantities (e.g. density, Bloch waves)
* Support for arbitrary floating point types, including `Float32` (single precision)
  or `Double64` (from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)).
  For DFT this is currently restricted to LDA (with Slater exchange and VWN correlation).

All this in about 5k lines of pure Julia code. The code emphasizes simplicity and
flexibility, with the goal of facilitating methodological development and
interdisciplinary collaboration.
It has not been properly optimized and fine-tuned yet,
but the performance is of the same order of magnitude as established packages.

**Note:** DFTK has only been compared against standard packages
for a small number of test cases and might still contain bugs.

## Getting started
The package is not yet registered in the [General](https://github.com/JuliaRegistries/General)
registry of Julia. Instead you can obtain it from
the [MolSim](https://github.com/JuliaMolSim/MolSim.git) registry,
which contains a bunch of packages related to performing molecular simulations in Julia.
Note that at least **Julia 1.3** is required.

First add `MolSim` to your installed registries. For this use
```
] registry add https://github.com/JuliaMolSim/MolSim.git
```
for a Julia command line.
Afterwards you can install DFTK like any other package in Julia:
```
] add DFTK
```
or if you like the bleeding edge:
```
] add DFTK#master
```

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
You can then run the code in the `examples/` directory.

## Citation
[![DOI](https://zenodo.org/badge/181734238.svg)](https://zenodo.org/badge/latestdoi/181734238)

## Contact
Feel free to contact us (@mfherbst and @antoine-levitt) directly,
open issues or submit pull requests. Any contribution or discussion is welcome!
