# DFTK.jl: The density-functional toolkit.

The density-functional toolkit, **DFTK** for short, is a library of
Julia routines for playing with plane-wave
density-functional theory (DFT) algorithms.
In its basic formulation it solves periodic Kohn-Sham equations.
The unique feature of the code is its **emphasis on simplicity
and flexibility**
with the goal of facilitating methodological development and
interdisciplinary collaboration.
In about 5k lines of pure Julia code
we already support a [sizeable set of features](@ref package-features),
after just a good year of development.
Our performance is of the same order of magnitude as much larger production
codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/).

## [Package features](@id package-features)
* Methods and models:
    - Kohn-Sham-like models, with an emphasis on flexibility: compose your own model,
      from Cohen-Bergstresser linear eigenvalue equations to Gross-Pitaevskii equations
      and sophisticated LDA/GGA functionals (any functional from the
      [libxc](https://tddft.org/programs/libxc/) library)
    - Analytic potentials or Godecker norm-conserving pseudopotentials (GTH, HGH)
    - Brillouin zone symmetry for k-Point sampling using [spglib](https://atztogo.github.io/spglib/)
    - Smearing functions for metals
    - Self-consistent field approaches: Damping, Kerker mixing, Anderson/Pulay/DIIS mixing
    - Direct minimization
    - Multi-level threading (kpoints, eigenvectors, FFTs, linear algebra)
    - 1D / 2D / 3D systems
    - Magnetic fields
    - Treat systems beyond 500 electrons
* Ground-state properties and post-processing:
    - Total energy
    - Forces
    - Density of states (DOS), local density of states (LDOS)
    - Band structures
    - Easy access to all intermediate quantities (e.g. density, Bloch waves)
* Support for arbitrary floating point types, including `Float32` (single precision)
  or `Double64` (from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)).
  For DFT this is currently restricted to LDA (with Slater exchange and VWN correlation).
* Third-party integrations:
    - Use structures prepared in [pymatgen](https://pymatgen.org),
      [ASE](https://wiki.fysik.dtu.dk/ase/) or [abipy](https://abinit.github.io/abipy/).
    - [asedftk](https://github.com/mfherbst/asedftk):
      DFTK-based calculator implementation for ASE.
    - Read data in [ETSF Nanoquanta](https://doi.org/10.1016/j.commatsci.2008.02.023) format.

Missing a feature? Look for an open issue or [create a new one](https://github.com/JuliaMolSim/DFTK.jl/issues).

## [Example index](@id example-index)
First, new users should take a look at the [Installation](@ref)
and [Tutorial](@ref) sections. More details about DFTK are explained
in the examples as we go along:

```@contents
Pages = [
    "examples/metallic_systems.md",
    "examples/pymatgen.md",
    "examples/ase.md",
    "examples/geometry_optimization.md",
    "examples/polarizability.md",
    "examples/gross_pitaevskii.md",
    "examples/gross_pitaevskii_2D.md",
    "examples/cohen_bergstresser.md",
    "examples/arbitrary_floattype.md",
    "examples/scf_callbacks.md",
    "examples/custom_solvers.md",
    "examples/custom_potential.md",
]
Depth = 1
```

These and more examples can be found in the
[`examples` directory](https://dftk.org/tree/master/examples) of the main code.

If you have a great example you think would fit here,
please open a [pull request](https://github.com/JuliaMolSim/DFTK.jl/pulls)!
