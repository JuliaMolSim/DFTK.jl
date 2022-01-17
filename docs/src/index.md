# DFTK.jl: The density-functional toolkit.

The density-functional toolkit, **DFTK** for short, is a library of
Julia routines for playing with plane-wave
density-functional theory (DFT) algorithms.
In its basic formulation it solves periodic Kohn-Sham equations.
The unique feature of the code is its **emphasis on simplicity
and flexibility**
with the goal of facilitating methodological development and
interdisciplinary collaboration.
In about 7k lines of pure Julia code
we support a [sizeable set of features](@ref package-features).
Our performance is of the same order of magnitude as much larger production
codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/).
DFTK's source code is [publicly available on github](https://dftk.org).

If you are new to density-functional theory or plane-wave methods,
see our notes on [Periodic problems](@ref periodic-problems) and our
collection of [lectures, workshops and literature on DFT](@ref density-functional-theory).

!!! tip "DFTK summer school: 29th to 31st August 2022 in Paris, France"
    We will organise a summer school centred around the DFTK code
    and modern approaches to density-functional theory
    from **29 to 31 August 2022** at **Sorbonne Université, Paris**.
    For more details and registration info see the [school's website](https://school2022.dftk.org).

## [Package features](@id package-features)
* Methods and models:
    - Kohn-Sham-like models, with an emphasis on flexibility: compose your own model,
      from Cohen-Bergstresser linear eigenvalue equations to Gross-Pitaevskii equations
      and sophisticated LDA/GGA functionals (any functional from the
      [libxc](https://tddft.org/programs/libxc/) library)
    - Analytic potentials or Godecker norm-conserving pseudopotentials (GTH, HGH)
    - Brillouin zone symmetry for ``k``-point sampling using [spglib](https://atztogo.github.io/spglib/)
    - Smearing functions for metals
    - Collinear spin polarization for magnetic systems
    - Self-consistent field approaches: Damping, Kerker mixing,
      [LDOS mixing](https://doi.org/10.1088/1361-648X/abcbdb), Anderson/Pulay/DIIS acceleration
    - Direct minimization, Newton solver
    - Multi-level threading (``k``-points eigenvectors, FFTs, linear algebra)
    - MPI-based distributed parallelism (distribution over ``k``-points)
    - 1D / 2D / 3D systems
    - External magnetic fields
    - Treat systems beyond 800 electrons
* Ground-state properties and post-processing:
    - Total energy
    - Forces, stresses
    - Density of states (DOS), local density of states (LDOS)
    - Band structures
    - Easy access to all intermediate quantities (e.g. density, Bloch waves)
* Support for arbitrary floating point types, including `Float32` (single precision)
  or `Double64` (from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)).
  For DFT this is currently restricted to LDA (with Slater exchange and VWN correlation).
* Runs out of the box on Linux, macOS and Windows
* Third-party integrations:
    - Seamless integration with many standard [Input and output formats](@ref).
    - Use structures prepared in [pymatgen](https://pymatgen.org),
      [ASE](https://wiki.fysik.dtu.dk/ase/) or [abipy](https://abinit.github.io/abipy/).
    - [asedftk](https://github.com/mfherbst/asedftk):
      DFTK-based calculator implementation for ASE.

Missing a feature? Look for an open issue or [create a new one](https://github.com/JuliaMolSim/DFTK.jl/issues).
Want to contribute? See our [contributing notes](https://github.com/JuliaMolSim/DFTK.jl#contributing).

## [Example index](@id example-index)
First, new users should take a look at the [Installation](@ref)
and [Tutorial](@ref) sections. More details about DFTK are explained
in the examples as we go along:

```@contents
Pages = [
    "examples/metallic_systems.md",
    "examples/pymatgen.md",
    "examples/ase.md",
    "examples/collinear_magnetism.md",
    "examples/geometry_optimization.md",
    "examples/scf_callbacks.md",
    "examples/scf_checkpoints.md",
    "examples/polarizability.md",
    "examples/gross_pitaevskii.md",
    "examples/gross_pitaevskii_2D.md",
    "examples/cohen_bergstresser.md",
    "examples/arbitrary_floattype.md",
    "examples/forwarddiff.md",
    "examples/custom_solvers.md",
    "examples/custom_potential.md",
    "examples/wannier90.md",
    "examples/error_estimates_forces.md",
]
Depth = 1
```

These and more examples can be found in the
[`examples` directory](https://dftk.org/tree/master/examples) of the main code.


!!! note "Convergence parameters in the documentation"
    In the documentation we use very rough convergence parameters to be able
    to automatically generate this documentation very quickly.
    Therefore results are far from converged.
    Tighter thresholds and larger grids should be used for
    more realistic results.

If you have a great example you think would fit here,
please open a [pull request](https://github.com/JuliaMolSim/DFTK.jl/pulls)!
