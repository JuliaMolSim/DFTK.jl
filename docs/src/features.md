# [DFTK features](@id package-features)

* Standard methods and models:
    - Standard DFT models (LDA, GGA, meta-GGA): Any functional from the
      [libxc](https://tddft.org/programs/libxc/) library
    - Norm-conserving pseudopotentials: Goedecker-type (GTH, HGH)
      or numerical (in UPF pseudopotential format),
      see [Pseudopotentials](@ref) for details.
    - Brillouin zone symmetry for ``k``-point sampling using [spglib](https://atztogo.github.io/spglib/)
    - Standard smearing functions (including Methfessel-Paxton
      and Marzari-Vanderbilt cold smearing)
    - Collinear spin polarization for magnetic systems
    - Self-consistent field approaches including Kerker mixing,
      [LDOS mixing](https://doi.org/10.1088/1361-648X/abcbdb),
      [adaptive damping](https://arxiv.org/abs/2109.14018)
    - Direct minimization, Newton solver
    - Multi-level threading (``k``-points eigenvectors, FFTs, linear algebra)
    - MPI-based distributed parallelism (distribution over ``k``-points)
    - Treat systems of 1000 electrons

* Ground-state properties and post-processing:
    - Total energy
    - Forces, stresses
    - Density of states (DOS), local density of states (LDOS)
    - Band structures
    - Easy access to all intermediate quantities (e.g. density, Bloch waves)

* Unique features
    - Support for arbitrary floating point types, including `Float32` (single precision)
      or `Double64` (from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)).
    - Forward-mode algorithmic differentiation (see [Polarizability using automatic differentiation](@ref))
    - Flexibility to build your own Kohn-Sham model:
      Anything from [analytic potentials](@ref custom-potential),
      linear [Cohen-Bergstresser model](@ref),
      the [Gross-Pitaevskii equation](@ref gross-pitaevskii),
      [Anyonic models](@ref), etc.
    - Analytic potentials (see [Tutorial on periodic problems](@ref periodic-problems))
    - 1D / 2D / 3D systems (see [Tutorial on periodic problems](@ref periodic-problems))

* Third-party integrations:
    - Seamless integration with many standard [Input and output formats](@ref).
    - Integration with [ASE](https://wiki.fysik.dtu.dk/ase/) and
      [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) for passing
      atomic structures (see [AtomsBase integration](@ref)).
    - [Wannierization using Wannier.jl or Wannier90](@ref)


* Runs out of the box on Linux, macOS and Windows

Missing a feature? Look for an open issue or [create a new one](https://github.com/JuliaMolSim/DFTK.jl/issues).
Want to contribute? See our [contributing notes](https://github.com/JuliaMolSim/DFTK.jl#contributing).
