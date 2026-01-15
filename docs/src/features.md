# [DFTK features](@id package-features)

The following lists the functionality of DFTK
achieved in **less than 10 000 lines** of code.
Our code has a performance comparable to standard DFT codes
and runs out of the box on Linux, Windows and macOS, see [Installation](@ref).
We obtain similar results to standard codes, see
[the recent verification dataset by Bosoni and others](https://acwf-verification.materialscloud.org/).

## Standard methods and models
- Any DFT exchange-correlation functional from the [libxc](https://libxc.gitlab.io/) library
  at the LDA, GGA, meta-GGA level.
- [Hubbard correction (DFT+U)](@ref).
- **Norm-conserving pseudopotentials**: Goedecker-type (GTH)
  or numerical (in UPF pseudopotential format),
  see [Pseudopotentials](@ref).
- Collinear spin, see [Collinear spin and magnetic systems](@ref).
- **Black-box self-consistent field approaches**, such as
  [LDOS mixing](https://doi.org/10.1088/1361-648X/abcbdb) (autodetects metal versus insulator)
  or [adaptive damping](https://arxiv.org/abs/2109.14018).
- Direct minimisation methods, see [Comparison of DFT solvers](@ref).
- Various smearing methods, see [Temperature and metallic systems](@ref metallic-systems)
  and [Energy cutoff smearing](@ref).

## Parallelisation
- **MPI-based distributed parallelism** (distribution over ``k``-points)
- **[Using DFTK on GPUs](@ref)**: Nvidia *(mostly supported)* and AMD GPUs *(preliminary support)*
- Multi-level threading (``k``-points eigenvectors, FFTs, linear algebra)
- See also: [Using DFTK on compute clusters](@ref).

## Ground-state properties and post-processing
- Total energy, forces, stresses
- Density of states (DOS), local density of states (LDOS), projected density of states (PDOS)
- Band structures
- [Geometry optimization](@ref)
- Easy access to all intermediate quantities (e.g. density, Bloch waves)

## Response and response properties
- Density-functional perturbation theory (DFPT)
- Integration of DFPT with **algorithmic differentiation**,
  e.g. [Elastic constants](@ref),
  [Polarizability using automatic differentiation](@ref)
- [Phonon computations](@ref) *(preliminary implementation)*

## Unique features
- Support for **arbitrary floating point types**,
  including `Float32` (single precision)
  or `Double64` (from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)).
- Forward-mode algorithmic differentiation
  (see [Polarizability using automatic differentiation](@ref))
- Flexibility to **build your own Kohn-Sham model**:
  Anything from [analytic potentials](@ref custom-potential),
  linear [Cohen-Bergstresser model](@ref),
  the [Gross-Pitaevskii equation](@ref gross-pitaevskii),
  [Anyonic models](@ref), etc.
- Analytic potentials (see [Tutorial on periodic problems](@ref periodic-problems))
- 1D / 2D / 3D systems (see [Tutorial on periodic problems](@ref periodic-problems))

## Third-party integrations
- Many standard [Input and output formats](@ref).
- [AtomsBase integration](@ref) and via this ecosystem an integration
  with the [Atomistic simulation environment (ASE)](@ref).
- [Wannierization using Wannier.jl or Wannier90](@ref)

## Missing a feature?
Look for an open issue or [create a new one](https://github.com/JuliaMolSim/DFTK.jl/issues).
Want to contribute? See our [contributing notes](https://github.com/JuliaMolSim/DFTK.jl#contributing).
