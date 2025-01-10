# Achieving DFT convergence

Some systems are tricky to converge. Here are some collected tips and tricks
you can try and which may help. Take these as a source
of inspiration for what you can try. Your mileage may vary.

```@setup convergence
using DFTK
using AtomsBuilder
model = model_DFT(bulk(:Si); functionals=LDA(),
                  pseudopotentials=Dict(:Si => "hgh/lda/si-q4"))
basis = PlaneWaveBasis(model; Ecut=15, kgrid=(3, 3, 3))
```

- Even if modelling an insulator, add a temperature to your [`Model`](@ref).
  Values up to `1e-2` atomic units may be sometimes needed. Note, that this
  can change the physics of your system, so if in doubt perform a second SCF
  with a lower temperature afterwards, starting from the final density of the first.

- Increase the history size of the Anderson acceleration
  by passing a custom `solver` to [`self_consistent_field`](@ref), e.g.
  ```@example convergence
  solver = scf_anderson_solver(; m=15)
  ```
  All keyword arguments are passed through to [`DFTK.AndersonAcceleration`](@ref).

- Try increasing convergence for for the bands in each SCF step
  by increasing the `ratio_ρdiff` parameter of the [`AdaptiveDiagtol`](@ref)
  algorithm. For example:
  ```@example convergence
  diagtolalg = AdaptiveDiagtol(; ratio_ρdiff=0.05)
  ```

- Increase the number of bands, which are fully converged in each SCF step
  by tweaking the [`AdaptiveBands`](@ref) algorithm. For example:
  ```@example convergence
  nbandsalg = AdaptiveBands(model; temperature_factor_converge=1.1)
  ```

- Try the adaptive damping algorithm by
  using [`DFTK.scf_potential_mixing_adaptive`](@ref)
  instead of `self_consistent_field`:
  ```@example convergence
  DFTK.scf_potential_mixing_adaptive(basis; tol=1e-10)
  ```
