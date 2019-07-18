# DFTK.jl: The density-functional toolkit.

DFTK is a `julia` package of for playing with
plane-wave density-functional theory algorithms.


## Terminology and Definitions
General terminology used throughout the documentation
of the plane-wave aspects of the code.

## Plane wave basis functions
At the moment the code works exclusively with orthonormal plane waves.
In other words our bases consist of functions
```math
e_G = 1/\sqrt{\Omega} e^{i\, G \cdot x}
```
where $\Omega$ is the unit cell volume.

## Basis sets
- The **wave-function basis** $B_{Ψ,k}$ (used to be $X_k$), consisting of all
  plane-wave basis functions below the desired energy cutoff $E_\text{cut}$
  for each $k$-point:
  ```math
  B_{Ψ,k} = \{ e_G : 1/2 |G + k|^2 ≤ E_\text{cut}
  ```
- The **potential** or **density basis** $B_\rho$, consisting of
  all plane waves on which a potential needs to be known in order to be
  consistent with the union of all $B_{Ψ,k}$ for all $k$. In practice
  we do not take the smallest possible set of wave vectors $G$ for this, but
  instead the smallest *cubic* grid, which satisfies this, i.e.
  ```math
  B_\rho = \{ e_G : 1/2 |G|_\text{max}^2 ≤ α E_\text{cut} \},
  ```
  where a supersampling factor $\alpha = 4$ is required to give a numerically
  exact result, since
  ```math
  B_\rho = \{ e_{G+G'} : ∃k e_G, e_{G'} ∈ B_{Ψ,k} \}.
  ```
  The choice of using a cubic grid is done in order to be consistent with usual
  fast Fourier transform implementations, which work on cubic Fourier grids.
- The **XC basis** $B_\text{XC}$, which is used for computing the application
  of the exchange-correlation potential operator to the density $\rho$,
  represented in the basis $B_\rho$, that is
  ```math
  B_\text{XC}  = \{e_G : 1/2 |G|_\text{max}^2 ≤ β E_\text{cut} \}.
  ```
  Since the exchange-correlation potential might involve arbitrary powers of the
  density $ρ$, a numerically exact computation of the integral
  ```math
  \langle e_G | V_\text{XC}(ρ) e_{G'} \rangle \qquad \text{with} \qquad e_G, e_{G'} ∈ B_{Ψ,k}
  ```
  requires the exchange-correlation supersampling factor $\beta$ to be infinite.
  In practice, $\beta =4$ is usually chosen, such that $B_\text{XC} = B_\rho$.

## Real-space grids
Due to the Fourier-duality of reciprocal-space and real-space lattice,
the above basis sets define corresponding real-space grids as well:

- The grid $B_\rho^\ast$, the **potential integration grid**,
  which is the grid used for convolutions of a potential with the discretized
  representation of a DFT orbital. It is simply the iFFT-dual real-space grid
  of $B_\rho$.
- The grid $B^\ast_\text{XC}$, the **exchange-correlation integration grid**,
  i.e. the grid used for convolutions of the exchange-correlation functional
  terms with the density or derivatives of it. It is the iFFT-dual of $B_\text{XC}$.

## Core

```@docs
PlaneWaveBasis
set_kpoints!
basis_ρ
DFTK.G_to_r!
DFTK.r_to_G!
PotLocal
PotNonLocal
Kinetic
Hamiltonian
apply_hamiltonian!
DFTK.apply_fourier!
DFTK.apply_real!
DFTK.compute_potential!
DFTK.empty_potential
PreconditionerKinetic
DFTK.lobpcg
DFTK.lobpcg_qr
DFTK.lobpcg_scipy
DFTK.lobpcg_itsolve
DFTK.occupation_zero_temperature
self_consistent_field
DFTK.scf_nlsolve
DFTK.scf_damped
PspHgh
eval_psp_projection_radial
eval_psp_local_real
eval_psp_local_fourier
compute_density
```

## Utilities
```@docs
Species
determine_grid_size
build_local_potential
build_nonlocal_projectors
kblock_as_matrix
load_psp
guess_gaussian_sad
guess_hcore
```
