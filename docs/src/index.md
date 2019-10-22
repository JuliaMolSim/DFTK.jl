# DFTK.jl: The density-functional toolkit.

DFTK is a `julia` package for playing with plane-wave
density-functional theory algorithms. This page documents conventions
used in the code.

## Terminology and definitions
- $\mathcal R$ is the real-space lattice. A basis is given in `lattice`.
- $\mathcal R^*$ is the reciprocal-space lattice. A basis is given in `recip_lattice`. TODO demonstrate `recip_lattice'lattice = 2π I`
- $\Omega$ is an arbitrary unit cell in real space.
- $\mathcal B$ is the Brillouin zone, an arbitrary unit cell in reciprocal space. TODO which one do we take?
- The plane-waves are the orthonormal functions
```math
e_G(r) = 1/\sqrt{|\Omega|} e^{i\, G \cdot r}
```
for $G \in \mathcal R^*$.

## Basis sets and grids

- The **wave-function basis** $B_{k}^*$ ($B$ as in ball, $*$ to recall it's in reciprocal space), consisting of all
  plane-wave basis functions below the desired energy cutoff $E_\text{cut}$ for each $k$-point:
  ```math
  B_{k} = \{ e_G : 1/2 |G + k|^2 ≤ E_\text{cut}\}
  ```
- The **reciprocal-space basis** $C^*$ which is used for the densities and potentials. This is a rectangular grid to be able to perform FFTs. It is chosen to be able to represent densities and potentials, ie $e_{G-G'} \in C^*$ for $e_G, e_{G'} \in B_k$. We choose (TODO add link to the code) for $C^*$ the smallest rectangular basis set such that
```math
\{ e_G : 1/2 |G|_\text{max}^2 ≤ 4 E_\text{cut} \} \subset C^*
```
TODO add a demo of that property. Optionally, this can be enlarged when more precision is required on the exchange-correlation energy, with a `supersampling` factor that replaces the factor 4 above.
- The **real-space grid** $C$, which is the dual of $C^*$

The IFFT operation converts from $C^*$ to $C$. $B_k^* \subset C^*$, so zero-padding is used when necessary. The FFT converts from $C$ to $C^*$.

## Normalization and conventions
Reduced units are used everywhere in the code: $R$ and $G$ are expressed in units of `lattice` and `recip_lattice` respectively, `r` in fractional coordinates, etc. TODO demonstrate

The basis set $B_k^*$ has orthonormal functions, and we use the same convention for $C^*$. Quantities expressed in the real-space grid $C$ are in real units; this means that the IFFT has a prefactor $1/\sqrt{|Ω|}$, and the FFT a prefactor $\sqrt{|Ω|}/(N_g)$. TODO fix that in the code

## The energy
The unknowns are the periodic parts $u_{nk}$ of the Bloch waves $\psi_{nk}(x) = e^{ikx} u_{nk}(x)$, discretized on $B_k$. TODO write down the energy as a function of the coefficients. Fix a consistent notation for the coefficients that we use in the code.

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
DFTK.update_potential!
DFTK.update_energies_potential!
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
