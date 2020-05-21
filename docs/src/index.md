# DFTK.jl: The density-functional toolkit.

DFTK is a `julia` package of for playing with plane-wave
density-functional theory algorithms. In its basic formulation, it
solves the periodic Kohn-Sham equations. 

The following documentation is an overview of the structure of the
code, and of the formalism used. It assumes basic familiarity with the
concepts of plane-wave density functional theory. Users wanting to
simply run computations or get an overview of features should first
look at the `examples/` directory.

# Notations and conventions
## Units

Atomic units are used throughout: lengths are in Bohr, energies in
Hartree. In particular, ``\hbar = m_e = e = 1``. For instance, the
Schrödinger equation for the electron of the hydrogen atom is
``i\partial_t \psi = -\frac 1 2 \Delta \psi - \frac 1 {|r|} \psi``.

## Coordinates

Computations take place in the unit cell of a lattice defined by a 3x3
matrix (`model.lattice`) with lattice vectors as columns. Note that
Julia stores matrices as column-major, so care has to be taken when
interfacing with other libraries in row-major languages (eg Python).
The reciprocal lattice `model.recip_lattice` (the lattice of Fourier
coefficients of functions with the periodicity of the lattice) is
defined by the matrix ``2\pi A^{-T}`` where ``A`` is the unit cell.

Vectors in real space (``r`` vectors in the unit cell, and ``R``
vectors of the lattice) and in reciprocal space (``k`` vectors in the
Brillouin zone - the unit cell of the reciprocal lattice - and ``G``
vectors in the reciprocal lattice) are represented in the code in
**reduced coordinates**. One switches to cartesian coordinates by
``x_{\rm cart} = M x_{\rm red}``, where ``M`` is either
`model.unit_cell` (for real-space vectors) or `model.recip_lattice`
(for reciprocal-space vectors). A useful relationship is ``b_{\rm
cart} \cdot a_{\rm cart}=2\pi b_{\rm red} \cdot a_{\rm red}`` if ``a``
and ``b`` are real-space and reciprocal-space vectors respectively.

## Naming conventions

DFTK liberally uses Unicode characters to represent greek characters
(eg ψ, ρ, ε...). Input them at the Julia REPL by their latex command
and TAB, and configure your editor to support easy input.

- Reciprocal-space vectors: ``k`` for vectors in the Brillouin zone, ``G`` for vectors of the reciprocal lattice, ``q`` for general vectors
- Real-space vectors: ``R`` for lattice vectors. ``r`` and ``x`` are usually used for unit for vectors in the unit cell and general vectors respectively, but this convention is less consistently applied.
- ``\Omega`` is the unit cell, and ``|\Omega|`` (or sometimes just ``\Omega``) is its volume.
- The Bloch waves are ``\psi_{nk}(x) = e^{ik\cdot x} u_{nk}(x)``,
  where ``n`` is the band index and ``k`` the kpoint. In the code we
  sometimes use ``\psi`` and ``u`` interchangeably.
- ``\varepsilon`` are the eigenvalues.
- ``\rho`` is the density.
- Normalized plane waves: 
```math
e_G(r) = \frac 1 {\sqrt{\Omega}} e^{i G \cdot r}
```
- ``Y^l_m`` are the complex spherical harmonics, and ``Y_{lm}`` the real ones.
- ``j_l`` are the Bessel functions. In particular, ``j_{0}(x) = \frac{\sin x}{x}``.

# Basic datastructures

## Model
The physical model to be solved is defined by the `Model`
datastructure. It contains the unit cell, number of electrons, atoms,
type of spin polarization and temperature. Each atom has an atomic
type (`Element`) specifying their number of valence electrons and the
potential (or pseudopotential) it creates. The `Model` structure also
contains the list of energy terms defining the model. These can be of
the following types (for now), defined in the `src/terms/` directory:

- Kinetic energy
- Local potential energy, either given by analytic potentials or
  specified by the type of atoms.
- Nonlocal potential energy, for norm-conserving pseudopotentials
- Nuclei energies (eg Ewald or pseudopotential correction)
- Hartree energy
- Exchange-correlation energy
- Power nonlinearities (useful for Gross-Pitaevskii type models)
- Magnetic field energy
- Entropy term

By mixing and matching these terms, the user can create custom models.
Convenience constructors are provided for commonly used models.

## PlaneWaveBasis

The `PlaneWaveBasis` datastructure handles the discretization of a
given `Model` in a plane-wave basis. The discretization is along two
axes: the Brillouin zone is sampled in a discrete set of ``k`` points
(controlled by the `kgrid` argument, or by an explicit list of
``k``-points), and the reciprocal lattice is restricted to a finite
set (controlled by the `Ecut` argument).

The periodic part of Bloch waves is expanded in a set of normalized
plane waves ``e_G``:
```math
\begin{aligned}
  \psi_{k}(x) &= e^{i k \cdot x} u_{k}(x)\\
  &= \sum_{G \in \mathcal R^{*}} c_{}  e^{i  k \cdot  x} e_{G}(x)
\end{aligned}
```
where ``\mathcal R^*`` is the set of reciprocal lattice vectors.
The ``c_{{G}}`` are ``\ell^{2}``-normalized. The summation is truncated to a
"spherical", ``k``-dependent basis set
```math
  S_{k} = \left\{G \in \mathcal R^{*} \,\middle|\, \frac 1 2 |k+ G|^{2} \le E_{\rm cut}\right\}
```
where ``E_{\rm cut}`` is the cutoff energy.

Densities involve terms like ``|\psi_{k}|^{2} = |u_{k}|^{2}`` and
therefore products ``e_{-{G}} e_{{G}'}`` for ``{G}, {G}'`` in
``X_{k}``. To represent these we use a "cubic", ``k``-independent
basis set large enough to contain the set ``\{{G}-G' \,|\, G, G' \in
S_{k}\}``. We can obtain the decomposition of densities on the
``e_{G}`` basis by a convolution, which can be performed efficiently
with FFTs. Potentials are discretized on this same set.

The normalization conventions used in the code is that quantities
stored in reciprocal space are coefficients in the ``e_{G}`` basis,
and quantities stored in real space use real physical values. This
means for instance that wavefunctions in the real space grid are
normalized as ``\frac{|\Omega|}{N} \sum_{r} |f(r)|^{2} = 1`` where
``N`` is the number of grid points.

The list of ``k`` points can be obtained with `basis.kpoints`; the
``G`` vectors of the "spherical", ``k``-dependent grid can be obtained
with `G_vectors(basis.kpoints[ik])` with an index `ik`. The list of
``G`` vectors (Fourier modes) of the "cubic", ``k``-independent basis
set can be obtained with `G_vectors(basis)`. The list of ``r`` vectors
(real-space grid) can be obtained with `r_vectors(basis)`.

## Storage of various quantities
Wavefunctions are stored in an array `ψ` as `ψ[ik][iG, iband]` where
`ik` is the index of the kpoint (in `basis.kpoints`), `iG` is the
index of the plane wave (in `G_vectors(basis.kpoints[ik])`) and
`iband` is the index of the band. Densities are usually stored in a
special type, `RealFourierArray`, from which the representation in
real and reciprocal space can be accessed using `ρ.real` and
`ρ.fourier` respectively.

# Useful formulas

- The Fourier transform is
```math
  \begin{aligned}
    \widehat{f}( q) = \int_{{\mathbb R}^{3}} e^{-i q \cdot  x} dx
  \end{aligned}
```
- Plane wave expansion formula
```math
  \begin{aligned}
    e^{i {q} \cdot {r}} =
  4 \pi \sum_{l = 0}^\infty \sum_{m = -l}^l
  i^l j_l(|q| |r|) Y_l^m(q/|q|) Y_l^{m\ast}(r/|r|)
\end{aligned}
```
- Spherical harmonics orthogonality
```math
\int_{\mathbb{S}^2} Y_l^{m*}(r)Y_{l'}^{m'}(r) dr
  = \delta_{l,l'} \delta_{m,m'}
```
This also holds true for real spherical harmonics.

- Fourier transforms of centered functions.
If 
``f({x}) = R(x) Y_l^m(x/|x|)``,
then
```math
\begin{aligned}
  \hat f( q)
  &= \int_{{\mathbb R}^3} R(x) Y_{l}^{m}(x/|x|) e^{-i {q} \cdot {x}} d{x} \\
  &= \sum_{l = 0}^\infty 4 \pi i^l 
  \sum_{m = -l}^l \int_{{\mathbb R}^3}
  R(x) j_{l'}(|q| |x|)Y_{l'}^{m'}(-q/|q|) Y_{l}^{m}(x/|x|)
   Y_{l'}^{m'\ast}(x/|x|)
  d{x} \\
  &= 4 \pi Y_{l}^{m}(-q/|q|) i^{l}
  \int_{{\mathbb R}^+} r^2 R(r) \ j_{l}(|q| r) dr
 \end{aligned}
```
This also holds true for real spherical harmonics.


# Crystal symmetries

For simplicity we will only deal with a monoatomic crystal ``\mathcal C \subset \mathbb R^3``, the extension being easy. A symmetry of the crystal is a real-space unitary matrix ``\tilde S`` and a real-space vector ``\tilde \tau`` such that
```math
\tilde{S} \mathcal{C} + \tilde{\tau} = \mathcal{C}.
```
The symmetries where ``\tilde S = 1`` and ``\tilde \tau`` is a lattice vector are always assumed and ignored in the following.

We can define a corresponding unitary operator
```math
 {U} : L^2_\text{per} \to L^2_\text{per}
```
with action
```math
 {U} : u \mapsto u\left( S^{-1} (x-\tau) \right),
```
where we set
```math
\begin{aligned}
S &= \tilde{S}^{-1}\\
\tau &= -\tilde{S}^{-1}\tilde{\tau}.
\end{aligned}
```

This unitary operator acts on the Fourier coefficients of lattice-periodic functions as
```math
(Uu)(G) = e^{-i G \cdot \tau} u(S^-1 G)
```
and so
```math
U (-i∇ + k) U^* = (-i∇ + Sk)
```
Furthermore, since the potential ``V`` is the sum over radial potentials centered at atoms, it is easily seen that ``U V U^* = V``, i.e. that ``U`` and ``V`` commute.

It follows that if the Bloch wave ``ψ_k(x) = e^{ik\cdot x} u_k(x)`` is an eigenfunction of the Hamiltonian, then ``e^{i (Sk) \cdot x} (Uu_k)(x)`` is also an eigenfunction, and so we can take
```math
u_{Sk} = U u_k.
```

This is used to reduce the computations needed. For a uniform sampling of the Brillouin zone (the "reducible kpoints"), one can find a reduced set of kpoints (the "irreducible kpoints") such that the eigenvectors at the reducible kpoints can be deduced from those at the irreducible kpoints.
