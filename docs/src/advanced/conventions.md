# Notation and conventions

## Usage of unicode characters
DFTK liberally uses unicode characters to represent Greek characters
(e.g. ψ, ρ, ε...). Make sure you use the proper
[Julia plugins](https://github.com/JuliaEditorSupport/)
to simplify typing them.

## [Symbol conventions](@id symbol-conventions)

- **Reciprocal-space vectors:** ``k`` for vectors in the Brillouin zone,
  ``G`` for vectors of the reciprocal lattice,
  ``q`` for general vectors
- **Real-space vectors:** ``R`` for lattice vectors,
  ``r`` and ``x`` are usually used for unit for vectors in the unit cell
  or general real-space vectors, respectively.
  This convention is, however, less consistently applied.
- ``\Omega`` is the unit cell, and ``|\Omega|``
  (or sometimes just ``\Omega``) is its volume.
- ``A`` are the real-space lattice vectors (`model.lattice`)
  and ``B`` the Brillouin zone lattice vectors (`model.recip_lattice`).
- The **Bloch waves** are
  ```math
  \psi_{nk}(x) = e^{ik\cdot x} u_{nk}(x),
  ```
  where ``n`` is the band index and ``k`` the ``k``-point. In the code we
  sometimes use ``\psi`` and ``u`` interchangeably.
- ``\varepsilon`` are the **eigenvalues**,
  ``\varepsilon_F`` is the **Fermi level**.
- ``\rho`` is the **density**.
- In the code we use **normalized plane waves**:
  ```math
  e_G(r) = \frac 1 {\sqrt{\Omega}} e^{i G \cdot r}.
  ```
- ``Y^l_m`` are the complex **spherical harmonics**, and ``Y_{lm}`` the real ones.
- ``j_l`` are the **Bessel functions**. In particular, ``j_{0}(x) = \frac{\sin x}{x}``.

## Units
In DFTK atomic units are used thoughout, most importantly
lengths are in Bohr and energies in Hartree.
See [wikipedia](https://en.wikipedia.org/wiki/Hartree_atomic_units)
for a list of conversion factors. Useful conversion factors
can also be found in `DFTK.units` and using [`DFTK.unit_to_au`](@ref).

## [Lattices and lattice vectors](@id conventions-lattice)
Both the real-space lattice (i.e. `model.lattice`) and reciprocal-space lattice
(`model.recip_lattice`) contain the lattice vectors in columns. If 1D or 2D
problems are to be treated these arrays are still ``3 \times 3`` matrices,
but contain two or one zero-columns, respectively.
The real-space lattice vectors are sometimes referred to by ``A`` and the
reciprocal-space lattice vectors by ``B = 2\pi A^{-T}``.

We use the convention that the unit cell in real space is
``[0, 1)^3`` in reduced coordinates and the unit cell in reciprocal
space (the reducible Brillouin zone) is ``[-1/2, 1/2)^3``.

## Lattice coordinate systems
Unless denoted otherwise the code uses **reduced coordinates**
for reciprocal-space vectors such as ``k``,  ``G``, ``q``
or real-space vectors like ``r`` and ``R``
(see [Symbol conventions](@ref symbol-conventions)).
One switches to Cartesian coordinates by
```math
x_\text{cart} = M x_\text{red}
```
where ``M`` is either ``A`` / `model.lattice` (for real-space vectors) or
``B`` / `model.recip_lattice` (for reciprocal-space vectors).
A useful relationship is
```math
b_\text{cart} \cdot a_\text{cart}=2\pi b_\text{red} \cdot a_\text{red}
```
if ``a`` and ``b`` are real-space and reciprocal-space vectors respectively.
Other names for reduced coordinates are **integer coordinates**
(usually for ``G``-vectors) or **fractional coordinates**
(usually for ``k``-points).

## Normalisation convention
The normalization conventions used in the code is that quantities
stored in reciprocal space are coefficients in the ``e_{G}`` basis,
and quantities stored in real space use real physical values.
This means for instance that wavefunctions in the real space grid are
normalized as ``\frac{|\Omega|}{N} \sum_{r} |\psi(r)|^{2} = 1`` where
``N`` is the number of grid points
and in reciprocal space its coefficients are ``\ell^{2}``-normalized,
see the [Tutorial](@ref) for an example where this is demonstrated.
