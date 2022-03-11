# Crystal symmetries
## Theory
In this discussion we will only describe the situation for a monoatomic crystal
``\mathcal C \subset \mathbb R^3``, the extension being easy.
A symmetry of the crystal is an orthogonal matrix ``W``
and a real-space vector ``w`` such that
```math
W \mathcal{C} + w = \mathcal{C}.
```
The symmetries where ``W = 1`` and ``w``
is a lattice vector are always assumed and ignored in the following.

We can define a corresponding unitary operator ``U`` on ``L^2(\mathbb R^3)``
with action
```math
 (Uu)(x) = u\left( W x + w \right).
```
We assume that the atomic potentials are radial and that any self-consistent potential
also respects this symmetry, so that ``U`` commutes with the Hamiltonian.

This operator acts on a plane-wave as
```math
\begin{aligned}
(U e^{iq\cdot x}) (x) &= e^{iq \cdot w} e^{i (W^T q) x}\\
&= e^{- i(S q) \cdot \tau } e^{i (S q) \cdot x}
\end{aligned}
```
where we set
```math
\begin{aligned}
S &= W^{T}\\
\tau &= -W^{-1}w.
\end{aligned}
```
It follows that the Fourier transform satisfies
```math
\widehat{Uu}(q) = e^{- iq \cdot \tau} \widehat u(S^{-1} q)
```
(all of these equations being also valid in reduced coordinates).
In particular, if ``e^{ik\cdot x} u_{k}(x)`` is an eigenfunction, then by decomposing
``u_k`` over plane-waves ``e^{i G \cdot x}`` one can see that
``e^{i(S^T k) \cdot x} (U u_k)(x)`` is also an eigenfunction: we can choose
```math
u_{Sk} = U u_k.
```

This is used to reduce the computations needed. For a uniform sampling of the
Brillouin zone (the *reducible ``k``-points*),
one can find a reduced set of ``k``-points
(the *irreducible ``k``-points*) such that the eigenvectors at the
reducible ``k``-points can be deduced from those at the irreducible ``k``-points.

## Symmetrization
Quantities that are calculated by summing over the reducible ``k`` points can be
calculated by first summing over the irreducible ``k`` points and then symmetrizing.
Let ``\mathcal{K}_\text{reducible}`` denote the reducible ``k``-points
sampling the Brillouin zone,
``\mathcal{S}`` be the group of all crystal symmetries that leave this BZ mesh invariant
(``\mathcal{S}\mathcal{K}_\text{reducible} = \mathcal{K}_\text{reducible}``)
and ``\mathcal{K}`` be the irreducible ``k``-points obtained
from ``\mathcal{K}_\text{reducible}`` using the symmetries ``\mathcal{S}``.
Clearly
```math
\mathcal{K}_\text{red} = \{Sk \, | \, S \in \mathcal{S}, k \in \mathcal{K}\}.
```

Let ``Q`` be a ``k``-dependent quantity to sum (for instance, energies, densities, forces, etc).
``Q`` transforms in a particular way under symmetries: ``Q(Sk) = S(Q(k))`` where the
(linear) action of ``S`` on ``Q`` depends on the particular ``Q``.
```math
\begin{aligned}
\sum_{k \in \mathcal{K}_\text{red}} Q(k)
&= \sum_{k \in \mathcal{K}} \ \sum_{S \text{ with } Sk \in \mathcal{K}_\text{red}} S(Q(k)) \\
&= \sum_{k \in \mathcal{K}} \frac{1}{N_{\mathcal{S},k}} \sum_{S \in \mathcal{S}} S(Q(k))\\
&= \frac{1}{N_{\mathcal{S}}} \sum_{S \in \mathcal{S}}
   \left(\sum_{k \in \mathcal{K}} \frac{N_\mathcal{S}}{N_{\mathcal{S},k}} Q(k) \right)
\end{aligned}
```
Here, ``N_\mathcal{S} = |\mathcal{S}|`` is the total number of symmetry operations and
``N_{\mathcal{S},k}`` denotes the number of operations such that leave ``k`` invariant.
The latter operations form a subgroup of the group of all symmetry operations,
sometimes called the *small/little group of ``k``*.
The factor ``\frac{N_\mathcal{S}}{N_{S,k}}``, also equal to the ratio of number of
reducible points encoded by this particular irreducible ``k`` to the total number of
reducible points, determines the weight of each irreducible ``k`` point.

## Example
```@setup symmetries
using DFTK
a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
Ecut = 5
kgrid = [4, 4, 4]
```
Let us demonstrate this in practice.
We consider silicon, setup appropriately in the `lattice`, `atoms` and `positions`
objects as in [Tutorial](@ref) and to reach a fast execution, we take a small `Ecut` of `5`
and a `[4, 4, 4]` Monkhorst-Pack grid.
First we perform the DFT calculation disabling symmetry handling
```@example symmetries
model_nosym = model_LDA(lattice, atoms, positions; symmetries=false)
basis_nosym = PlaneWaveBasis(model_nosym; Ecut, kgrid)
scfres_nosym = @time self_consistent_field(basis_nosym, tol=1e-8)
nothing  # hide
```
and then redo it using symmetry (the default):
```@example symmetries
model_sym = model_LDA(lattice, atoms, positions)
basis_sym = PlaneWaveBasis(model_sym; Ecut, kgrid)
scfres_sym = @time self_consistent_field(basis_sym, tol=1e-8)
nothing  # hide
```
Clearly both yield the same energy
but the version employing symmetry is faster,
since less ``k``-points are explicitly treated:
```@example symmetries
(length(basis_sym.kpoints), length(basis_nosym.kpoints))
```
Both SCFs would even agree in the convergence history
if exact diagonalization was used for the eigensolver
in each step of both SCFs.
But since DFTK adjusts this `diagtol` value adaptively during the SCF
to increase performance, a slightly different history is obtained.
Try adding the keyword argument
`determine_diagtol=(args...; kwargs...) -> 1e-8`
in each SCF call to fix the diagonalization tolerance to be `1e-8` for all SCF steps,
which will result in an almost identical convergence history.

We can also explicitly verify both methods to yield the same density:
```@example symmetries
using LinearAlgebra  # hide
(norm(scfres_sym.ρ - scfres_nosym.ρ),
 norm(values(scfres_sym.energies) .- values(scfres_nosym.energies)))
```

The symmetries can be used to map reducible to irreducible points:
```@example symmetries
ikpt_red = rand(1:length(basis_nosym.kpoints))
# find a (non-unique) corresponding irreducible point in basis_nosym,
# and the symmetry that relates them
ikpt_irred, symop = DFTK.unfold_mapping(basis_sym, basis_nosym.kpoints[ikpt_red])
[basis_sym.kpoints[ikpt_irred].coordinate symop.S * basis_nosym.kpoints[ikpt_red].coordinate]
```
The eigenvalues match also:
```@example symmetries
[scfres_sym.eigenvalues[ikpt_irred] scfres_nosym.eigenvalues[ikpt_red]]
```
