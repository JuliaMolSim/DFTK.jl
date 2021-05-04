# Crystal symmetries
## Theory
In this discussion we will only describe the situation for a monoatomic crystal
``\mathcal C \subset \mathbb R^3``, the extension being easy.
A symmetry of the crystal is an orthogonal matrix ``\widetilde{S}``
and a real-space vector ``\widetilde{\tau}`` such that
```math
\widetilde{S} \mathcal{C} + \widetilde{\tau} = \mathcal{C}.
```
The symmetries where ``\widetilde{S} = 1`` and ``\widetilde{\tau}``
is a lattice vector are always assumed and ignored in the following.

We can define a corresponding unitary operator ``U`` on ``L^2(\mathbb R^3)``
with action
```math
 (Uu)(x) = u\left( \widetilde{S} x + \widetilde{\tau} \right).
```
We assume that the atomic potentials are radial and that any self-consistent potential
also respects this symmetry, so that ``U`` commutes with the Hamiltonian.

This operator acts on a plane-wave as
```math
\begin{aligned}
(U e^{iq\cdot x}) (x) &= e^{iq \cdot \widetilde{\tau}} e^{i (\widetilde{S}^T q) x}\\
&= e^{- i(S q) \cdot \tau } e^{i (S q) \cdot x}
\end{aligned}
```
where we set
```math
\begin{aligned}
S &= \widetilde{S}^{T}\\
\tau &= -\widetilde{S}^{-1}\widetilde{\tau}.
\end{aligned}
```
(these equations being also valid in reduced coordinates).

It follows that the Fourier transform satisfies
```math
\widehat{Uu}(q) = e^{- iq \cdot \tau} \widehat u(S^{-1} q)
```
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

## Example
```@setup symmetries
using DFTK
a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
Ecut = 5
kgrid = [4, 4, 4]
```
Let us demonstrate this in practice.
We consider silicon, setup appropriately in the `lattice` and `atoms` objects
as in [Tutorial](@ref) and to reach a fast execution, we take a small `Ecut` of `5`
and a `[4, 4, 4]` Monkhorst-Pack grid.
First we perform the DFT calculation disabling symmetry handling
```@example symmetries
model = model_LDA(lattice, atoms)
basis_nosym = PlaneWaveBasis(model, Ecut; kgrid=kgrid, use_symmetry=false)
scfres_nosym = @time self_consistent_field(basis_nosym, tol=1e-8)
nothing  # hide
```
and then redo it using symmetry (the default):
```@example symmetries
basis_sym = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
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

To demonstrate the mapping between `k`-points due to symmetry,
we pick an arbitrary `k`-point in the irreducible Brillouin zone:
```@example symmetries
ikpt_irred = 2
kpt_irred_coord = basis_sym.kpoints[ikpt_irred].coordinate
basis_sym.ksymops[ikpt_irred]
```
This is a list of all symmetries operations ``(S, \tau)``
that can be used to map this irreducible ``k``-point to reducible ``k``-points.
Let's pick the third symmetry operation of this ``k``-point and check.
```@example symmetries
S, τ = basis_sym.ksymops[ikpt_irred][3]
kpt_red_coord = S * basis_sym.kpoints[ikpt_irred].coordinate
ikpt_red = findfirst(kcoord -> kcoord ≈ kpt_red_coord,
                     [k.coordinate for k in basis_nosym.kpoints])
[scfres_sym.eigenvalues[ikpt_irred] scfres_nosym.eigenvalues[ikpt_red]]
```
