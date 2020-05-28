#src This file is fed to Literate, which translates it to Markdown.
#src The markdown is then used to generate docs. See `../make.jl`.

# # DFTK.jl: The density-functional toolkit.

# DFTK is a [Julia](https://julialang.org/) package for playing with plane-wave
# density-functional theory algorithms. In its basic formulation it
# solves periodic Kohn-Sham equations.

# The following documentation is an overview of the structure of the
# code, and of the formalism used. It assumes basic familiarity with the
# concepts of plane-wave density functional theory. Users wanting to
# simply run computations or get an overview of features should
# look at [`examples`](https://dftk.org/tree/master/examples)
# directory in the main code.

# In the following we will illustrate the concepts on the example
# of computing the LDA ground state of the Silicon crystal.

using DFTK
using Plots
using LinearAlgebra

## 1. Define lattice and atomic positions
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]

## Load HGH pseudopotential for Silicon
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))

## Specify type and positions of atoms
atoms = [Si => [ones(3)/8, -ones(3)/8]]

## 2. Select model and basis
model = model_LDA(lattice, atoms)
kgrid = [4, 4, 4]  # k-Point grid (Regular Monkhorst-Pack grid)
Ecut = 15          # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

## 3. Run the SCF
scfres = self_consistent_field(basis, tol=1e-8);

# # Notations and conventions
# ## Units

# DFTK uses atomic units throughout: lengths are in Bohr, energies in
# Hartree. In particular, ``\hbar = m_e = e = 4\pi \epsilon_0 = 1``.
# In this convention the Schrödinger equation for the electron of the hydrogen atom is
# ```math
# i\partial_t \psi = -\frac 1 2 \Delta \psi - \frac 1 {|r|} \psi.
# ```
# Useful conversion factors can be found in `DFTK.units`:

DFTK.units.eV

# ## Coordinates

# Computations take place in the unit cell of a lattice defined by a ``3 × 3``
# matrix (`model.lattice`) with lattice vectors as columns. Note that
# Julia stores matrices as column-major, so care has to be taken when
# interfacing with other libraries in row-major languages (e.g. Python).
# The reciprocal lattice `model.recip_lattice` (the lattice of Fourier
# coefficients of functions with the periodicity of the lattice) is
# defined by the matrix
# ```math
# B = 2\pi A^{-T} = 2π {A^{T}}^{-1}
# ```
# where ``A`` is the unit cell.

model.recip_lattice' * model.lattice

# Vectors in real space are denoted by ``r`` if they reside inside
# the unit cell and by ``R`` for lattice vectors. Vectors in
# reciprocal space are analogously ``k`` (for vectors in the Brillouin zone)
# and ``G`` for vectors on the reciprocal lattice. Commonly ``q``
# is used to refer to ``k + G``. Unless denoted otherwise the
# code uses **reduced coordinates**
# for these vectors. One switches to Cartesian coordinates by
# ```math
# x_\text{cart} = M x_\text{red}
# ```
# where ``M`` is either `model.lattice` (for real-space vectors) or
# `model.recip_lattice` (for reciprocal-space vectors).
# A useful relationship is
# ```math
# b_\text{cart} \cdot a_\text{cart}=2\pi b_\text{red} \cdot a_\text{red}
# ```
# if ``a`` and ``b`` are real-space and reciprocal-space vectors respectively.
# Other names for reduced coordinates are **integer coordinates**
# (usually for ``G``-vectors) or **fractional coordinates**
# (usually for ``k``-points).
#
# We use the convention that the unit cell in real space is ``[0,
# 1)^3`` (in reduced coordinates) and the unit cell in reciprocal
# space (the reducible Brillouin zone) is ``[-1/2, 1/2)^3``.

# ## Naming conventions

# DFTK liberally uses Unicode characters to represent Greek characters
# (e.g. ψ, ρ, ε...). Input them at the Julia REPL by their latex command
# and press "TAB".
# For all major editors there are
# [great Julia plugins](https://github.com/JuliaEditorSupport/)
# offering easy support for such characters as well.
#
# - **Reciprocal-space vectors:** ``k`` for vectors in the Brillouin zone,
#   ``G`` for vectors of the reciprocal lattice,
#   ``q`` for general vectors
# - **Real-space vectors:** ``R`` for lattice vectors,
#   ``r`` and ``x`` are usually used for unit for vectors in the unit cell
#   or general real-space vectors, respectively.
#   This convention is, however, less consistently applied.
# - ``\Omega`` is the unit cell, and ``|\Omega|``
#   (or sometimes just ``\Omega``) is its volume.
# - ``A`` are the real-space lattice vectors (`model.lattice`)
#   and ``B`` the Brillouin zone lattice vectors (`model.recip_lattice`).
# - The **Bloch waves** are
#   ```math
#   \psi_{nk}(x) = e^{ik\cdot x} u_{nk}(x),
#   ```
#   where ``n`` is the band index and ``k`` the ``k``-point. In the code we
#   sometimes use ``\psi`` and ``u`` interchangeably.
# - ``\varepsilon`` are the **eigenvaluesi**,
#   ``\varepsilon_F`` is the **Fermi level**.
# - ``\rho`` is the **density**.
# - In the code we use **normalized plane waves**:
#   ```math
#   e_G(r) = \frac 1 {\sqrt{\Omega}} e^{i G \cdot r}.
#   ```
# - ``Y^l_m`` are the complex **spherical harmonics**, and ``Y_{lm}`` the real ones.
# - ``j_l`` are the Bessel functions. In particular, ``j_{0}(x) = \frac{\sin x}{x}``.

# # Basic datastructures

# ## Model
# The physical model to be solved is defined by the `Model`
# datastructure. It contains the unit cell, number of electrons, atoms,
# type of spin polarization and temperature. Each atom has an atomic
# type (`Element`) specifying their number of valence electrons and the
# potential (or pseudopotential) it creates. The `Model` structure also
# contains the list of energy terms defining the model. These can be of
# the following types (for now), defined in the
# [`src/terms`](https://dftk.org/tree/master/src/terms) directory:

# - Kinetic energy
# - Local potential energy, either given by analytic potentials or
#   specified by the type of atoms.
# - Nonlocal potential energy, for norm-conserving pseudopotentials
# - Nuclei energies (eg Ewald or pseudopotential correction)
# - Hartree energy
# - Exchange-correlation energy
# - Power nonlinearities (useful for Gross-Pitaevskii type models)
# - Magnetic field energy
# - Entropy term

# By mixing and matching these terms, the user can create custom models.
# Convenience constructors are provided for commonly used models:
# - `model_DFT`: density-functional theory Hamiltonian using
#    any of the LDA or GGA functionals of the
#    [libxc](https://tddft.org/programs/libxc/functionals/)
#    library.
# - `model_LDA`: LDA Hamiltonian using the
#   [Teter parametrisation](https://doi.org/10.1103/PhysRevB.54.1703)

# For the silicon example above the following terms were used[^1]:

typeof.(model.term_types)

# [^1]: If you are not familiar with Julia syntax, this is equivalent to `[typeof(t) for t in model.term_types]`.
#
# DFTK computes energies for all terms of the model Hamiltonian.
# In the silicon example from above:

display(scfres.energies)

# ## PlaneWaveBasis

# The `PlaneWaveBasis` datastructure handles the discretization of a
# given `Model` in a plane-wave basis.
# As usual in plane-wave methods the discretization is twofold:
# Once the ``k``-point grid, which determines how the Brillouin zone
# is sampled in the discrete basis and once the sampling of the
# reciprocal-space lattice, which is restricted to a finite set
# of plane waves.
# The former aspect is controlled by the `kgrid` agrument
# (or by an explicit list of ``k``-points) and the latter is controlled
# by the cutoff energy parameter `Ecut`.

# The periodic parts of Bloch waves are expanded in a set of normalized
# plane waves ``e_G``:
# ```math
# \begin{aligned}
#   \psi_{k}(x) &= e^{i k \cdot x} u_{k}(x)\\
#   &= \sum_{G \in \mathcal R^{*}} c_{G}  e^{i  k \cdot  x} e_{G}(x)
# \end{aligned}
# ```
# where ``\mathcal R^*`` is the set of reciprocal lattice vectors.
# The ``c_{{G}}`` are ``\ell^{2}``-normalized. The summation is truncated to a
# "spherical", ``k``-dependent basis set
# ```math
#   S_{k} = \left\{G \in \mathcal R^{*} \,\middle|\, \frac 1 2 |k+ G|^{2} \le E_\text{cut}\right\}
# ```
# where ``E_\text{cut}`` is the cutoff energy.

# Densities involve terms like ``|\psi_{k}|^{2} = |u_{k}|^{2}`` and
# therefore products ``e_{-{G}} e_{{G}'}`` for ``{G}, {G}'`` in
# ``S_{k}``. To represent these we use a "cubic", ``k``-independent
# basis set large enough to contain the set
# ``\{{G}-G' \,|\, G, G' \in S_{k}\}``.
# We can obtain the coefficients of densities on the
# ``e_{G}`` basis by a convolution, which can be performed efficiently
# with FFTs (see `G_to_r` and `r_to_G` functions).
# Potentials are discretized on this same set.

# The normalization conventions used in the code is that quantities
# stored in reciprocal space are coefficients in the ``e_{G}`` basis,
# and quantities stored in real space use real physical values. This
# means for instance that wavefunctions in the real space grid are
# normalized as ``\frac{|\Omega|}{N} \sum_{r} |\psi(r)|^{2} = 1`` where
# ``N`` is the number of grid points.

# For example let us check the normalization of the first eigenfunction
# at the first ``k``-Point in reciprocal space:

ψtest = scfres.ψ[1][:, 1]
sum(abs2.(ψtest))

# We now perform an IFFT to get ψ in real space. The ``k``-Point has to be
# passed because ψ is expressed on the ``k``-dependent basis.
# Again the function is normalised:

ψreal = G_to_r(basis, basis.kpoints[1], ψtest)
sum(abs2.(ψreal)) * model.unit_cell_volume / prod(basis.fft_size)

# The list of ``k`` points can be obtained with `basis.kpoints`.

basis.kpoints

# The ``G`` vectors of the "spherical", ``k``-dependent grid can be obtained
# with `G_vectors(basis.kpoints[ik])` with an index `ik`:

[length(G_vectors(k)) for k in basis.kpoints]
#-
ik = 1
G_vectors(basis.kpoints[ik])[1:4]

# The list of ``G`` vectors (Fourier modes) of the "cubic", ``k``-independent basis
# set can be obtained with `G_vectors(basis)`.

length(G_vectors(basis)), prod(basis.fft_size)
#-
collect(G_vectors(basis))[1:4]
#-

# Analogously the list of ``r`` vectors
# (real-space grid) can be obtained with `r_vectors(basis)`:

length(r_vectors(basis))
#-
collect(r_vectors(basis))[1:4]

# As seen above, wavefunctions are stored in an array `ψ`
# as `ψ[ik][iG, iband]` where
# `ik` is the index of the kpoint (in `basis.kpoints`), `iG` is the
# index of the plane wave (in `G_vectors(basis.kpoints[ik])`) and
# `iband` is the index of the band.
# Densities are usually stored in a
# special type, `RealFourierArray`, from which the representation in
# real and reciprocal space can be accessed using `ρ.real` and
# `ρ.fourier` respectively.

rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
plot(x, scfres.ρ.real[:, 1, 1], label="", xlabel="x", ylabel="ρ", marker=2)
#-
G_energies = [sum(abs2.(model.recip_lattice * G)) ./ 2 for G in G_vectors(basis)][:]
scatter(G_energies, abs.(scfres.ρ.fourier[:]);
        yscale=:log10, ylims=(1e-12, 1), label="", xlabel="Energy", ylabel="|ρ|^2")

# (the density has no components on wavevectors above a certain energy,
# because the wavefunctions are limited to ``\frac 1 2|k+G|^2 ≤ E_{\rm cut}``)

# # Useful formulas

# - The Fourier transform is
# ```math
# \widehat{f}( q) = \int_{{\mathbb R}^{3}} e^{-i q \cdot  x} dx
# ```
# - Plane wave expansion formula
# ```math
# e^{i {q} \cdot {r}} =
#      4 \pi \sum_{l = 0}^\infty \sum_{m = -l}^l
#      i^l j_l(|q| |r|) Y_l^m(q/|q|) Y_l^{m\ast}(r/|r|)
# ```
# - Spherical harmonics orthogonality
# ```math
#    \int_{\mathbb{S}^2} Y_l^{m*}(r)Y_{l'}^{m'}(r) dr
#      = \delta_{l,l'} \delta_{m,m'}
# ```
# This also holds true for real spherical harmonics.

# - Fourier transforms of centered functions: If
# ``f({x}) = R(x) Y_l^m(x/|x|)``,
# then
# ```math
# \begin{aligned}
#   \hat f( q)
#   &= \int_{{\mathbb R}^3} R(x) Y_{l}^{m}(x/|x|) e^{-i {q} \cdot {x}} d{x} \\
#   &= \sum_{l = 0}^\infty 4 \pi i^l 
#   \sum_{m = -l}^l \int_{{\mathbb R}^3}
#   R(x) j_{l'}(|q| |x|)Y_{l'}^{m'}(-q/|q|) Y_{l}^{m}(x/|x|)
#    Y_{l'}^{m'\ast}(x/|x|)
#   d{x} \\
#   &= 4 \pi Y_{l}^{m}(-q/|q|) i^{l}
#   \int_{{\mathbb R}^+} r^2 R(r) \ j_{l}(|q| r) dr
#  \end{aligned}
# ```
# This also holds true for real spherical harmonics.


# # Crystal symmetries

# In this discussion we will only describe the situation for a monoatomic crystal
# ``\mathcal C \subset \mathbb R^3``, the extension being easy.
# A symmetry of the crystal is a real-space unitary matrix ``\tilde{S}``
# and a real-space vector ``\tilde{τ}`` such that
# ```math
# \tilde{S} \mathcal{C} + \tilde{\tau} = \mathcal{C}.
# ```
# The symmetries where ``\tilde S = 1`` and ``\tilde{τ}``
# is a lattice vector are always assumed and ignored in the following.

# We can define a corresponding unitary operator
# ``{U} : L^2_\text{per} \to L^2_\text{per}``
# with action
# ```math
#  (Uu)(x) = u\left( S^{-1} (x-\tau) \right),
# ```
# where we set
# ```math
# \begin{aligned}
# S &= \tilde{S}^{-1}\\
# \tau &= -\tilde{S}^{-1}\tilde{\tau}.
# \end{aligned}
# ```

# This unitary operator acts on the Fourier coefficients of lattice-periodic functions as
# ```math
# (Uu)(G) = e^{-i G \cdot \tau} u(S^{-1} G)
# ```
# and so
# ```math
# U (-i∇ + k) U^* = (-i∇ + Sk)
# ```
# Furthermore, since the potential ``V`` is the sum over radial potentials centered
# at atoms, it is easily seen that ``U V U^* = V``, i.e. that ``U`` and ``V`` commute.

# It follows that if the Bloch wave ``ψ_k = e^{ik\cdot x} u_k`` is an eigenfunction
# of the Hamiltonian, then ``e^{i (Sk) \cdot x} (Uu_k)`` is also an eigenfunction,
# and so we can take
# ```math
# u_{Sk} = U u_k.
# ```

# This is used to reduce the computations needed. For a uniform sampling of the
# Brillouin zone (the *reducible ``k``-Points*),
# one can find a reduced set of ``k``-Points
# (the *irreducible ``k``-Points*) such that the eigenvectors at the
# reducible ``k``-Points can be deduced from those at the irreducible ``k``-Points.

basis_irred = basis
scfres_irred = scfres
## Redo the same computation but disabling symmetry handling
basis_red = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)
scfres_red = self_consistent_field(basis_red, tol=1e-8)
(norm(scfres_irred.ρ.real - scfres_red.ρ.real),
 norm(values(scfres_irred.energies) .- values(scfres_red.energies)))

# The results are identical up to the convergence threshold,
# but compared to the first calculation on the top of the page,
# disabling symmetry leads to a substantially larger computational time,
# since more ``k``-Points are explicitly treated:
(length(basis_red.kpoints), length(basis_irred.kpoints))
#-
# !!!note "The `tol` argument in `self_consistent_field`"
#     The `tol` argument to `self_consistent_field` is a convergence threshold
#     in the total energy, such that less agreement is found in the density.
#
# To demonstrate the mapping, let us consider an example:
ikpt_irred = 2 # pick an arbitrary kpoint in the irreducible BZ
kpt_irred_coord = basis_irred.kpoints[ikpt_irred].coordinate
basis_irred.ksymops[ikpt_irred]
#-
# This is a list of all symmetries operations ``(S,\tau)``
# that can be used to map this irreducible kpoint to reducible kpoints.
# Let's pick the third symmetry operation of this ``k``-Point and check.
S, τ = basis_irred.ksymops[ikpt_irred][3]
kpt_red_coord = S * basis_irred.kpoints[ikpt_irred].coordinate
ikpt_red = findfirst(kcoord -> kcoord ≈ kpt_red_coord,
                     [k.coordinate for k in basis_red.kpoints])
(scfres_irred.eigenvalues[ikpt_irred], scfres_red.eigenvalues[ikpt_red])
