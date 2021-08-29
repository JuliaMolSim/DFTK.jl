# # Introduction to periodic problems and plane-wave discretisations
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/guide/@__NAME__.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/guide/@__NAME__.ipynb)

# In this example we want to show how DFTK can be used to solve simple one-dimensional
# periodic problems. Along the lines this notebook serves as a concise introduction into
# the underlying theory and jargon for solving periodic problems using plane-wave
# discretisations.

# ## Periodicity and lattices
# A periodic problem is characterised by being invariant to certain translations.
# For example the ``sin`` function is periodic with periodicity ``2π``, i.e.
# ```math
#    sin(x) = sin(x + n ⋅ 2π) \quad ∀ n ∈ \mathbb{Z},
# ```
# which is nothing else than saying that any translation by an integer multiple of ``2π``
# keeps the ``sin`` function invariant. A more formal way of writing this is using the
# translation operator ``T_{-2πn}``:
# ```math
#    T_{-2πn} sin(x) = sin(x + 2πn) = sin(x).
# ```
#
# Whenever such periodicity exists it offers the potential to save some computational work.
# Consider a problem where one wants to find a function ``f : \mathbb{R} → \mathbb{R}``,
# but one a priori knows this solution to be periodic with periodicity ``a``. As a consequence
# of said periodicity it is sufficient to know the values of ``f`` for all ``x`` from the
# interval ``[-a/2, a/2)`` to uniquely define ``f`` on the full real axis. Naturally exploiting
# the periodicity in a computational procedure thus greatly reduces the amount of work.
#
# Let us introduce some jargon: The periodicity of our problem implies that we may define
# a **lattice**
# ```
#           -a       -a/2      +a/2       +a
#        ... |---------|---------|---------| ...
#                 a         a         a
# ```
# with lattice constant ``a``. Each cell of the lattice is an identical periodic image of
# any of its neighbours. For finding ``f`` it is thus sufficient to consider only the
# problem inside the **unit cell** ``[-a/2, a/2)``. In passing we note that the definition
# of the unit cell is arbitrary up to translations. A choice ``[0, a)``, for example,
# would have done just as well.
#
# ## Periodic operators and the Bloch transform
# Not only functions, but also operators can feature periodicity.
# Consider for example the **free-electron Hamiltonian**
# ```math
#     H = -\frac12 Δ.
# ```
# The free-electron model is a model in which the electron motion is only governed
# by their own kinetic energy. As this model features no potential which could make one point
# in space more preferred than another, we would expect this model to be periodic.
# If an operator is periodic with respect to a lattice like the one defined above,
# than it commutes with all lattice translations. For the free-electron case ``H``
# one can easily show exactly that, i.e.
# ```math
#    T_{na} H = H T_{na} \quad  ∀ n ∈ \mathbb{Z}.
# ```
#
# **Bloch's theorem** now tells us that for such types of operators,
# the solutions to the eigenproblem
# ```math
#     H v_{kj} = ε_{kj} v_{kj}
# ```
# satisfy a factorisation
# ```math
#     v_{kj}(x) = e^{i k⋅x} ψ_{kj}(x)
# ```
# into a plane wave ``e^{i k⋅x}`` and a lattice-periodic function
# ```math
#    T_{na} ψ_{kj}(x) = ψ_{kj}(x - na) = ψ_{kj}(x) \quad ∀ n ∈ \mathbb{Z}.
# ```
# In this ``j`` is a labelling integer index and ``k`` is a real number,
# whose details will be clarified in the next section.
# Functions ``v_{kj}`` satisfying this factorisation are also known as
# **Bloch functions** or **Bloch states**.
#
# Applying ``2H = -Δ = - \frac{d^2}{d x^2}`` to such a Bloch wave yields
# ```math
# \begin{aligned}
# -\Delta \left(e^{i k⋅x} ψ_{kj}(x)\right)
#    &= k^2 e^{ik⋅x} ψ_{kj}(x) - 2ik e^{ik⋅x} (∇ψ_{kj}(x)) - e^{ik⋅x} (Δψ_{kj}(x))
#    &= e^{ik⋅x} (k^2 - 2ik ∇ - Δ) ) ψ_{kj}(x) \\
#    &= e^{i k⋅x} (-i∇ + k)^2 ψ_{kj}(x) \\
#    &= e^{i k⋅x} 2H_k ψ_{kj}(x),
# \end{aligned}
# where we used
# ```math
#     H_k = \frac12 (-i∇ + k)^2.
# ```
# The action of this operator on a function ``ψ_{kj}`` is given by
# ```math
#     H_k ψ_{kj} = e^{-i k⋅x} H e^{i k⋅x} ψ_{kj},
# ```
# which in particular implies that
# ```math
#    H_k ψ_{kj} = ε_{kj} ψ_{kj} \quad \Rightleftarrow \quad H (e^{i k⋅x} ψ_{kj}) = ε_{kj} (e^{i k⋅x} ψ_{kj}).
# ```
# To seek the eigenpairs of ``H`` we may thus equivalently
# find the eigenpairs of *all* ``H_k``.
#
# A more detailed mathematical analysis shows that the transformation from ``H``
# to the set of all ``H_k`` for a suitable set of values for ``k`` (detailes below)
# is actually a unitary transformation, the so-called **Bloch transform**.
# This transform brings the Hamiltonian into the symmetry-adapted basis for
# translational symmetry, which are exactly the Bloch functions.
# Similar to the case of choosing a symmetry-adapted basis for other kinds of symmetries
# (like the point group symmetry in molecules), the Bloch transform also makes
# the Hamiltonian ``H`` block-diagonal[^1]:
# ```math
#     T_B H T_B^{-1} \longrightarrow \left( \begin{array}{cccc} H_1 \\ &H_2\\&&H_3\\&&&\ddots \right)
# ```
# with each block ``H_k`` taking care of one value ``k``.
# This block-diagonal structure under the basis of Bloch functions lets us
# completly describe the spectrum of ``H`` by looking only at the the spectrum
# of all ``H_k`` blocks.
#
# [^1]: Notice that block-diagonal is a bit an abuse of terms here, since the Hamiltonian
#       is not a matrix but an operator and the number of blocks is essentially infinite.
#       The mathematically precise term is that the Bloch transform reveals the fibers
#       of the Hamiltonian.
#
# ## The Brillouin zone
#
# The big mystery in the discussion so far is the parameter ``k`` of the Hamiltonian blocks.
#
# - As discussed ``k`` is essentially a real number. It turns out, however, that some of
#   these ``k∈\mathbb{R}`` give rise to operators related by unitary transformations
#   (again due to translational symmetry).
# - Since such operators have the same eigenspectrum, only one version needs to be considered.
# - The smallest subset from which ``k`` is chosen is the **Brillouin zone** (BZ).
#
# - The BZ is the unit cell of the **reciprocal lattice**, which may be constructed from
#   the **real-space lattice** by a Fourier transform.
# - In our simple 1D case the reciprocal lattice is just
#   ```
#     ... |--------|--------|--------| ...
#            2π/a     2π/a     2π/a
#   ```
#   i.e. like the real-space lattice, but just with a different lattice constant
#   ``b = 2π / a``.
# - The BZ in our example is thus ``B = [-π/a, π/a)``. The members of ``B``
#   are typically called ``k``-points.
#
# ## Discretisation and plane-wave basis sets
#
# With what we discussed so far the strategy to find all eigenpairs of a periodic
# Hamiltonian ``H`` thus reduces to finding the eigenpairs of all ``H_k`` with ``k ∈ B``.
# This requires *two* discretisations:
#
#   - ``B`` is an overcountable set. To discretise we first only pick a finite number
#     of ``k``-points. Usually this **``k``-point samping** is done by picking ``k``-points
#     along a regular grid inside the BZ, the **``k``-grid**.A
#   - Each ``H_k`` is still an infinite-dimensional operator.
#     Following a standard Ritz-Galerkin ansatz we project the operator into a finite basis
#     and diagonalise the resulting matrix.
#
# For the second step multiple types of bases are used in practice (finite differences,
# finite elements, Gaussians, ...). In DFTK we currently support only plane-wave
# discretisations.
#
# For our 1D example normalised plane waves are defined as the functions
# ```math
# e_{G}(x) = \frac{e^{i G x}}{\sqrt{a}}  \qquad G \in b\mathbb{Z} $$
# ```
# and typically one forms basis sets from these by specifying a
# **kinetic energy cutoff** ``E_\text{cut}``:
# ```math
# \left\{ e_{G} \, \big| \, (G + k)^2 \leq 2E_\text{cut} \right\}
# ```
#
# ## Solving the free-electron Hamiltonian
#
# One typical approach to get physical insight into a Hamiltonian ``H`` is to plot
# a so-called **band structure**, that is the eigenvalues of ``H_k`` versus ``k``.
# In DFTK we achieve this using the following steps:
#
# Step 1: Build the 1D lattice. DFTK is mostly tailored for 3D problems.
# Therefore quantities related to the problem space are have a fixed
# dimension 3. The convention is that for 1D / 2D problems the
# tailling entries are always zero and ignored in the computation.
# For the lattice we therefore construct a 3x3 matrix with only one entry.
using DFTK

lattice = zeros(3, 3)
lattice[1, 1] = 20.

# Step 2: Select a model. In this case we choose a free-electron model,
# which is the same as saying that there is only a Kinetic term
# (and no potential) in the model. The `n_electrons` is dummy here.
model = Model(lattice; n_electrons=0, terms=[Kinetic()])

# Step 3: Define a plane-wave basis using this model and a cutoff ``E_\text{cut}``
# of 300 Hartree. The ``k``-point grid is given as a regular grid in the BZ
# (a so-called **Monkhorst-Pack** grid). Here we select only one ``k``-point (1x1x1).
basis = PlaneWaveBasis(model; Ecut=300, kgrid=(1, 1, 1));

# !!! note "k-point grids in more complicated models"
#     You might wonder why we only selected a single ``k``-point (clearly a very crude
#     and inaccurate approximation). It turns out the `kgrid` parameter specified here
#     is not actually used for plotting the bands. It is only used when solving more
#     involved models like density-functional theory (DFT) where the Hamiltonian is
#     non-linear and before plotting the bands therefore the self-consistent field
#     equations need to be solved first. This is typically done on a different ``k``-point
#     grid than the grid used for the bands later on. In our case we don't need
#     this extra step and therefore the `kgrid` value passed to `PlaneWaveBasis`
#     is actually arbitrary.

# Step 4: Plot the bands! Select a density of ``k``-points for the ``k``-grid to use
# for the bandstructure calculation, discretise the problem and diagonalise it.
# Afterwards plot the bands.

using Unitful
using UnitfulAtomic
using Plots

n_bands = 6
ρ0 = guess_density(basis)  # Just dummy, has no meaning in this model
p  = plot_bandstructure(basis, ρ0, n_bands, kline_density=15, unit=u"hartree")

# ## Adding potentials
# So far so good. But free electrons are actually a little boring,
# so let's add a potential interacting with the electrons.
#
# - The modified problem we will look at consists of diagonalising
#   ```math
#   H_k = \frac12 (-i \nabla + k)^2 + V
#   ```
#   for all ``k \in B`` with a periodic potential ``V`` interacting with the electrons.
#
# - A number of "standard" potentials are readily implemented in DFTK and
#   can be assembled using the `terms` kwarg of the model.
#   This allows to seamlessly construct
#
#   * density-functial theory models for treating electronic structures
#     (see the [Tutorial](@ref)).
#   * Gross-Pitaevskii models for bosonic systems
#     (see [Gross-Pitaevskii equation in one dimension](@ref))
#   * even some more unusual cases like anyonic models.
#
# In this tutorial we will go a little more low-level and directly provide
# an analytic potential describing the interaction with the electrons to DFTK.
#
# First we define a custom Gaussian potential as a new "element" inside DFTK:

struct ElementGaussian <: DFTK.Element
    α  # Prefactor
    L  # Extend
end

## Some default values
ElementGaussian() = ElementGaussian(0.3, 10.0)

## Real-space representation of a Gaussian
function DFTK.local_potential_real(el::ElementGaussian, r::Real)
    -el.α / (√(2π) * el.L) * exp(- (r / el.L)^2 / 2)
end

## Fourier-space representation of the Gaussian
function DFTK.local_potential_fourier(el::ElementGaussian, q::Real)
    ## = ∫ -α exp(-(r/L)^2 exp(-ir⋅q) dr
    -el.α * exp(- (q * el.L)^2 / 2)
end

# A single potential looks like:

using Plots
nucleus = ElementGaussian()
plot(r -> DFTK.local_potential_real(nucleus, norm(r)), xlims=(-50, 50))

# With this element at hand we can easily construct a setting
# where two potentials of this form are located at positions
# ``20`` and ``80`` inside the lattice ``[0, 100]``:

using LinearAlgebra

## Define the 1D lattice [0, 100]
lattice = diagm([100., 0, 0])

## Place them at 20 and 80 in *fractional coordinates*,
## that is 0.2 and 0.8, since the lattice is 100 wide.
nucleus = ElementGaussian()
atoms = [nucleus => [[0.2, 0, 0], [0.8, 0, 0]]]

## Assemble the model, discretise and build the Hamiltonian
model = Model(lattice; atoms=atoms, terms=[Kinetic(), AtomicLocal()])
basis = PlaneWaveBasis(model; Ecut=300, kgrid=(1, 1, 1));
ham   = Hamiltonian(basis)

## Extract the total potential term of the Hamiltonian and plot it
potential = DFTK.total_local_potential(ham)[:, 1, 1]
rvecs = collect(r_vectors_cart(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                        # only keep the x coordinate
plot(x, potential, label="", xlabel="x", ylabel="V(x)")

# Notice how DFTK took care of the periodic wrapping of the potential values going
# from ``0`` and ``100``.
#
# With this setup, let's look at the bands:

using Unitful
using UnitfulAtomic

n_bands = 6
ρ0 = zeros(eltype(basis), basis.fft_size..., 1)  # Just dummy, has no meaning in this model
p = plot_bandstructure(basis, ρ0, n_bands, kline_density=15, unit=u"hartree")

# The bands are noticably different.
#  - The bands no longer overlap, meaning that the spectrum of $H$ is no longer continous
#    but has gaps.
#
# - The two lowest bands are almost flat, which means that they represent
#   two tightly bound and localised electrons inside the two Gaussians.
#
# - The higher the bands, the more curved they become. In other words the higher the
#   kinetic energy of the electrons the more delocalised they become and the less they feel
#   the effect of the two Gaussian potentials.
