# # Comparing discretization techniques

# In [Periodic problems and plane-wave discretisations](@ref periodic-problems) we saw
# how simple 1D problems can be modelled using plane-wave basis sets. This example
# invites you to work out some details on these aspects yourself using a number of exercises.
# The solutions are given at the bottom of the page.

# For this example we consider the discretization of
# ```math
#    H = - \frac12 Δ + V(x) \quad \text{with $V(x) = \cos(x)$}
# ```
# on $[0, 2π]$ with periodic boundary conditions. The $\cos(x)$ takes the role of
# a lattice-periodic potential. We will be interested in the smallest eigenvalues
# of this discretized Hamiltonian. Of note the boundary condition matters:
# The spectrum we will get is different from e.g. the spectrum of $H$ on $\mathbb{R}$.
#
# ## Finite differences
# We approximate functions $ψ$ on $[0, 2\pi]$ by their values at grid points
# $x_k = 2\pi \frac{k}{N}$, $k=1, \dots, N$.
# The boundary conditions are imposed by $ψ(x_0) = ψ(x_N), ψ(x_{N+1}) = ψ(x_0)$. We then have
# ```math
#    \big(Hψ\big)(x_k) \approx \frac 1 2 \frac{-ψ_{k-1} + 2 ψ_k - ψ_{k+1}}{2 δx^2}
#    + V(x_k) ψ(x_k)
# ```
# with $δx = \frac{2π}{N}$.
#
# This can be put in matrix form in the following way:

## Finite differences Hamiltonian -1/2 Delta + V on [0, 2pi] with periodic bc.
## Pass it a function V.
using LinearAlgebra

function build_finite_differences_matrix(Vfunction, N::Integer)
    δx = 2π/N

    ## Finite-difference approximation to -½Δ
    T = 1/(2δx^2) * Tridiagonal(-ones(N-1), 2ones(N), -ones(N-1))
    ## The type Tridiagonal is efficient, but to establish the periodic boundary conditions
    ## we need to add elements not on the three diagonals, so convert to dense matrix
    T = Matrix(T)
    T[1, N] = T[N, 1] = -1 / (2δx^2)

    ## Finite-difference approximation to potential: We collect all coordinates ...
    x_coords = [k * δx for k=1:N]
    V = Diagonal(Vfunction.(x_coords))  # ... and evaluate V on each of the x_coords

    T + V
end;

# !!! tip "Exercise 1"
#     Show that the finite-difference approximation of -½Δ is indeed an
#     approximation of the second derivative. Obtain an estimate of the first
#     eigenvalue of $H$.  
#     *Hint:* Take a look at the `eigen` function from `LinearAlgebra`.

# ## Plane waves method

# In this method, we expand states on the basis
# ```math
#    e_G(x) = \frac{1}{\sqrt{2\pi}} e^{iGx} \qquad \text{for $G=-N,\dots,N$}.
# ```
#
# !!! tip "Exercise 2"
#     Show that
#     ```math
#        \langle e_G, e_{G'}\rangle = ∫_0^{2π} e_G(x) e_{G'}(x) d x = δ_{G, G'}
#     ```
#     and (assuming $V(x) = \cos(x)$)
#     ```math
#        \langle e_G, H e_{G'}\rangle = \frac 1 2 \left(|G|^2 \delta_{G,G'} + \delta_{G, G'+1} + \delta_{G, G'-1}\right).
#     ```
#     What happens for a more general $V(x)$?
#
# !!! tip "Exercise 3"
#     Code this and check the first eigenvalue agrees
#     with the finite-difference case. Compare accuracies at various basis set sizes $N$.

# ## Using DFTK

# We now use DFTK to do the same plane-wave discretisation in this 1D system.
# To deal with a 1D case we use a 3D lattice with two lattice vectors set to zero.

using DFTK
a = 2π
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];

# Define Hamiltonian: Kinetic + Potential
terms = [Kinetic(),
         ExternalFromReal(r -> cos(r[1]))]  # r is a vector of size 3
model = Model(lattice; n_electrons=1, terms, spin_polarization=:spinless);  # One spinless electron

# Ecut defines the number of plane waves by selecting all those $G$, which satisfy
# the relationship $½ |G|^2 ≤ \text{Ecut}$.
Ecut = 500
basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))

# We now seek the ground state. To better separate the two steps (SCF outer loop
# and diagonalization inner loop), we set the diagtol (the tolerance of the eigensolver)
# to a small value.
diagtolalg = AdaptiveDiagtol(; diagtol_max=1e-8, diagtol_first=1e-8)
scfres = self_consistent_field(basis; tol=1e-4, diagtolalg)
scfres.energies

# On this simple linear (non-interacting) model, the SCF converges in one step.
# The ground state energy of is simply the lowest eigenvalue; it should match
# the smallest eigenvalue of $H$ computed above.

# ## Plotting
# We can also get the first eigenvector (in the plane wave basis) and plot it
using Plots

ψ_fourier = scfres.ψ[1][:, 1];    # first k-point, all G components, first eigenvector
## Transform the wave function to real space
ψ = ifft(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1]
## Eigenvectors are only defined up to a phase. We fix it by imposing that psi(0) is real
ψ /= (ψ[1] / abs(ψ[1]))
plot(real(ψ); label="")

# Again this should match with the result above.
#
# **Exercise 4:** Look at the Fourier coefficients of $\psi$ ($\psi$_fourier)
# and compare with the result above.

# ## The DFTK Hamiltonian
# We can ask DFTK for the Hamiltonian
E, ham = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
H = ham.blocks[1]
typeof(H)

# This is an opaque data structure, which encodes the Hamiltonian. What can we do with it?

using InteractiveUtils
methodswith(typeof(H), supertypes=true)

# This defines a number of methods. For instance, it can be used as a linear operator:

H * DFTK.random_orbitals(basis, basis.kpoints[1], 1)

# We can also get its full matrix representation:

Array(H)

# !!! tip "Exercise 5"
#     Compare this matrix with the one you obtained previously, get its
#     eigenvectors and eigenvalues. Try to guess the ordering of $G$-vectors in DFTK.

# !!! tip "Exercise 6"
#     Increase the size of the problem, and compare the time spent
#     by DFTK's internal diagonalization algorithms to a full diagonalization of `Array(H)`.  
#     *Hint:* The `@time` macro is handy for this task.

# ## Solutions
#
# ### Exercise 1
# If we consider a function $f : [0, 2π] → \mathbb{R}$, to first order
# ```math
# f(x + δx) = f(x) + δx f'(x) + O(δx^2)
# ```
# therefore after rearrangement
# ```math
# f'(x) = \frac{f(x + δx) - f(x)}{δx} + O(δx).
# ```
# Similarly
# ```math
# f''(x) = \frac{f'(x + δx) - f'(x)}{δx} + O(δx),
# ```
# such that overall
# ```math
# f''(x) \simeq \frac{f(x + 2δx) - f(x + δx) - f(x + δx) + f(x)}{δx^2}
#        = \frac{f(x + 2δx) - 2f(x + δx) f(x)}{δx^2}
# ```
# In finite differences we consider a stick basis of vectors
# ```math
# \left\{ e_i = (0, …, 0, \underbrace{δx}_\text{$i$-th position}, 0, …, 0)
#         \middle| i = 1, … N \right\}.
# ```
# Keeping in mind the periodic boundary conditions (i.e. $e_0 = e_N$) projecting the
# Hamiltonian $H$ onto this basis thus yields the proposed structure.
#
# !!! note "TODO More details"
#     More details needed
#
# We start off with $N = 500$ to obtain

Hfd = build_finite_differences_matrix(cos, 100)
L, V = eigen(Hfd)
L[1:5]

# This is already pretty accurate (to about 4 digits) as can be estimated looking at
# the following convergence plot:

using Plots
function fconv(N)
    L, V = eigen(build_finite_differences_matrix(cos, N))
    first(L)
end
Nrange = 10:10:100
plot(Nrange, abs.(fconv.(Nrange) .- fconv(200)); yaxis=:log, legend=false)

# ### Exercise 2
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.
#
# ### Exercise 3
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.
#
# ### Exercise 4
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.
#
# ### Exercise 5
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.
#
# ### Exercise 6
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.

