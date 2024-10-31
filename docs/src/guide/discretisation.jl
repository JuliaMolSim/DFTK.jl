# # Comparing discretization techniques

# In [Periodic problems and plane-wave discretizations](@ref periodic-problems) we saw
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
# The boundary conditions are imposed by $ψ(x_0) = ψ(x_N), ψ(x_{N+1}) = ψ(x_1)$. We then have
# ```math
#    \big(Hψ\big)(x_k) \approx \frac 1 2 \frac{-ψ_{k-1} + 2 ψ_k - ψ_{k+1}}{δx^2}
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
#        \langle e_G, e_{G'}\rangle = ∫_0^{2π} e_G^\ast(x) e_{G'}(x) d x = δ_{G, G'}
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

# We now use DFTK to do the same plane-wave discretization in this 1D system.
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

# We now seek the ground state using the self-consistent field algorithm.
scfres = self_consistent_field(basis; tol=1e-4)
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
# !!! tip "Exercise 4"
#     Look at the Fourier coefficients of `ψ_fourier` and compare with the result above.

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
#     Compare this matrix `Array(H)` with the one you obtained in Exercise 3,
#     get its eigenvectors and eigenvalues.
#     Try to guess the ordering of $G$-vectors in DFTK.

# !!! tip "Exercise 6"
#     Increase the size of the problem, and compare the time spent
#     by DFTK's internal diagonalization algorithms to a full diagonalization of `Array(H)`.  
#     *Hint:* The `@belapsed` macro (from the [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) package)
#     is handy for this task. Note that there are some subtleties with global variables
#     (see the BenchmarkTools docs for details). E.g. to use it to benchmark a function
#     like `eigen(H)` run it as
#     ```julia
#     using BenchmarkTools
#     [@benchmark](https://github.com/benchmark) eigen($H)
#     ```
#     (note the `$`).

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
#        = \frac{f(x + 2δx) - 2f(x + δx) + f(x)}{δx^2}
# ```
# In finite differences we consider a stick basis of vectors
# ```math
# \left\{ e_i = (0, …, 0, \underbrace{δx}_\text{$i$-th position}, 0, …, 0)
#         \middle| i = 1, … N \right\}.
# ```
# Keeping in mind the periodic boundary conditions (i.e. $e_0 = e_N$) projecting the
# Hamiltonian $H$ onto this basis thus yields the proposed structure.
#
# We start off with $N = 100$ to obtain

Hfd = build_finite_differences_matrix(cos, 100)
L, V = eigen(Hfd)
L[1:5]

# This is already pretty accurate (to about 4 digits) as can be estimated looking at
# the following convergence plot:

function fconv(N)
    L, V = eigen(build_finite_differences_matrix(cos, N))
    first(L)
end
Nrange = 10:10:100
plot(Nrange, abs.(fconv.(Nrange) .- fconv(200)); yaxis=:log, legend=false)

# ### Exercise 2
# - We note that
#   ```math
#   \langle e_G, e_{G'}\rangle = ∫_0^{2π} e_G^\ast(x) e_{G'}(x) d x = 1/2π ∫_0^{2π} e^{i(G'-G)x} d x
#   ```
#   Since $e^{iy}$ is a periodic function with period $2\pi$, $\int_0^{2\pi} e^{i m y} = \delta_{0,m}$.
#   Therefore if $G≠G'$ we have that $\langle e_G, e_{G'}\rangle = 0$,
#   while $G=G'$ implies $\langle e_G, e_{G'}\rangle = 1$. In summary:
#   ```math
#   \langle e_G, e_{G'}\rangle = δ_{G, G'}
#   ```
# - Next fo $V(x) = \cos(x)$ we obtain
#   ```math
#   \langle e_G, H e_{G'}\rangle = \frac 1 2 ∫_0^{2π} e_G^\ast(x) H e_{G'}(x) d x
#   ```
#   We start by applying the Hamiltonian to a plane-wave:
#   ```math
#   H e_{G'}(x) = - \frac 1 2 (-|G|^2) \frac 1 {\sqrt{2π}} e^{iG'x) + cos(x) \frac 1 {\sqrt{2π}} e{iG'x}
#   ```
#   Then, using the result of the first part of the exercise and the fact that
#   $cos(x) = \frac 1 2 \left(e{ix} + e{-ix})$, we get:
#   ```math
#   \begin{align*}
#   \langle e_G, H e_{G'}\rangle
#   &= \frac 1 2 G^2  δ_{G, G'} + \frac 1 {4π} \left(∫_0^{2π} (e^{ix})^{G'-G+1} d x + ∫_0^{2π} (e^{ix})^{G'-G-1} d x) \\
#   &= \frac 1 2 \left(|G|^2 \delta_{G,G'} + \delta_{G, G'+1} + \delta_{G, G'-1}\right)
#   \end{align*}
#   ```
# - In case a more general $V(x)$ was employed, this potential still has to be periodic
#   over $[0, 2\pi]$ to fit our setting. Assuming sufficient regularity in $V$ we can employ
#   a Fourier series:
#   ```math
#   V(x) = \sum_{G=- \infty}^{\infty} \hat{V}_G e_G(x)
#   ```
#   where
#   ```math
#   \hat{V} = \frac{1}{\sqrt{2π}} ∫_0^{2π} V(x) e^{-iGx} dx = ∫_0^{2π} V(x) e_G^\ast dx .
#   ```
#   Note that one can change of this as a change of basis
#   from the position basis to the plane-wave basis.
#
#   Based on this expansion
#   ```math
#   \begin{align*}
#   ⟨ e_G, V e_{G'} ⟩ &= ⟨ e_G, ∑_{G''} \hat{V}_{G''} e_{G'+G''} ⟩ \\
#   &= ∑_{G''} \hat{V}_{G''} ⟨ e_G, e_{G'+G''} ⟩ \\
#   &= ∑_{G''} \hat{V}_{G''} δ_{G-G', G''} ⟩ \\
#   &= \hat{V}_{G-G'}
#   \end{align*}
#   ```
#   and therefore
#   ```math
#   ⟨ e_G, H e_{G'} ⟩ = \frac 1 2 |G|^2 \delta_{G,G'} + \hat{V}_{G-G'},
#   ```
#   i.e. essentially the Fourier transform of $V$ determines the contribution
#   to the matrix elements of the Hamiltonian.
#
#
# ### Exercise 3
# The Hamiltonian matrix for the plane waves method can be found this way:

## Plane waves Hamiltonian -½Δ + cos on [0, 2pi].
function build_plane_waves_matrix_cos(N::Integer)
    ## Plane wave approximation to -½Δ
    Gsq = [float(i)^2 for i in -N:N]
    ## Hamiltonian as derived in Exercise 2:
    1/2 * Tridiagonal(ones(2N), Gsq, ones(2N))
end

# Then we check that the first eigenvalue agrees with the finite-difference case, using $N = 10$:

Hpw_cos = build_plane_waves_matrix_cos(10)
L, V = eigen(Hpw_cos)
L[1:5]

# We look at the convergence plot to compare the accuracy for various numbers of plane-waves $N$:

function fconv(N)
    L = eigvals(build_plane_waves_matrix_cos(N))
    first(L)
end

Nrange = 2:10
plot(Nrange, abs.(fconv.(Nrange) .- fconv(200)); yaxis=:log, legend=false,
     ylims=(1e-16,Inf), ylabel="Absolute error", xlabel="N")

# Notice how compared to exercise 1 the considered basis size $n$ is much smaller,
# indicating that plane-wave methods more quickly lead to accurate solutions than
# finite-difference methods.


# ### Exercise 4
# For efficiency reasons the data in Fourier space is not ordered increasingly with $G$.
# Therefore to plot the Fourier space representation sensibly, we need to sort by ascending
# values of the $G$ vectors first. For this we extract the Fourier vector of each plane-wave
# basis function in the index order:

coords_G_vectors = G_vectors_cart(basis, basis.kpoints[1])  # Get coordinates of first and only k-point

## Only keep first component of each vector (because the others are zero for 1D problems):
coords_Gx = [G[1] for G in coords_G_vectors]

p = plot(coords_Gx, real(ψ_fourier); label="real part")
plot!(p, coords_Gx, imag(ψ_fourier); label="imaginary part")

# The plot is symmetric about the zero (confirming that the orbitals are real)
# and only takes peaked values, which corresponds
# to the expected result for a cosine potential.
#
# ### Exercise 5
# To figure out the ordering we consider a small basis and build the Hamiltonian:

basis_small  = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
ham_small = Hamiltonian(basis_small)
H_small = Array(ham_small.blocks[1])
H_small[abs.(H_small) .< 1e-12] .= 0  # Drop numerically zero entries

# The equivalent version using the `build_plane_waves_matrix_cos` function
# is `N=3` (both give rice to a 7×7 matrix).
Hother = build_plane_waves_matrix_cos(3)

# By comparing the entries we find the ordering is 0,1,2,...,-2,-1,
# which can also be found by inspecting
first.(G_vectors(basis_small, basis_small.kpoints[1]))

# Both matrices have the same eigenvalues:

maximum(abs, eigvals(H_small) - eigvals(Hother))

# and in the eigenvectors we find the same patterns:

eigvecs(Hother)[:, 1]
#-
eigvecs(H_small)[:, 1]


# ### Exercise 6
#
# We benchmark the time needed for a full diagonalization (instantiation of the Array
# plus call of `eigen`) versus the time needed for running the SCF (i.e. iterative
# diagonalization using plane waves).

using Printf

for Ecut in 200:200:1600
   basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
   t_eigen = @elapsed eigen(Array(Hamiltonian(basis).blocks[1]))
   t_scf   = @elapsed self_consistent_field(basis; tol=1e-6, callback=identity);
   @printf "%4i  eigen=%8.6f  scf=%8.6f\n" Ecut 1000t_eigen 1000t_scf
end
