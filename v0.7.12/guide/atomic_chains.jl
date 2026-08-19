# # Modelling atomic chains
#
# In [Periodic problems and plane-wave discretisations](@ref periodic-problems) we already
# summarised the net effect of Bloch's theorem.
# In this notebook, we will explore some basic facts about periodic systems,
# starting from the very simplest model, a tight-binding monoatomic chain.
# The solutions to the hands-on exercises are given at the bottom of the page.

# ## Monoatomic chain
#
# In this model, each site of an infinite 1D chain is a degree of freedom, and
# the Hilbert space is $\ell^2(\mathbb Z)$, the space of square-summable
# biinfinite sequences $(\psi_n)_{n \in \mathbb Z}$.
#
# Each site interacts by a "hopping term" with its neighbors, and the
# Hamiltonian is
# ```math
# H = \left(\begin{array}{ccccc}
#   \dots&\dots&\dots&\dots&\dots \\
#   \dots& 0 & 1 & 0 & \dots\\
#   \dots&1 & 0 &1&\dots \\
#   \dots&0 & 1 & 0& \dots  \\
#   \dots&\dots&\dots&\dots&…
# \end{array}\right)
# ```
#
# !!! tip "Exercise 1"
#     Find the eigenstates and eigenvalues of this Hamiltonian by
#     solving the second-order recurrence relation.
#
# !!! tip "Exercise 2"
#     Do the same when the system is truncated to a finite number of $N$
#     sites with periodic boundary conditions.
#
# We are now going to code this:

function build_monoatomic_hamiltonian(N::Integer, t)
    H = zeros(N, N)
    for n = 1:N-1
        H[n, n+1] = H[n+1, n] = t
    end
    H[1, N] = H[N, 1] = t  # Periodic boundary conditions
    H
end

# !!! tip "Exercise 3"
#     Compute the eigenvalues and eigenvectors of this Hamiltonian.
#     Plot them, and check whether they agree with theory.

# ## Diatomic chain
# Now we are going to consider a diatomic chain `A B A B ...`, where the coupling
# `A<->B` ($t_1$) is different from the coupling `B<->A` ($t_2$). We will use a new
# index $\alpha$ to denote the `A` and `B` sites, so that wavefunctions are now
# sequences $(\psi_{\alpha n})_{\alpha \in \{1, 2\}, n \in \mathbb Z}$.
#
# !!! tip "Exercise 4"
#     Show that eigenstates of this system can be looked for in the form
#     ```math
#        \psi_{\alpha n} = u_{\alpha} e^{ikn}
#     ```
#
# !!! tip "Exercise 5"
#     Show that, if $\psi$ is of the form above
#     ```math
#        (H \psi)_{\alpha n} = (H_k u)_\alpha e^{ikn},
#     ```
#     where
#     ```
#     H_k = \left(\begin{array}{cc}
#     0                & t_1 + t_2 e^{-ik}\\
#     t_1 + t_2 e^{ik} & 0
#     \end{array}\right)
#     ```
#
# Let's now check all this numerically:

function build_diatomic_hamiltonian(N::Integer, t1, t2)
    ## Build diatomic Hamiltonian with the two couplings
    ## ... <-t2->   A <-t1-> B <-t2->   A <-t1-> B <-t2->   ...
    ## We introduce unit cells as such:
    ## ... <-t2-> | A <-t1-> B <-t2-> | A <-t1-> B <-t2-> | ...
    ## Thus within a cell the A<->B coupling is t1 and across cell boundaries t2

    H = zeros(2, N, 2, N)
    A, B = 1, 2
    for n = 1:N
        H[A, n, B, n] = H[B, n, A, n] = t1  # Coupling within cell
    end
    for n = 1:N-1
        H[B, n, A, n+1] = H[A, n+1, B, n] = t2  # Coupling across cells
    end
    H[A, 1, B, N] = H[B, N, A, 1] = t2  # Periodic BCs (A in cell1 with B in cell N)
    reshape(H, 2N, 2N)
end

function build_diatomic_Hk(k::Integer, t1, t2)
    ## Returns Hk such that H (u e^ikn) = (Hk u) e^ikn
    ##
    ## intra-cell AB hopping of t1, plus inter-cell hopping t2 between
    ## site B (no phase shift) and site A (phase shift e^ik)
    [0                 t1 + t2*exp(-im*k);
     t1 + t2*exp(im*k) 0                 ]
end

using Plots
function plot_wavefunction(ψ)
    p = plot(real(ψ[1:2:end]), label="Re A")
    plot!(p, real(ψ[2:2:end]), label="Re B")
end

# !!! tip "Exercise 6"
#     Check the above assertions. Use a $k$ of the form
#     $2 π \frac{l}{N}$ in order to have a $\psi$ that has the periodicity
#     of the supercell ($N$).

# !!! tip "Exercise 7"
#     Plot the band structure, i.e. the eigenvalues of $H_k$ as a function of $k$
#     Use the function `build_diatomic_Hk` to build the Hamiltonians.
#     Compare with the eigenvalues of the ("supercell") Hamiltonian from
#     `build_diatomic_hamiltonian`. In the case $t_1 = t_2$, how do the bands follow
#     from the previous study of the monoatomic chain?

# !!! tip "Exercise 8"
#     Repeat the above analysis in the case of a finite-difference
#     discretization of a continuous Hamiltonian $H = - \frac 1 2 \Delta + V(x)$
#     where $V$ is periodic
#     *Hint:* It is advisable to work through [Comparing discretization techniques](@ref)
#     before tackling this question.

# ## Solutions
#
# ### Exercise 1
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.
#
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
#
# ### Exercise 7
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.
#
# ### Exercise 8
# !!! note "TODO"
#     This solution has not yet been written. Any help with a PR is appreciated.

