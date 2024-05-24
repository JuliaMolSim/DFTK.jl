# # Analysing SCF convergence
#
# The goal of this example is to explain the differing convergence behaviour
# of SCF algorithms depending on the choice of the mixing.
# For this we look at the eigenpairs of the Jacobian governing the SCF convergence,
# that is
# ```math
# 1 - α P^{-1} \varepsilon^\dagger \qquad \text{with} \qquad \varepsilon^\dagger = (1-\chi_0 K).
# ```
# where ``α`` is the damping ``P^{-1}`` is the mixing preconditioner
# (e.g. [`KerkerMixing`](@ref), [`LdosMixing`](@ref))
# and $\varepsilon^\dagger$ is the dielectric operator.
#
# We thus investigate the largest and smallest eigenvalues of
# $(P^{-1} \varepsilon^\dagger)$ and $\varepsilon^\dagger$.
# The ratio of largest to smallest eigenvalue of this operator is the condition number
# ```math
# \kappa = \frac{\lambda_\text{max}}{\lambda_\text{min}},
# ```
# which can be related to the rate of convergence of the SCF.
# The smaller the condition number, the faster the convergence.
# For more details on SCF methods, see [Self-consistent field methods](@ref).
#
# For our investigation we consider a crude aluminium setup:

using ASEconvert
using DFTK
using LazyArtifacts
import Main: @artifact_str # hide

ase_Al    = ase.build.bulk("Al"; cubic=true) * pytuple((4, 1, 1))
system_Al = attach_psp(pyconvert(AbstractSystem, ase_Al);
                       Al=artifact"pd_nc_sr_pbe_standard_0.4.1_upf/Al.upf")

# and we discretise:

model_Al = model_LDA(system_Al; temperature=1e-3, symmetries=false)
basis_Al = PlaneWaveBasis(model_Al; Ecut=7, kgrid=[1, 1, 1]);

# On aluminium (a metal) already for moderate system sizes (like the 8 layers
# we consider here) the convergence without mixing / preconditioner is slow:

## Note: DFTK uses the self-adapting LdosMixing() by default, so to truly disable
##       any preconditioning, we need to supply `mixing=SimpleMixing()` explicitly.
scfres_Al = self_consistent_field(basis_Al; tol=1e-12, mixing=SimpleMixing());

# while when using the Kerker preconditioner it is much faster:

scfres_Al = self_consistent_field(basis_Al; tol=1e-12, mixing=KerkerMixing());

# Given this `scfres_Al` we construct functions representing
# $\varepsilon^\dagger$ and $P^{-1}$:

## Function, which applies P^{-1} for the case of KerkerMixing
Pinv_Kerker(δρ) = DFTK.mix_density(KerkerMixing(), basis_Al, δρ)

## Function which applies ε† = 1 - χ0 K
function epsilon(δρ)
    δV   = apply_kernel(basis_Al, δρ; ρ=scfres_Al.ρ)
    χ0δV = apply_χ0(scfres_Al, δV)
    δρ - χ0δV
end

# With these functions available we can now compute the desired eigenvalues.
# For simplicity we only consider the first few largest ones.

using KrylovKit
λ_Simple, X_Simple = eigsolve(epsilon, randn(size(scfres_Al.ρ)), 3, :LM;
                              tol=1e-3, eager=true, verbosity=2)
λ_Simple_max = maximum(real.(λ_Simple))

# The smallest eigenvalue is a bit more tricky to obtain, so we will just assume
λ_Simple_min = 0.952

# This makes the condition number around 30:
cond_Simple = λ_Simple_max / λ_Simple_min

# This does not sound large compared to the condition numbers you might know
# from linear systems.
#
# However, this is sufficient to cause a notable slowdown, which would be even more
# pronounced if we did not use Anderson, since we also would need to drastically
# reduce the damping (try it!).

#  Having computed the eigenvalues of the dielectric matrix
# we can now also look at the eigenmodes, which are responsible for
# the bad convergence behaviour. The largest eigenmode for example:

using Statistics
using Plots
mode_xy = mean(real.(X_Simple[1]), dims=3)[:, :, 1, 1]  # Average along z axis
heatmap(mode_xy', c=:RdBu_11, aspect_ratio=1, grid=false,
        legend=false, clim=(-0.006, 0.006))

# This mode can be physically interpreted as the reason why this SCF converges
# slowly. For example in this case it displays a displacement of electron
# density from the centre to the extremal parts of the unit cell. This
# phenomenon is called charge-sloshing.

# We repeat the exercise for the Kerker-preconditioned dielectric operator:

λ_Kerker, X_Kerker = eigsolve(Pinv_Kerker ∘ epsilon,
                              randn(size(scfres_Al.ρ)), 3, :LM;
                              tol=1e-3, eager=true, verbosity=2)

mode_xy = mean(real.(X_Kerker[1]), dims=3)[:, :, 1, 1]  # Average along z axis
heatmap(mode_xy', c=:RdBu_11, aspect_ratio=1, grid=false,
        legend=false, clim=(-0.006, 0.006))

# Clearly the charge-sloshing mode is no longer dominating.
#
# The largest eigenvalue is now
maximum(real.(λ_Kerker))

# Since the smallest eigenvalue in this case remains of similar size (it is now
# around 0.8), this implies that the conditioning improves noticeably when
# `KerkerMixing` is used.
#
# **Note:** Since LdosMixing requires solving a linear system at each
# application of $P^{-1}$, determining the eigenvalues of
# $P^{-1} \varepsilon^\dagger$ is slightly more expensive and thus not shown. The
# results are similar to `KerkerMixing`, however.

# We could repeat the exercise for an insulating system (e.g. a Helium chain).
# In this case you would notice that the condition number without mixing
# is actually smaller than the condition number with Kerker mixing. In other
# words employing Kerker mixing makes the convergence *worse*. A closer
# investigation of the eigenvalues shows that Kerker mixing reduces the
# smallest eigenvalue of the dielectric operator this time, while keeping
# the largest value unchanged. Overall the conditioning thus workens.

# **Takeaways:**  
# - For metals the conditioning of the dielectric matrix increases steeply with system size.
# - The Kerker preconditioner tames this and makes SCFs on large metallic
#   systems feasible by keeping the condition number of order 1.
# - For insulating systems the best approach is to not use any mixing.
# - **The ideal mixing** strongly depends on the dielectric properties of
#   system which is studied (metal versus insulator versus semiconductor).
