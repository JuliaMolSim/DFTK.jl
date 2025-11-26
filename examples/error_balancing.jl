# # Error splitting between SCF and discretization contributions
#
# This is an example showing how to compute and use the bounds derived in
# [^BCDKLS2025]. The bounds can be split into two contributions, one coming from
# the SCF iterations (``\eta_{\rm SCF}``) and the other coming from the
# plane-wave basis approximation (``\eta_{\rm disc}``). While Theorem 3.7 from
# this reference is only valid for convex xc functionals, the bounds can be
# computed for general functionals and still yield satisfactory results, mainly
# because the nonconvex terms are of higher order. The discretization error is
# evaluated using the zero-th order from this paper (see Table 1). Note that it
# only works for zero temperature systems at the moment.
#
# [^BCDKLS2025]:
#     A. Bordignon, E. Cancès, G. Dusson, G. Kemlin, R.A. Lainez Reyes and B. Stamm
#     *Fully guaranteed and computable error bounds on the energy for periodic Kohn-Sham equations with convex density functionals*
#     [SIAM Journal on Scientific Computing 47 (5), A2881-A2905](https://doi.org/10.1137/25M1735676)

using DFTK
using PseudoPotentialData
using Plots
using LinearAlgebra
using Statistics

# ## Setup
# We use a standard Silicon crystal with LDA exchange-correlation and zero
# temperature.
a = 10.26;  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]];
Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"));
atoms     = [Si, Si];
positions = [ones(3)/8, -ones(3)/8];
model = model_DFT(lattice, atoms, positions; functionals=LDA());
kgrid = [2,2,2];
tol = 1e-12;

# ## Reference calculation
# We run a (expensive) reference calculation for illustration purpose.
# Note that `basis_ref` will also be used to compute the residuals norms in the
# discretization error bound.
Ecut_ref = 100;
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid);
scfres_ref = self_consistent_field(basis_ref; tol);
energies_ref = scfres_ref.energies;

# ## SCF calculation
#
# ### Setup of the error estimation framework
# We start by creating a new structure to trigger convergence once the SCF
# contribution is smaller than the discretization contribution.
mutable struct ScfAdaptiveEnergy
    basis_ref        # reference basis to compute the residuals norms
    reduction_factor # reduction factor to stop SCF when error_SCF < reduction_factor * error_disc is reached
    ψ_prev           # previous state (required to compute the SCF contribution)
    error_disc       # discretization contribution
    error_scf        # SCF contribution
end
ScfAdaptiveEnergy(basis_ref, reduction_factor) = ScfAdaptiveEnergy(basis_ref, reduction_factor, nothing, NaN, NaN);

# Function to be passed to the convergence test.
# There is an additional approximation here since `ρin` is _not_ the density
# of `ψ_prev` (see `self_consistent_field.jl`). Still we use it to avoid an
# expensive additional diagonalisation of `info.ham` and the results are
# satisfying enough. We refer to [^BCDKLS2025] for the explicit formulas that
# are used here.
function compute_error_balancing(info, basis_ref, ψ_prev)
    basis      = info.basis

    ψ, occ     = DFTK.select_occupied_orbitals(basis, info.ψ, info.occupation) # select occupied orbitals (finite temperature not supported yet)
    ψ_prev, _  = DFTK.select_occupied_orbitals(basis, ψ_prev, info.occupation)
    ρ_prev     = info.ρin
    ham        = info.ham

    ψ_ref      = transfer_blochwave(ψ, basis, basis_ref) # _ref means it lives on the reference basis
    ψ_prev_ref = transfer_blochwave(ψ_prev, basis, basis_ref)
    ρ_prev_ref = DFTK.transfer_density(info.ρin, basis, basis_ref)
    _, ham_ref = energy_hamiltonian(basis_ref, ψ_prev_ref, occ; ρ=ρ_prev_ref)

    η_scf = map(enumerate(ψ)) do (ik, ψk) # SCF contribution
        ψk_prev = ψ_prev[ik]
        Hk = ham[ik]
        Ok = Diagonal(occ[ik])
        real(dot(ψk_prev*Ok, Hk*ψk_prev)) - real(dot(ψk*Ok, Hk*ψk))
    end
    error_scf = DFTK.weighted_ksum(basis, η_scf)

    residuals = map(enumerate(ψ_ref)) do (ik, ψk_ref) # compute the residuals in the reference basis
        Mk = length(occ[ik])
        ham_ref[ik] * ψk_ref - ψk_ref * Diagonal(info.eigenvalues[ik][1:Mk])
    end
    residuals_lf = DFTK.transfer_blochwave(residuals, basis_ref, basis)
    residuals    = residuals - DFTK.transfer_blochwave(residuals_lf, basis, basis_ref)

    avg_local_pot = mean(DFTK.total_local_potential(ham_ref)) # Average of the effective potential
    V_avg = map(enumerate(basis_ref.kpoints)) do (ik, kpt)
        non_local_op = [op for op in ham_ref[ik].operators if (op isa DFTK.NonlocalOperator)]
        if !isempty(non_local_op)
            avg_non_local_op = [real(dot(p, non_local_op[1].D * p)) for p in eachrow(non_local_op[1].P)]
            avg_local_pot .+ avg_non_local_op
        else
            M = length(basis_ref.terms[1].kinetic_energies[ik])
            avg_local_pot .+ zeros(M)
        end
    end

    η2 = map(enumerate(residuals)) do (ik, rk)
        kpt_ref = basis_ref.kpoints[ik]
        Mk      = length(occ[ik])
        eigk    = info.eigenvalues[ik]
        sk = -eigk[1] # Shift for the Hamiltonian to be > 0.
        sk = sk > 0 ? 1.1*sk : 0.9*sk

        hrk = similar(rk)
        Pk  = PreconditionerTPA(basis_ref, kpt_ref; default_shift=0)
        Pk.mean_kin = V_avg[ik]
        for i = 1:size(rk, 2) # Invert the (diagonal) kinetic energy in high frequencies.
            hrk[:,i] .= rk[:,i] ./ (Pk.kin .+ Pk.mean_kin .+ sk) # don't forget the shift
        end

        CNk = 4 * (eigk[Mk]+sk) * (eigk[Mk+1]+sk)^2 / (eigk[Mk+1]-eigk[Mk])^2 # Compute the gap constant (don't forget the shift).

        real(sum(i -> dot(rk[:,i], hrk[:,i]) + CNk * dot(hrk[:,i], hrk[:,i]), 1:Mk)) # Final zeroth order approximation.
    end

    η_disc = map(enumerate(ψ)) do (ik, ψk) # Finally compute discretization error.
        M  = length(occ[ik])
        μk = (sum(info.eigenvalues[ik][1:M]) - η2[ik]) / M
        Ok = Diagonal(occ[ik])
        Hk = ham[ik]
        real(dot(ψk*Ok, Hk*ψk) - μk * dot(ψk*Ok, ψk))
    end
    error_disc = DFTK.weighted_ksum(basis, η_disc)

    error_scf, error_disc # return both contributions
end;

# Next, we define a method to be passed to the `self_consistent_field` function
# to trigger convergence.
function (conv::ScfAdaptiveEnergy)(info)
    if info.n_iter > 1
        error_scf, error_disc = compute_error_balancing(info, conv.basis_ref, conv.ψ_prev)
        conv.error_scf  = error_scf
        conv.error_disc = error_disc
    end
    conv.ψ_prev = info.ψ
    conv.error_scf < conv.reduction_factor*conv.error_disc
end

# We also set up a callback that uses the previous structure to monitor the
# error estimates along the SCF iterations.
p = plot(; yaxis=:log)
η_scf  = Float64[];
η_disc = Float64[];
error  = Float64[];
function plot_callback(info)
    if info.stage == :finalize
        plot!(p, η_scf, label="η_scf", markershape=:x, linestyle=:dash)
        plot!(p, η_disc, label="η_disc", markershape=:+, linestyle=:dash)
        plot!(p, η_disc .+ η_scf, label="full bound : η_scf+η_disc", markershape=:square)
        plot!(p, error, label="true energy error", markershape=:circle)
        xlabel!(p, "SCF iteration")
    elseif info.n_iter > 1
        push!(η_scf, abs(is_converged.error_scf))
        push!(η_disc, abs(is_converged.error_disc))
        push!(error, abs(info.energies.total - energies_ref.total))
    end
    info
end
callback = ScfDefaultCallback() ∘ plot_callback;

# We now run the calculation with the custom callback and convergence
# criterion...
Ecut = 25;
basis = PlaneWaveBasis(model; Ecut, kgrid);
is_converged = ScfAdaptiveEnergy(basis_ref, 1e-3);
scfres = self_consistent_field(basis; tol, callback, is_converged);

# ... and show the plot. Notice that, even though not guaranteed, the total
# bound is still an accurate estimation of the actual error on the energy. In
# particular, the transition from a regime where the SCF contribution dominates
# to a regime where the discretization error dominates clearly appears.
p
