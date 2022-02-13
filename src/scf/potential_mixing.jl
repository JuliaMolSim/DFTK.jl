using Statistics

# Quick and dirty Anderson implementation ... lacks important things like
# control of the condition number of the anderson matrix.
# Not particularly optimised. Also should be moved to NLSolve ...
#
# Accelerates the iterative solution of f(x) = 0 according to a
# damped preconditioned scheme
#    xₙ₊₁ = xₙ + αₙ P⁻¹ f(xₙ)
# Where f(x) computes the residual (e.g. SCF(x) - x)
# Further define
#    preconditioned residual  Pf(x) = P⁻¹ f(x)
#    fixed-point map          g(x)  = V + α Pf(x)
# where the α may vary between steps.
#
# Finds the linear combination xₙ₊₁ = g(xₙ) + ∑ᵢ βᵢ (g(xᵢ) - g(xₙ))
# such that |Pf(xₙ) + ∑ᵢ βᵢ (Pf(xᵢ) - Pf(xₙ))|² is minimal
struct AndersonAcceleration
    m::Int                  # maximal history size
    iterates::Vector{Any}   # xₙ
    residuals::Vector{Any}  # Pf(xₙ)
    maxcond::Real           # Maximal condition number for Anderson matrix
end
AndersonAcceleration(;m=10, maxcond=1e6) = AndersonAcceleration(m, [], [], maxcond)

function Base.push!(anderson::AndersonAcceleration, xₙ, αₙ, Pfxₙ)
    push!(anderson.iterates,  vec(xₙ))
    push!(anderson.residuals, vec(Pfxₙ))
    if length(anderson.iterates) > anderson.m
        popfirst!(anderson.iterates)
        popfirst!(anderson.residuals)
    end
    @assert length(anderson.iterates) <= anderson.m
    @assert length(anderson.iterates) == length(anderson.residuals)
    anderson
end

# Gets the current xₙ, Pf(xₙ) and damping αₙ
function (anderson::AndersonAcceleration)(xₙ, αₙ, Pfxₙ)
    xs   = anderson.iterates
    Pfxs = anderson.residuals

    # Special cases with fast exit
    anderson.m == 0 && return xₙ .+ αₙ .* Pfxₙ
    if isempty(xs)
        push!(anderson, xₙ, αₙ, Pfxₙ)
        return xₙ .+ αₙ .* Pfxₙ
    end

    M = hcat(Pfxs...) .- vec(Pfxₙ)  # Mᵢⱼ = (Pfxⱼ)ᵢ - (Pfxₙ)ᵢ
    # We need to solve 0 = M' Pfxₙ + M'M βs <=> βs = - (M'M)⁻¹ M' Pfxₙ

    # Ensure the condition number of M stays below maxcond, else prune the history
    Mfac = qr(M)
    while size(M, 2) > 1 && cond(Mfac.R) > anderson.maxcond
        M = M[:, 2:end]  # Drop oldest entry in history
        popfirst!(anderson.iterates)
        popfirst!(anderson.residuals)
        Mfac = qr(M)
    end

    xₙ₊₁ = vec(xₙ) .+ αₙ .* vec(Pfxₙ)
    βs   = -(Mfac \ vec(Pfxₙ))
    for (iβ, β) in enumerate(βs)
        xₙ₊₁ .+= β .* (xs[iβ] .- vec(xₙ) .+ αₙ .* (Pfxs[iβ] .- vec(Pfxₙ)))
    end

    push!(anderson, xₙ, αₙ, Pfxₙ)
    reshape(xₙ₊₁, size(xₙ))
end

function ScfAcceptStepAll()
    accept_step(info, info_next) = true
end

"""
Accept a step if the energy is at most increasing by `max_energy` and the residual
is at most `max_relative_residual` times the residual in the previous step.
"""
function ScfAcceptImprovingStep(;max_energy_change=1e-12, max_relative_residual=1.0)
    function accept_step(info, info_next)
        energy_change = info_next.energies.total - info.energies.total
        relative_residual = norm(info_next.Pinv_δV) / norm(info.Pinv_δV)

        # Accept if energy goes down or residual decreases
        accept = energy_change < max_energy_change || relative_residual < max_relative_residual
        mpi_master() && @debug "Step $(accept ? "accepted" : "discarded")" energy_change relative_residual
        accept
    end
end

"""
Use the two iteration states `info` and `info_next` to find a damping value
from a quadratic model for the SCF energy. Returns `nothing` if the constructed
model is not considered trustworthy, else returns the suggested damping.
"""
function scf_damping_quadratic_model(info, info_next; modeltol=0.1)
    T     = eltype(info.Vin)
    dvol  = info.basis.dvol

    Vin   = info.Vin
    ρin   = info.ρout      # = ρ(Vin)
    Vout  = info.Vout      # = step(Vin), where step(V) = (Vext + Vhxc(ρ(V)))
    α0    = info_next.α
    Vnext = info_next.Vin  # = Vin + α0 * (Anderson(Vin, P⁻¹( Vout - Vin )) - Vin)
    ρnext = info_next.ρout # = ρ(Vnext)
    δρ    = ρnext - ρin
    # α0 * δV = α0 * (Vnext - Vin) = α0 * (Anderson(Vin, P⁻¹( Vout - Vin )) - Vin)

    # We build a quadratic model for
    #   ϕ(α) = E(Vin  + α δV)  at α = 0
    #        = E(Vin) + α ∇E|_(V=Vin) ⋅ δV + ½ α^2 <δV | ∇²E|_(V=Vin) | δV>
    #        = E(Vin) + (α/α0) ∇E|_(V=Vin) ⋅ (α0 * δV) + ½ (α/α0)² <α0 * δV | ∇²E|_(V=Vin) | α0 * δV>
    #
    # Now
    #      ∇E|_(V=Vin)  = - χ₀(Vout - Vin)
    #      ∇²E|_(V=Vin) ≃ - χ₀ (1 - K χ₀)        (only true if Vin is an SCF minimum, K taken at Vin)
    # and therefore using the self-adjointness of χ₀
    #      ∇E|_(V=Vin) ⋅ δV         = -(Vout - Vin) ⋅ χ₀(δV) = - (Vout - Vin) ⋅ δρ
    #      <δV | ∇²E|_(V=Vin) | δV> = - δV ⋅ δρ + δρ ⋅ K(δρ)
    #
    slope = dot(Vout .- Vin, δρ) / α0 * dvol
    Kδρ   = apply_kernel(info.basis, δρ; ρ=ρin)
    curv  = dvol * (-dot(Vnext .- Vin, δρ) + dot(δρ, Kδρ)) / α0^2
    Emodel(α) = info.energies.total + slope * α + curv * α^2 / 2

    # Relative error of the model at α0 (the damping we used to get info_next)
    Etotal_next = info_next.energies.total

    # TODO Is this a good error measure ... for larger problems it seems to be over-demanding
    model_relerror = abs(Etotal_next - Emodel(α0)) / abs(Etotal_next - info.energies.total)

    minimum_exists = curv > eps(T)  # Does the model predict a minimum along the search direction
    trusted_model  = model_relerror < modeltol      # Model fits observation
    tight_model    = model_relerror < modeltol / 5  # Model fits observation very well

    # Accept model if it leads to minimum and is either tight or shows a negative slope
    α_model = -slope / curv
    if minimum_exists && (tight_model || (slope < -eps(T) && trusted_model))
        mpi_master() && @debug "Quadratic model accepted" model_relerror slope curv α_model
        (α=α_model, relerror=model_relerror)
    else
        mpi_master() && @debug "Quadratic model discarded" model_relerror slope curv α_model
        (α=nothing, relerror=model_relerror) # Model not trustworthy ...
    end
end

# Adaptive damping using a quadratic model
@kwdef struct AdaptiveDamping
    α_min = 0.05        # Minimal damping
    α_max = 1.0         # Maximal damping
    α_trial_init = 0.8  # Initial trial damping used (i.e. in the first SCF step)
    α_trial_min = 0.2   # Minimal trial damping used in a step
    α_trial_enhancement = 1.1  # Enhancement factor to α_trial in case a step is immediately successful
    modeltol = 0.1      # Maximum relative error on the predicted energy for model
    #                     to be considered trustworthy
end
function AdaptiveDamping(α_trial_min; kwargs...)
    # Select some reasonable defaults.
    # The free tweaking parameter here should be increased a bit for cases,
    # where the Anderson does weird stuff in case of too small damping.
    AdaptiveDamping(;α_min=α_trial_min / 4,
                     α_max=max(1.25α_trial_min, 1.0),
                     α_trial_init=max(α_trial_min, 0.8),
                     α_trial_min=α_trial_min,
                     kwargs...)
end

function ensure_damping_within_range(damping::AdaptiveDamping, α, α_next)
    α_sign = sign(α_next)
    abs(α_next) ≤ damping.α_min / 5 && (α_sign = +1.0)
    if α_sign > 0.0
        # Avoid getting stuck
        α_next = min(0.95abs(α), abs(α_next))
    else
        # Don't move too far backwards (where model validity cannot be ensured)
        α_next = min(0.50abs(α), abs(α_next))
    end
    α_next = clamp(α_next, damping.α_min, damping.α_max)
    α_sign * α_next
end

function propose_backtrack_damping(damping::AdaptiveDamping, info, info_next)
    if abs(info_next.α) < 1.75damping.α_min
        # Too close to α_min to be worth making another step ... just give up
        return info_next.α
    end

    α_next, relerror = scf_damping_quadratic_model(info, info_next; modeltol=damping.modeltol)
    if isnothing(α_next)
        # Model failed ... use heuristics: Half for small model error, else use a quarter
        α_next = info_next.α / (relerror < 10 ? 2 : 4)
    end
    ensure_damping_within_range(damping, info_next.α, α_next)
end

trial_damping(damping::AdaptiveDamping) = damping.α_trial_init
function trial_damping(damping::AdaptiveDamping, info, info_next, step_successful)
    n_backtrack = length(info_next.diagonalization)

    α_trial = abs(info_next.α)  # By default use the α that worked in this step
    if step_successful && n_backtrack == 1  # First step was good => speed things up
        α_trial ≥ damping.α_max && return damping.α_max  # No need to compute model
        α_model = scf_damping_quadratic_model(info, info_next; modeltol=damping.modeltol).α
        if !isnothing(α_model)  # Model is meaningful
            α_trial = max(damping.α_trial_enhancement * abs(α_model), α_trial)
        end
    end

    # Ensure returned damping is in valid range
    clamp(α_trial, damping.α_trial_min, damping.α_max)
end

struct FixedDamping
    α
end
FixedDamping() = FixedDamping(0.8)
propose_backtrack_damping(damping::FixedDamping) = damping.α
trial_damping(damping::FixedDamping, args...) = damping.α


# Notice: For adaptive damping to run smoothly, multiple defaults need to be changed.
#         See the function scf_potential_mixing_adaptive for that use case.
@timing function scf_potential_mixing(
    basis::PlaneWaveBasis;
    damping=FixedDamping(0.8),
    n_bands=default_n_bands(basis.model),
    ρ=guess_density(basis),
    V=nothing,
    ψ=nothing,
    tol=1e-6,
    maxiter=100,
    eigensolver=lobpcg_hyper,
    n_ep_extra=3,
    diag_miniter=1,
    determine_diagtol=ScfDiagtol(),
    mixing=SimpleMixing(),
    is_converged=ScfConvergenceEnergy(tol),
    callback=ScfDefaultCallback(),
    acceleration=AndersonAcceleration(;m=10),
    accept_step=ScfAcceptStepAll(),
    max_backtracks=3,  # Maximal number of backtracking line searches
)
    damping isa Number && (damping = FixedDamping(damping))

    if ψ !== nothing
        @assert length(ψ) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end

    # Initial guess for V (if none given)
    energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ)
    isnothing(V) && (V = total_local_potential(ham))

    function EVρ(Vin; diagtol=tol / 10, ψ=nothing)
        ham_V = hamiltonian_with_total_potential(ham, Vin)
        res_V = next_density(ham_V; n_bands=n_bands, ψ=ψ, n_ep_extra=n_ep_extra,
                             miniter=diag_miniter, tol=diagtol, eigensolver=eigensolver)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρout, eigenvalues=res_V.eigenvalues,
                                            εF=res_V.εF)
        (basis=basis, ham=new_ham, energies=new_E, Vin=Vin,
         Vout=total_local_potential(new_ham), res_V...)
    end

    n_iter    = 1
    converged = false
    α_trial   = trial_damping(damping)
    diagtol   = determine_diagtol((ρin=ρ, Vin=V, n_iter=n_iter))
    info      = EVρ(V; diagtol=diagtol, ψ=ψ)
    Pinv_δV   = mix_potential(mixing, basis, info.Vout - info.Vin;
                              ρin=ρ, n_iter, n_ep_extra, info...)
    info      = merge(info, (α=NaN, diagonalization=[info.diagonalization], ρin=ρ,
                             n_iter=n_iter, Pinv_δV=Pinv_δV))
    ΔEdown    = 0.0

    while n_iter < maxiter
        info = merge(info, (stage=:iterate, algorithm="SCF", converged=converged))
        callback(info)
        if MPI.bcast(is_converged(info), 0, MPI.COMM_WORLD)
            # TODO Debug why these MPI broadcasts are needed
            converged = true
            break
        end
        n_iter += 1
        info = merge(info, (n_iter=n_iter, ))

        # Ensure same α on all processors
        α_trial = MPI.bcast(α_trial, 0, MPI.COMM_WORLD)
        δV = (acceleration(info.Vin, α_trial, info.Pinv_δV) - info.Vin) / α_trial

        # Determine damping and take next step
        guess   = ψ
        α       = α_trial
        successful  = false  # Successful line search (final step is considered good)
        n_backtrack = 1
        diagonalization = empty(info.diagonalization)
        info_next = info
        while n_backtrack ≤ max_backtracks
            diagtol = determine_diagtol(info_next)
            mpi_master() && @debug "Iteration $n_iter linesearch step $n_backtrack   α=$α diagtol=$diagtol"
            Vnext = info.Vin .+ α .* δV

            info_next    = EVρ(Vnext; ψ=guess, diagtol=diagtol)
            Pinv_δV_next = mix_potential(mixing, basis, info_next.Vout - info_next.Vin;
                                         ρin=info.ρout, n_iter, n_ep_extra, info_next...)
            push!(diagonalization, info_next.diagonalization)
            info_next = merge(info_next, (α=α, diagonalization=diagonalization,
                                          ρin=info.ρout, n_iter=n_iter,
                                          Pinv_δV=Pinv_δV_next))

            successful = accept_step(info, info_next)
            successful = MPI.bcast(successful, 0, MPI.COMM_WORLD)  # Ensure same successful
            if successful || n_backtrack ≥ max_backtracks
                break
            end
            n_backtrack += 1

            # Adjust α to try again ...
            α_next = propose_backtrack_damping(damping, info, info_next)
            α_next = MPI.bcast(α_next, 0, MPI.COMM_WORLD)  # Ensure same α on all processors
            if α_next == α  # Backtracking further not useful ...
                break
            end

            # Adjust to guess fitting α best:
            guess = α_next > α / 2 ? info_next.ψ : ψ
            α = α_next
        end

        # Switch off acceleration in case of very bad steps
        ΔE = info_next.energies.total - info.energies.total
        ΔE < 0 && (ΔEdown = -max(abs(ΔE), tol))

        # Update α_trial and commit the next state
        α_trial = trial_damping(damping, info, info_next, successful)
        info = info_next
    end

    ham  = hamiltonian_with_total_potential(ham, info.Vout)
    info = (ham=ham, basis=basis, energies=info.energies, converged=converged,
            ρ=info.ρout, eigenvalues=info.eigenvalues, occupation=info.occupation,
            εF=info.εF, n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=info.ψ,
            diagonalization=info.diagonalization, stage=:finalize, algorithm="SCF")
    callback(info)
    info
end


# Wrapper function setting a few good defaults for adaptive damping
function scf_potential_mixing_adaptive(basis; tol=1e-6, damping=AdaptiveDamping(), kwargs...)
    @assert damping isa AdaptiveDamping
    scf_potential_mixing(basis; tol=tol, diag_miniter=2,
                         accept_step=ScfAcceptImprovingStep(max_energy_change=tol),
                         determine_diagtol=ScfDiagtol(ratio_ρdiff=0.03, diagtol_max=5e-3),
                         damping=damping, kwargs...)
end
