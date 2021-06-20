using Statistics

# Quick and dirty Anderson implementation ... lacks important things like
# control of the condition number of the anderson matrix.
# Not particularly optimised. Also should be moved to NLSolve ...
function AndersonAcceleration(;m=Inf)
    # Accelerates the iterative solution of f(V) = 0, where for our DFT case:
    #    f(V) = Vext + Vhxc(ρ(V)) - V
    # Further define
    #    preconditioned update    Pf(V) = P⁻¹ f(V)
    #    fixed-point map          g(V)  = V + α Pf(V)
    # where the α may vary between steps.
    #
    # Finds the linear combination Vₙ₊₁ = g(Vₙ) + ∑ᵢ βᵢ (g(Vᵢ) - g(Vₙ))
    # such that |Pf(Vₙ) + ∑ᵢ βᵢ (Pf(Vᵢ) - Pf(Vₙ))|² is minimal
    #
    Vs   = []  # The V     for each iteration
    PfVs = []  # The Pf(V) for each iteration

    function extrapolate(Vₙ, αₙ, PfVₙ)
        m == 0 && return Vₙ .+ αₙ .* PfVₙ

        # Gets the current Vₙ, Pf(Vₙ) and damping αₙ
        #
        Vₙ₊₁ = vec(Vₙ) .+ αₙ .* vec(PfVₙ)
        if !isempty(Vs)
            M = hcat(PfVs...) .- vec(PfVₙ)  # Mᵢⱼ = (PfVⱼ)ᵢ - (PfVₙ)ᵢ
            # We need to solve 0 = M' PfVₙ + M'M βs <=> βs = - (M'M)⁻¹ M' PfVₙ
            βs = -M \ vec(PfVₙ)
            for (iβ, β) in enumerate(βs)
                Vₙ₊₁ .+= β .* (Vs[iβ] .- vec(Vₙ) .+ αₙ .* (PfVs[iβ] .- vec(PfVₙ)))
            end
        end

        push!(Vs,   vec(Vₙ))
        push!(PfVs, vec(PfVₙ))
        if length(Vs) > m
            Vs   = Vs[2:end]
            PfVs = PfVs[2:end]
        end
        @assert length(Vs) <= m

        reshape(Vₙ₊₁, size(Vₙ))
    end
end


"""
Accept a step if the energy is at most increasing by `max_energy` and the residual
is at most `max_relative_residual` times the residual in the previous step.
"""
function ScfAcceptImprovingStep(;max_energy_change=1e-12, max_relative_residual=0.9)
    function accept_step(info, info_next)
        energy_change = info_next.energies.total - info.energies.total
        relative_residual = norm(info_next.Pinv_δV) / norm(info.Pinv_δV)

        # Accept if energy goes down or residual decreases
        accept = energy_change < max_energy_change || relative_residual < max_relative_residual
        mpi_master() && @debug "Step $(accept ? "accepted" : "discarded")" energy_change relative_residual
        accept
    end
end
function ScfAcceptStepAll()
    accept_step(info, info_next) = true
end


function scf_quadratic_model(info, info_next; modeltol=0.1)
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
    model_relerror = abs(Etotal_next - Emodel(α0)) / abs(Etotal_next - info.energies.total)

    minimum_exists = curv > eps(T)  # Does the model predict a minimum along the search direction
    trusted_model  = model_relerror < modeltol      # Model fits observation
    tight_model    = model_relerror < modeltol / 5  # Model fits observation very well

    # Accept model if it leads to minimum and is either tight or shows a negative slope
    if minimum_exists && (tight_model || (slope < -eps(T) && trusted_model))
        α_model = -slope / curv
        mpi_master() && @debug "Quadratic model accepted" model_relerror slope curv α_model
        return α_model
    else
        mpi_master() && @debug "Quadratic model discarded" model_relerror slope curv
        nothing  # Model not trustworthy ... better return nothing
    end
end

# Adaptive damping using a quadratic model
@kwdef struct AdaptiveDamping
    α_init = 0.5       # The initial damping value (used as first trial value in first SCF step)
    α_min = 0.01       # Minimal damping
    α_max = 1.0        # Maximal damping
    α_trial_min = 0.1  # Minimal first trial damping used in a step
    α_trial_enhancement = 1.1  # Enhancement factor to α_trial in case a step is immediately successful
    modeltol = 0.1     # Maximum relative error for model to be considered trustworthy
end

function propose_backtrack_damping(damping::AdaptiveDamping, info, info_next)
    if abs(info_next.α) < 1.75damping.α_min
        # Too close to α_min to be worth making another step ... just give up
        return info_next.α
    end

    α = scf_quadratic_model(info, info_next; modeltol=damping.modeltol)
    isnothing(α) && (α = info_next.α / 2)  # Model failed ... use heuristics

    # Adjust α to stay within desired range
    α_sign = sign(α)
    α = clamp(abs(α), damping.α_min, damping.α_max)
    α = min(0.95abs(info_next.α), α)  # Avoid to get stuck
    return α_sign * α
end

initial_damping(damping::AdaptiveDamping) = damping.α_init

function next_trial_damping(damping::AdaptiveDamping, info, info_next, step_successful)
    n_backtrack = length(info_next.diagonalization)

    α_trial = abs(info_next.α)  # By default use the α that worked in this step
    if step_successful && n_backtrack == 1
        # First step was directly ok => Allow to accelerate a little
        α_model = scf_quadratic_model(info, info_next; modeltol=damping.modeltol)
        if !isnothing(α_model)  # Model is meaningful
            α_trial = max(damping.α_trial_enhancement * abs(α_model), α_trial)
        end
    end

    # Ensure the range stays valid
    α_trial = clamp(α_trial, damping.α_trial_min, damping.α_max)
end

struct FixedDamping
    α
end
FixedDamping() = FixedDamping(0.8)
propose_backtrack_damping(damping::FixedDamping) = damping.α
initial_damping(damping::FixedDamping) = damping.α
next_trial_damping(damping::FixedDamping, info, info_next, successful) = damping.α


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
    ratio_failure_accel_off=Inf,  # Acceleration never switched off
    α_accel_min=0.0,  # Minimal damping passed to an accelerator (e.g. Anderson)
                      # Increasing this a bit for adaptive damping speeds up convergence
    accept_step=ScfAcceptStepAll(),
    max_backtracks=3,  # Maximal number of backtracking line searches
)
    # TODO Test other mixings and lift this
    @assert (   mixing isa SimpleMixing
             || mixing isa KerkerMixing
             || mixing isa KerkerDosMixing)
    damping isa Number && (damping = FixedDamping(damping))

    # Initial guess for V and ψ (if none given)
    if ψ !== nothing
        @assert length(ψ) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end
    energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ)
    isnothing(V) && (V = total_local_potential(ham))

    function EVρ(Vin; diagtol=tol / 10, ψ=nothing)
        ham_V = hamiltonian_with_total_potential(ham, Vin)
        res_V = next_density(ham_V; n_bands=n_bands, ψ=ψ, n_ep_extra=n_ep_extra,
                             miniter=diag_miniter, tol=diagtol)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρout, eigenvalues=res_V.eigenvalues,
                                            εF=res_V.εF)
        (basis=basis, ham=new_ham, energies=new_E, Vin=Vin,
         Vout=total_local_potential(new_ham), res_V...)
    end

    n_iter    = 1
    converged = false
    α_trial   = initial_damping(damping)
    diagtol   = determine_diagtol((ρin=ρ, Vin=V, n_iter=n_iter))
    info      = EVρ(V; diagtol=diagtol, ψ=ψ)
    Pinv_δV   = mix_potential(mixing, basis, info.Vout - info.Vin; info...)
    info      = merge(info, (α=NaN, diagonalization=[info.diagonalization], ρin=ρ,
                             n_iter=n_iter, Pinv_δV=Pinv_δV))
    ΔEdown    = 0.0
    n_acceleration_off = 0  # >0 switches acceleration off for a few steps if in difficult region

    while n_iter < maxiter
        info = merge(info, (stage=:iterate, converged=converged,
                            n_acceleration_off=n_acceleration_off))
        callback(info)
        if is_converged(info)
            converged = true
            break
        end
        n_iter += 1
        info = merge(info, (n_iter=n_iter, ))

        # New search direction via convergence accelerator:
        αdiis = max(α_accel_min, α_trial)
        δV    = (acceleration(info.Vin, αdiis, info.Pinv_δV) - info.Vin) / αdiis
        if n_acceleration_off > 0
            δV = Pinv_δV
        end

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
            Vnext = info.Vin + α * δV

            info_next    = EVρ(Vnext; ψ=guess, diagtol=diagtol)
            Pinv_δV_next = mix_potential(mixing, basis, info_next.Vout - info_next.Vin; info_next...)
            push!(diagonalization, info_next.diagonalization)
            info_next = merge(info_next, (α=α, diagonalization=diagonalization,
                                          ρin=info.ρout, n_iter=n_iter,
                                          Pinv_δV=Pinv_δV_next))

            successful = accept_step(info, info_next)
            if successful || n_backtrack ≥ max_backtracks
                break
            end
            n_backtrack += 1

            # Adjust α to try again ...
            α_next = propose_backtrack_damping(damping, info, info_next)
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
        if !successful && n_acceleration_off == 0
            if abs(ΔE) > abs(ratio_failure_accel_off * ΔEdown)
                n_acceleration_off = 2  # will be reduced to 2 in the next line ...
                if mpi_master()
                    @warn "Backtracking linesearch failed badly. Acceleration not used for two steps."
                    @debug ΔE ΔEdown ratio_failure_accel_off
                end
            end
        else
            n_acceleration_off = max(0, n_acceleration_off - 1)
        end

        # Update α_trial and commit the next state
        α_trial = next_trial_damping(damping, info, info_next, successful)
        info = info_next
    end

    ham  = hamiltonian_with_total_potential(ham, info.Vout)
    info = (ham=ham, basis=basis, energies=info.energies, converged=converged,
            ρ=info.ρout, eigenvalues=info.eigenvalues, occupation=info.occupation,
            εF=info.εF, n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=info.ψ,
            diagonalization=info.diagonalization, stage=:finalize)
    callback(info)
    info
end


function scf_potential_mixing_adaptive(basis; tol=1e-6, mode=:standard, kwargs...)
    extraargs = (
        determine_diagtol=ScfDiagtol(ratio_ρdiff=0.03, diagtol_max=5e-3),
        damping=AdaptiveDamping(),
        diag_miniter=2,
    )
    if mode == :standard
        # Standard settings (decent compromise between speed and robustness)
        scf_potential_mixing(basis; tol=tol,
                             accept_step=ScfAcceptImprovingStep(max_energy_change=tol),
                             ratio_failure_accel_off=0.01,
                             α_accel_min=0.2,
                             extraargs..., kwargs...)
    else
        # Slower but more robust settings
        accept_step = ScfAcceptImprovingStep(max_energy_change=tol, max_relative_residual=0.0)
        scf_potential_mixing(basis; tol=tol,
                             accept_step=accept_step,
                             ratio_failure_accel_off=0.0,
                             α_accel_min=0.0,
                             extraargs..., kwargs...)
    end
end
