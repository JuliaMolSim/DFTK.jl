include("scf_callbacks.jl")

# Struct to store some options for forward-diff / reverse-diff response
# (unused in primal calculations)
@kwdef struct ResponseOptions
    verbose = false
    occupation_threshold = 1e-10
end

function default_n_bands(model)
    min_n_bands = div(model.n_electrons, filled_occupation(model), RoundUp)
    n_extra = model.temperature == 0 ? 0 : max(4, ceil(Int, 0.2 * min_n_bands))
    min_n_bands + n_extra
end

"""
Obtain new density ρ by diagonalizing `ham`.
"""
function next_density(ham::Hamiltonian;
                      n_bands=default_n_bands(ham.basis.model),
                      ψ=nothing, n_ep_extra=3,
                      eigensolver=lobpcg_hyper, kwargs...)
    if ψ !== nothing
        @assert length(ψ) == length(ham.basis.kpoints)
        for ik in 1:length(ham.basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end

    # Diagonalize
    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands + n_ep_extra; ψguess=ψ,
                                     n_conv_check=n_bands, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Update density from new ψ
    occupation, εF = compute_occupation(ham.basis, eigres.λ)
    ρout = compute_density(ham.basis, eigres.X, occupation)

    (ψ=eigres.X, eigenvalues=eigres.λ, occupation=occupation, εF=εF,
     ρout=ρout, diagonalization=eigres)
end


"""
Solve the Kohn-Sham equations with a SCF algorithm, starting at ρ.
"""
@timing function self_consistent_field(basis::PlaneWaveBasis;
                                       n_bands=default_n_bands(basis.model),
                                       ρ=guess_density(basis),
                                       ψ=nothing,
                                       tol=1e-6,
                                       maxiter=100,
                                       solver=scf_nlsolve_solver(),
                                       eigensolver=lobpcg_hyper,
                                       n_ep_extra=3,
                                       determine_diagtol=ScfDiagtol(),
                                       damping=0.8,  # Damping parameter
                                       mixing=LdosMixing(),
                                       is_converged=ScfConvergenceEnergy(tol),
                                       callback=ScfDefaultCallback(; show_damping=false),
                                       compute_consistent_energies=true,
                                       response=ResponseOptions()  # Dummy here, only needed
                                                                   # for forward-diff.
                                      )
    T = eltype(basis)
    model = basis.model

    # All these variables will get updated by fixpoint_map
    if ψ !== nothing
        @assert length(ψ) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end
    occupation = nothing
    eigenvalues = nothing
    ρout = ρ
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing
    info = (n_iter=0, ρin=ρ)   # Populate info with initial values
    converged = false

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin)
        converged && return ρin  # No more iterations if convergence flagged

        n_iter += 1

        # Build next Hamiltonian, diagonalize it, get ρout
        if n_iter == 1 # first iteration
            _, ham = energy_hamiltonian(basis, nothing, nothing;
                                        ρ=ρin, eigenvalues=nothing, εF=nothing)
        else
            # Note that ρin is not the density of ψ, and the eigenvalues
            # are not the self-consistent ones, which makes this energy non-variational
            energies, ham = energy_hamiltonian(basis, ψ, occupation;
                                               ρ=ρin, eigenvalues, εF)
        end

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham; n_bands, ψ, eigensolver,
                                 miniter=1, tol=determine_diagtol(info),
                                 n_ep_extra)
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, n_ep_extra,
                nextstate..., diagonalization=[nextstate.diagonalization])

        # Compute the energy of the new state
        if compute_consistent_energies
            energies, _ = energy_hamiltonian(basis, ψ, occupation;
                                             ρ=ρout, eigenvalues=eigenvalues, εF=εF)
        end
        info = merge(info, (energies=energies, ))

        # Apply mixing and pass it the full info as kwargs
        δρ = mix_density(mixing, basis, ρout - ρin; info...)
        ρnext = ρin .+ T(damping) .* δρ
        info = merge(info, (; ρnext=ρnext))

        callback(info)
        is_converged(info) && (converged = true)

        ρnext
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map
    fpres = solver(fixpoint_map, ρout, maxiter; tol=eps(T))

    # We do not use the return value of fpres but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform
    # a last energy computation to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation;
                                       ρ=ρout, eigenvalues=eigenvalues, εF=εF)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged,
            ρ=ρout, α=damping, eigenvalues, occupation, εF,
            n_iter, n_ep_extra, ψ, info.diagonalization,
            stage=:finalize, algorithm="SCF", norm_Δρ)
    callback(info)
    info
end
