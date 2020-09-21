using Plots
include("scf_callbacks.jl")

function default_n_bands(model)
    min_n_bands = div(model.n_electrons, filled_occupation(model))
    n_extra = model.temperature == 0 ? 0 : max(4, ceil(Int, 0.2 * min_n_bands))
    min_n_bands + n_extra
end

"""
Obtain new density ρ by diagonalizing `ham`.
"""
function next_density(ham::Hamiltonian;
                      n_bands=default_n_bands(ham.basis.model),
                      ψ=nothing, n_ep_extra=3,
                      eigensolver=lobpcg_hyper,
                      occupation_function=find_occupation, kwargs...)
    if ψ !== nothing
        @assert length(ψ) == length(ham.basis.kpoints)
        for ik in 1:length(ham.basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end

    # Diagonalize
    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands + n_ep_extra; guess=ψ,
                                     n_conv_check=n_bands, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Update density from new ψ
    occupation, εF = occupation_function(ham.basis, eigres.λ)
    ρnew = compute_density(ham.basis, eigres.X, occupation)

    (ψ=eigres.X, eigenvalues=eigres.λ, occupation=occupation, εF=εF, ρ=ρnew,
     diagonalization=eigres)
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
                                       mixing=SimpleMixing(),
                                       is_converged=ScfConvergenceEnergy(tol),
                                       callback=ScfDefaultCallback(),
                                       compute_consistent_energies=true,
                                       enforce_symmetry=false,
                                       occupation_function=find_occupation,
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

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(x)
        n_iter += 1
        ρin = from_real(basis, x)

        # Build next Hamiltonian, diagonalize it, get ρout
        if n_iter == 1 # first iteration
            _, ham = energy_hamiltonian(basis, nothing, nothing;
                                        ρ=ρin, eigenvalues=nothing, εF=nothing)
        else
            # Note that ρin is not the density of ψ, and the eigenvalues
            # are not the self-consistent ones, which makes this energy non-variational
            energies, ham = energy_hamiltonian(basis, ψ, occupation;
                                               ρ=ρin, eigenvalues=eigenvalues, εF=εF)
        end

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham; n_bands=n_bands, ψ=ψ, eigensolver=eigensolver,
                                 miniter=1, tol=determine_diagtol(info),
                                 n_ep_extra=n_ep_extra,
                                 occupation_function=occupation_function)
        ψ, eigenvalues, occupation, εF, ρout = nextstate
        enforce_symmetry && (ρout = DFTK.symmetrize(ρout))

        # Compute the energy of the new state
        if compute_consistent_energies
            energies, _ = energy_hamiltonian(basis, ψ, occupation;
                                             ρ=ρout, eigenvalues=eigenvalues, εF=εF)
        end

        # Update info with results gathered so far
        info = (ham=ham, basis=basis, energies=energies, converged=false, ρin=ρin, ρout=ρout,
                eigenvalues=eigenvalues, occupation=occupation, εF=εF, n_iter=n_iter, ψ=ψ,
                diagonalization=nextstate.diagonalization, stage=:iterate)

        # Apply mixing and pass it the full info as kwargs
        ρnext = mix(mixing, basis, ρin, ρout; info...)
        enforce_symmetry && (ρnext = DFTK.symmetrize(ρnext))
        info = merge(info, (ρnext=ρnext, ))

        callback(info)
        is_converged(info) && return x

        ρnext.real
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map
    fpres = solver(fixpoint_map, ρout.real, maxiter; tol=eps(T))

    # We do not use the return value of fpres but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform
    # a last energy computation to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation;
                                       ρ=ρout, eigenvalues=eigenvalues, εF=εF)

    # Callback is run one last time with final state to allow callback to clean up
    info = (ham=ham, basis=basis, energies=energies, converged=fpres.converged,
            ρ=ρout, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, ψ=ψ, diagonalization=info.diagonalization, stage=:finalize)
    callback(info)
    info
end
