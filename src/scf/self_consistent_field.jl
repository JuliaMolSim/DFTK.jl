default_n_bands(model) = div(model.n_electrons, filled_occupation(model))

"""
Obtain new density ρ by diagonalizing `ham`.
"""
function next_density(ham::Hamiltonian;
                      n_bands=default_n_bands(ham.basis.model),
                      ψ=nothing, n_ep_extra=3,
                      eigensolver=lobpcg_hyper, kwargs...)
    n_ep = n_bands + n_ep_extra
    if ψ !== nothing
        @assert length(ψ) == length(ham.basis.kpoints)
        for ik in 1:length(ham.basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end

    # Diagonalize
    eigres = diagonalise_all_kblocks(eigensolver, ham, n_ep; guess=ψ,
                                     n_conv_check=n_bands, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Update density from new ψ
    occupation, εF = find_occupation(ham.basis, eigres.λ)
    ρnew = compute_density(ham.basis, eigres.X, occupation)

    (ψ=eigres.X, eigenvalues=eigres.λ, occupation=occupation, εF=εF, ρ=ρnew,
     diagonalisation=eigres)
end

function scf_default_callback(info)
    E = info.energies === nothing ? Inf : sum(values(info.energies))
    res = norm(info.ρout.fourier - info.ρin.fourier)
    if info.neval == 1
        label = haskey(info.energies, "Entropy") ? "Free energy" : "Energy"
        @printf "Iter   %-15s    ρout-ρin\n" label
        @printf "----   %-15s    --------\n" "-"^length(label)
    end
    @printf "%3d    %-15.12f    %E\n" info.neval E res
end

"""
Flag convergence as soon as total energy change drops below tolerance
"""
function scf_convergence_energy_difference(tolerance)
    energy_total = NaN

    function is_converged(info)
        info.energies === nothing && return false # first iteration

        # For safety: The ρ change should also be below 10sqrt(tolerance)
        norm(info.ρout.fourier - info.ρin.fourier) > 10sqrt(tolerance) && return false

        etot_old = energy_total
        energy_total = sum(values(info.energies))
        abs(energy_total - etot_old) < tolerance
    end
    return is_converged
end

"""
Flag convergence by using the L2Norm of the change between
input density and unpreconditioned output density (ρout)
"""
function scf_convergence_density_difference(tolerance)
    info -> norm(info.ρout.fourier - info.ρin.fourier) < tolerance
end

"""
Determine the tolerance used for the next diagonalisation. This function takes
``|ρnext - ρin|`` and multiplies it with `ratio_ρdiff` to get the next `diagtol`,
ensuring additionally that the returned value is between `diagtol_min` and `diagtol_max`.
"""
function scf_determine_diagtol(ratio_ρdiff=0.2, diagtol_min=nothing, diagtol_max=0.1)
    function determine_diagtol(neval, norm_ρnext_ρin)
        isnothing(diagtol_min) && (diagtol_min = 500eps(real(eltype(norm_ρnext_ρin))))

        diagtol = norm_ρnext_ρin * ratio_ρdiff
        diagtol = min(diagtol_max, diagtol)  # Don't overshoot
        diagtol = max(diagtol_min, diagtol)  # Don't undershoot
        @assert isfinite(diagtol)

        # Adjust maximum to ensure diagtol may only shrink during an SCF
        diagtol_max = min(diagtol, diagtol_max)

        diagtol
    end
end


"""
Solve the Kohn-Sham equations with a SCF algorithm, starting at ρ.
"""
function self_consistent_field(basis::PlaneWaveBasis;
                               n_bands=default_n_bands(basis.model),
                               ρ=guess_density(basis),
                               ψ=nothing,
                               tol=1e-6,
                               maxiter=100,
                               solver=scf_nlsolve_solver(),
                               eigensolver=lobpcg_hyper,
                               n_ep_extra=3,
                               determine_diagtol=scf_determine_diagtol(),
                               mixing=SimpleMixing(),
                               callback=scf_default_callback,
                               is_converged=scf_convergence_energy_difference(tol),
                               compute_consistent_energies=true,
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
    neval = 0
    energies = nothing
    ham = nothing
    norm_ρnext_ρin = Inf  # The most recent norm(ρnext - ρin)

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(x)
        # Get ρout by diagonalizing the Hamiltonian
        ρin = from_real(basis, x)

        # Build next Hamiltonian, diagonalize it, get ρout
        if neval == 0 # first iteration
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
                                 miniter=1, tol=determine_diagtol(neval, norm_ρnext_ρin))
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        # This computes the energy of the new state
        if compute_consistent_energies
            energies, H = energy_hamiltonian(basis, ψ, occupation;
                                             ρ=ρout, eigenvalues=eigenvalues, εF=εF)
        end

        # mix it with ρin to get a proposal step
        ρnext = mix(mixing, basis, ρin, ρout)
        neval += 1

        info = (ham=ham, energies=energies, ρin=ρin, ρout=ρout, ρnext=ρnext,
                eigenvalues=eigenvalues, occupation=occupation, εF=εF, neval=neval, ψ=ψ,
                diagonalisation=nextstate.diagonalisation)
        callback(info)
        is_converged(info) && return x

        norm_ρnext_ρin = norm(ρnext.fourier - ρin.fourier)
        ρnext.real
    end

    fpres = solver(fixpoint_map, ρout.real, maxiter; tol=min(10eps(T), tol / 10))
    # Tolerance is only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map

    # We do not use the return value of fpres but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform
    # a last energy computation to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation;
                                       ρ=ρout, eigenvalues=eigenvalues, εF=εF)

    (ham=ham, energies=energies, converged=fpres.converged,
     ρ=ρout, ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF)
end
