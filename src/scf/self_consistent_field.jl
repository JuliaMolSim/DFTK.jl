"""
Obtain new density ρ by diagonalizing `ham`.
"""
function next_density(ham::Hamiltonian, n_bands; Psi=nothing,
                      prec_type=PreconditionerTPA, tol=1e-6, n_ep_extra=3,
                      eigensolver=lobpcg_hyper)
    n_ep = n_bands + n_ep_extra
    if Psi !== nothing
        @assert length(Psi) == length(ham.basis.kpoints)
        for ik in 1:length(ham.basis.kpoints)
            @assert size(Psi[ik], 2) == n_bands + n_ep_extra
        end
    end

    # Diagonalize
    eigres = diagonalise_all_kblocks(eigensolver, ham, n_ep, guess=Psi,
                                     n_conv_check=n_bands, prec_type=prec_type, tol=tol)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Update density from new Psi
    occupation, εF = find_occupation(ham.basis, eigres.λ)
    ρnew = compute_density(ham.basis, eigres.X, occupation)

    (Psi=eigres.X, orben=eigres.λ, occupation=occupation, εF=εF, ρ=ρnew)
end

function scf_default_callback(info)
    E = sum(values(info.energies))
    res = norm(info.ρout.fourier - info.ρin.fourier)
    has_entropy = get(info.energies, :Entropy, 0.0) > 1e-16
    if info.neval == 1
        label = has_entropy ? "E - TS" : "Energy"
        println("Iter   $label             ρout-ρin")
        println("----   ------             --------")
    end
    @printf "%3d    %-15.12f    %E\n" info.neval E res
end

"""
Flag convergence as soon as total energy change drops below tolerance
"""
function scf_convergence_energy_difference(tolerance)
    energy_total = NaN

    function is_converged(info)
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
Solve the KS equations with a SCF algorithm, starting at `ham`
"""
function self_consistent_field(ham::Hamiltonian, n_bands;
                               Psi=nothing, tol=1e-6, max_iter=100,
                               solver=scf_nlsolve_solver(),
                               eigensolver=lobpcg_hyper, n_ep_extra=3, diagtol=tol / 10,
                               mixing=SimpleMixing(), callback=scf_default_callback,
                               is_converged=scf_convergence_energy_difference(tol))
    T = real(eltype(ham.density))
    basis = ham.basis
    model = basis.model

    # All these variables will get updated by fixpoint_map
    if Psi !== nothing
        @assert length(Psi) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(Psi[ik], 2) == n_bands + n_ep_extra
        end
    end
    occupation = nothing
    orben = nothing
    energies = nothing
    ρout = ham.density  # Initial ρout is initial guess
    εF = nothing
    @assert ρout !== nothing
    neval = 0

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(x)
        # Get ρout by diagonalizing the Hamiltonian
        ρin = from_real(basis, x)

        # Build next Hamiltonian, diagonalize it, get ρout
        ham = update_hamiltonian(ham, ρin)
        Psi, orben, occupation, εF, ρout = next_density(ham, n_bands; Psi=Psi,
                                                        eigensolver=eigensolver, tol=diagtol)
        energies = update_energies(ham, Psi, occupation, ρout)
        energies[:Entropy] = -model.temperature * compute_entropy(basis, orben, εF=εF)

        # mix it with ρin to get a proposal step
        ρnext = mix(mixing, basis, ρin, ρout)
        neval += 1

        info = (ham=ham, energies=energies, ρin=ρin, ρout=ρout, ρnext=ρnext,
                orben=orben, occupation=occupation, εF=εF, neval=neval)
        callback(info)
        is_converged(info) && return x

        ρnext.real
    end

    fpres = solver(fixpoint_map, ρout.real, max_iter; tol=min(10eps(T), tol / 10))
    # Tolerance is only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map

    # Strip off the extra (unconverged) eigenpairs
    # TODO we might want to keep them
    Psi = [p[:, 1:end-n_ep_extra] for p in Psi]
    orben = [oe[1:end-n_ep_extra] for oe in orben]
    occupation = [occ[1:end-n_ep_extra] for occ in occupation]

    (ham=ham, energies=energies, converged=fpres.converged,
     ρ=ρout, Psi=Psi, orben=orben, occupation=occupation, εF=εF)
end
