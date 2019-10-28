"""
Obtain new density ρ by diagonalizing the Hamiltonian build from the current ρ.
`ham` and `Psi` (if given) are updated in-place.
"""
function iterate_density!(ham::Hamiltonian, n_bands, ρ=nothing; Psi=nothing,
                          prec=PreconditionerKinetic(ham, α=0.1), tol=1e-6,
                          compute_occupation=find_occupation_around_fermi,
                          eigensolver=lobpcg_hyper)
    # Update Hamiltonian from ρ
    ρ !== nothing && update_hamiltonian!(ham, ρ)

    # Update Psi from Hamiltonian (ask for a few more bands than the ones we need)
    n_ep = (Psi === nothing) ? n_bands + 3 : size(Psi[1], 2)
    eigres = diagonalise_all_kblocks(eigensolver, ham, n_ep, guess=Psi, n_conv_check=n_bands, prec=prec, tol=tol)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)
    Psi !== nothing && (Psi .= eigres.X)

    # Update density from new Psi
    εF, occupation = compute_occupation(ham.basis, eigres.λ, eigres.X)
    ρnew = compute_density(ham.basis, eigres.X, occupation)

    (ham=ham, Psi=eigres.X, orben=eigres.λ, occupation=occupation, εF=εF,
     ρ=ρnew)
end


"""
    self_consistent_field(ham::Hamiltonian, n_bands::Int, n_electrons::Int;
                          ρ=nothing, tol=1e-6, max_iter=100, algorithm=:scf_nlsolve,
                          lobpcg_prec=PreconditionerKinetic(ham, α=0.1))

Run a self-consistent field iteration for the Hamiltonian `ham`, returning the
self-consistnet density, Hartree potential values and XC potential values.
`n_bands` selects the number of bands to be computed, `n_electrons` the number of
electrons, `ρ` is the initial density, e.g. constructed via a SAD guess.
`lobpcg_prec` specifies the preconditioner used in the LOBPCG algorithms used
for diagonalisation. Possible `algorithm`s are `:scf_nlsolve` or `:scf_damped`.

compute_occupation is around to manipulate the way occupations are computed.
"""
function self_consistent_field!(ham::Hamiltonian, n_bands;
                                Psi=nothing, tol=1e-6, max_iter=100,
                                solver=scf_nlsolve_solver(),
                                eigensolver=lobpcg_hyper, n_ep_extra=3, diagtol=tol / 10)
    T = eltype(real(ham.density))
    basis = ham.basis
    model = basis.model
    if Psi === nothing   # Initialize random guess wavefunction
        # TODO This assumes CPU arrays
        Psi = [Matrix(qr(randn(Complex{T}, length(kpt.basis), n_bands + n_ep_extra)).Q)
               for kpt in basis.kpoints]
    end

    # TODO When https://github.com/JuliaNLSolvers/NLsolve.jl/pull/217
    # is in a release, then the fixpoint iteration could also be done
    # in Fourier space (i.e. in the actual discretisation basis),
    # which might make the procedure more general ...
    # would be good to give it a try and benchmark this.

    # NLSolve can only work with 1D arrays as parameters, so we need to
    # fold and unfold the Density object appropriately. Notice that we pick
    # the *real-space* representation of the density
    foldρ(ρ) = vec(real(ρ))
    unfoldρ(vec_real) = density_from_real(basis, reshape(vec_real, basis.fft_size))
    function fixpoint_map(x)
        ρ = iterate_density!(ham, n_bands, unfoldρ(x); Psi=Psi, eigensolver=eigensolver, tol=diagtol).ρ
        foldρ(ρ)
    end

    # Run fixpoint solver: Take guess density from Hamiltonian or iterate once
    #                      to generate it from its eigenvectors
    ρ = ham.density
    ρ === nothing && (ρ = iterate_density!(ham, n_bands; Psi=Psi, eigensolver=eigensolver, tol=diagtol).ρ)
    fpres = solver(fixpoint_map, foldρ(ρ), tol, max_iter)
    ρ = unfoldρ(fpres.fixpoint)

    # Extra step to get Hamiltonian, eigenvalues and eigenvectors wrt the fixpoint density
    itres = iterate_density!(ham, n_bands, ρ; Psi=Psi, eigensolver=eigensolver, tol=diagtol)

    # TODO energies ... maybe do them optionally in iterate_density along with
    #      the update_hamiltonian function
    energies = update_energies(ham, itres.Psi, itres.occupation, itres.ρ)

    # Strip off the extra (unconverged) eigenpairs
    Psi = [p[1:end-n_ep_extra, :] for p in itres.Psi]
    orben = [oe[1:end-n_ep_extra] for oe in itres.orben]
    occupation = [occ[1:end-n_ep_extra] for occ in itres.occupation]

    merge(itres, (energies=energies, converged=fpres.converged,
                  Psi=Psi, orben=orben, occupation=occupation))
end
