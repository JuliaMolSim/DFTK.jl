"""
Setup LOBPCG eigensolver
"""
function diag_lobpcg(;kwargs...)
    @warn "diag_lobpcg should be split into hyper and the different lobpcg flavours."
    # Return a function, which calls the lobpcg routine. By default the kwargs
    # from the scf (passed as scfkwargs) are used, unless they are overwritten
    # by the kwargs passed upon call to diag_lobpcg.
    (ham, n_ep; scfkwargs...) -> lobpcg(ham, n_ep; merge(scfkwargs, kwargs)...)
end


"""
Obtain new density ρ by diagonalizing the Hamiltonian build from the current ρ.
`ham` and `Psi` (if given) are updated in-place.
"""
function iterate_density!(ham::Hamiltonian, n_bands, ρ=nothing; Psi=nothing,
                          prec=PreconditionerKinetic(ham, α=0.1), tol=1e-6,
                          compute_occupation=find_occupation_around_fermi, diag=diag_lobpcg)
    # Update Hamiltonian from ρ
    ρ !== nothing && update_hamiltonian!(ham, ρ)

    # Update Psi from Hamiltonian (ask for a few more bands than the ones we need)
    n_ep = (Psi === nothing) ? n_bands + 3 : size(Psi[1], 2)
    eigres = diag(ham, n_ep; guess=Psi, n_conv_check=n_bands, prec=prec, tol=tol)
    eigres.converged || (@warn "LOBPCG not converged" iterations=res.iterations)
    Psi !== nothing && (Psi .= eigres.X)

    # Update density from new Psi
    occupation = compute_occupation(ham.basis, eigres.λ, eigres.X)
    ρnew = compute_density(ham.basis, eigres.X, occupation)
    ρ !== nothing && (ρ .= ρnew)

    (ham=ham, Psi=eigres.X, orben=eigres.λ, occupation=occupation, ρ=ρ)
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
                                diag=diag_lobpcg(), n_ep_extra=3)
    T = real(eltype(ham.density))
    diagtol = tol / 10.
    model = ham.basis.model
    if Psi === nothing   # Initialize random guess wavefunction
        # TODO This assumes CPU arrays
        Psi = [Matrix(qr(randn(Complex{T}, length(kpt.basis), n_bands + n_ep_extra)).Q)
               for kpt in ham.basis.kpoints]
    end

    # TODO remove foldρ and unfoldρ when https://github.com/JuliaNLSolvers/NLsolve.jl/pull/217 is in a release
    function foldρ(ρ)
        return vec(real(r_to_G(ham.basis, ρ .+ 0im)))

        # TODO Does not work and disabled for now
        # Fold a complex array representing the Fourier transform of a purely real
        # quantity into a real array
        half = ceil(Int, size(ρ, 1) / 2)
        ρcpx =  ρ[1:half, :, :]
        vcat(real(ρcpx), imag(ρcpx))
    end
    function unfoldρ(ρ)
        return G_to_r(ham.basis, reshape(ρ .+ 0im, ham.basis.fft_size))

        # TODO Does not work and disabled for now
        half = Int(size(ρ, 1) / 2)
        ρcpx = ρ[1:half, :, :] + im * ρ[half+1:end, :, :]

        # Undo "foldρ"
        vcat(ρcpx, conj(reverse(reverse(ρcpx[2:end, :, :], dims=2), dims=3)))
    end
    fixpoint_map(ρ) = foldρ(iterate_density!(ham, n_bands, unfoldρ(ρ); Psi=Psi, diag=diag,
                                             tol=diagtol).ρ)

    # Run fixpoint solver: Take guess density from Hamiltonian or iterate once
    #                      to generate it from its eigenvectors
    ρ = ham.density
    ρ === nothing && (ρ = iterate_density!(ham, n_bands; Psi=Psi, diag=diag, tol=diagtol).ρ)

    # TODO Temporary test
    ρ = r_to_G(ham.basis, (0im .+ real(G_to_r(ham.basis, ρ .+ 0im))))
    @assert unfoldρ(foldρ(ρ)) ≈ ρ

    fpres = solver(fixpoint_map, foldρ(ρ), tol, max_iter)
    ρ = unfoldρ(fpres.fixpoint)

    # Extra step to get Hamiltonian, eigenvalues and eigenvectors wrt the fixpoint density
    itres = iterate_density!(ham, n_bands, ρ; Psi=Psi, tol=diagtol, diag=diag)

    # TODO energies ... maybe do them optionally in iterate_density along with
    #      the update_hamiltonian function
    energies = Dict{Symbol, T}()

    # Strip off the extra (unconverged) eigenpairs
    Psi = [p[1:end-n_ep_extra, :] for p in itres.Psi]
    orben = [oe[1:end-n_ep_extra] for oe in itres.orben]
    occupation = [occ[1:end-n_ep_extra] for occ in itres.occupation]

    merge(itres, (energies=energies, converged=fpres.converged,
                  Psi=Psi, orben=orben, occupation=occupation))
end
