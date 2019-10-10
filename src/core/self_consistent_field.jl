"""
Obtain new density ρ by diagonalizing the Hamiltonian build from the current ρ.
If Psi is provided, overwrite it with the new wavefunctions as well.
"""
function iterate_density(ham::Hamiltonian, n_bands, compute_occupation, ρ;
                         Psi=nothing, lobpcg_kwargs...)
    pw = ham.basis
    T = real(eltype(ρ))
    # Initialize guess wavefunctions if needed
    if Psi === nothing
        Psi = [Matrix(qr(randn(Complex{T}, length(pw.basis_wf[ik]), n_bands)).Q)
               for ik in 1:length(pw.kpoints)]
    end
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    energies = Dict{Symbol, T}()
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)


    # Initialise guess for wave function and occupation
    Psi = [Matrix(qr(randn(Complex{T}, length(pw.basis_wf[ik]), n_bands)).Q)
           for ik in 1:length(pw.kpoints)]

    res = lobpcg(ham, n_bands; pot_hartree_values=values_hartree,
                 pot_xc_values=values_xc, guess=Psi, lobpcg_kwargs...)
    @assert(res.converged, "Not converged, iterations: $(res.iterations)")
    Psi .= res.X
    occupation = compute_occupation(ham.basis, res.λ, res.X)
    ρ_new = compute_density(pw, res.X, occupation)
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
"""
# Scaling is from 0 to 1. 0 is density mixing, 1 is "potential mixing"
# (at least, Hartree potential mixing). 1/2 results in a symmetric
# Jacobian of the SCF mapping (when there is no exchange-correlation)
function self_consistent_field(ham::Hamiltonian, n_bands::Integer;
                               ρ=nothing, tol=1e-6, max_iter=100,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                               solver=scf_nlsolve_solver(), den_scaling=0.0,
                               lobpcg_kwargs...)


    # if ρ is not nothing, initialise Hamiltonian no it
    # extract n_electrons from model


    if smearing === nothing
        compute_occupation =
            (basis, energies, Psi) -> occupation_step(basis, energies, Psi)
    else
        compute_occupation =
            (basis, energies, Psi) -> occupation_temperature(basis, energies, Psi,
                                                             T, smearing)
    end

    ham = copy(ham)
    ρ !== nothing && build_hamiltonian!(ham, ρ)







    pw = ham.basis
    T = real(eltype(ρ))
    Gsq = vec([T(4π) * sum(abs2, pw.recip_lattice * G) for G in basis_ρ(pw)])
    Gsq[pw.idx_DC] = 1.0 # do not touch the DC component
    den_to_mixed = Gsq.^T(-den_scaling)
    mixed_to_den = Gsq.^T(den_scaling)

    # TODO remove foldρ and unfoldρ when https://github.com/JuliaNLSolvers/NLsolve.jl/pull/217 is in a release
    function foldρ(ρ)
        ρ = den_to_mixed .* ρ
        # Fold a complex array representing the Fourier transform of a purely real
        # quantity into a real array
        half = Int((length(ρ) + 1) / 2)
        ρcpx =  ρ[1:half]
        vcat(real(ρcpx), imag(ρcpx))
    end
    function unfoldρ(ρ)
        # Undo "foldρ"
        half = Int(length(ρ) / 2)
        ρcpx = ρ[1:half] + im * ρ[half+1:end]
        ρ_unfolded = vcat(ρcpx, conj(reverse(ρcpx)[2:end]))
        ρ_unfolded .* mixed_to_den
    end
    function fp_map(ρ)
        foldρ(iterate_density(ham, n_bands, compute_occupation, unfoldρ(ρ);
                              tol=lobpcg_tol, prec=lobpcg_prec, Psi=Psi,
                              lobpcg_kwargs...))
    end

    nlres = fp_solver(fp_map, foldρ(ρ), tol, max_iter)
    ρ = unfoldρ(nlres.sol)
    energies = Dict{Symbol, real(eltype(ρ))}()
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

    # Final LOBPCG to get eigenvalues and eigenvectors
    res = lobpcg(ham, n_bands; pot_hartree_values=values_hartree,
                 pot_xc_values=values_xc, guess=Psi,
                 prec=lobpcg_prec, tol=lobpcg_tol, lobpcg_kwargs...)

    occupation = compute_occupation(ham.basis, res.λ, res.X)
    update_energies_1e!(energies, ham, ρ, res.X, occupation)

    (ρ=ρ, Psi=res.X, orben=res.λ, occupation=occupation, energies=energies,
     pot_hartree_values=values_hartree, pot_xc_values=values_xc, converged=nlres.converged)
end

# TODO Merge this function with core/scf
function self_consistent_field(ham::Hamiltonian, n_bands::Integer;
                               ρ=nothing, tol=1e-6,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                               max_iter=100, solver=scf_nlsolve_solver(),
                               den_scaling=0.0, lobpcg_kwargs...)

end
