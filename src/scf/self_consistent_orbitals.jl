"""
Obtain new density ρ by diagonalizing `ham`. Follows the policy imposed by the `bands`
data structure to determine and adjust the number of bands to be computed.
"""
function next_orbitals(ham::Hamiltonian,
                       nbandsalg::NbandsAlgorithm=AdaptiveBands(ham.basis.model),
                       fermialg::AbstractFermiAlgorithm=default_fermialg(ham.basis.model);
                       eigensolver=lobpcg_hyper, ψ=nothing, eigenvalues=nothing,
                       occupation=nothing, kwargs...)

    n_bands_converge, n_bands_compute = determine_n_bands(nbandsalg, occupation,
                                                          eigenvalues, ψ)

    if isnothing(ψ)
        increased_n_bands = true
    else
        @assert length(ψ) == length(ham.basis.kpoints)
        n_bands_compute = max(n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
        increased_n_bands = n_bands_compute > size(ψ[1], 2)
    end

    # TODO Synchronize since right now it is assumed that the same number of bands are
    #      computed for each k-Point
    n_bands_compute = mpi_max(n_bands_compute, ham.basis.comm_kpts)

    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute;
                                     ψguess=ψ, n_conv_check=n_bands_converge, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" n_iter=eigres.n_iter)

    # Check maximal occupation of the unconverged bands is sensible.
    occupation, εF = compute_occupation(ham.basis, eigres.λ, fermialg;
                                        tol_n_elec=nbandsalg.occupation_threshold)
    minocc = maximum(minimum, occupation)

    # TODO This is a bit hackish, but needed right now as we increase the number of bands
    #      to be computed only between SCF steps. Should be revisited once we have a better
    #      way to deal with such things in LOBPCG.
    if !increased_n_bands && minocc > nbandsalg.occupation_threshold && mpi_master(ham.basis.comm_kpts)
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`nbandsalg=AdaptiveBands(model)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`nbandsalg=AdaptiveBands(model; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    (; ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, diagonalization=eigres,
     n_bands_converge, nbandsalg.occupation_threshold,
     n_matvec=mpi_sum(eigres.n_matvec, ham.basis.comm_kpts))
end


@timing function self_consistent_orbitals(
    basis::PlaneWaveBasis{T};
    ψ=nothing,
    occupation=nothing,
    eigenvalues=nothing,
    εF=nothing,
    tol=1e-10,
    is_converged=ScfConvergenceEnergy(tol),
    miniter=0,
    maxiter=100,
    maxtime=Year(1),
    solver::OrbitalSolver=OrbitalSimpleSolver(),
    eigensolver=lobpcg_hyper,
    diagtolalg=default_diagtolalg(basis; tol=1e-6),
    nbandsalg::NbandsAlgorithm=AdaptiveBands(basis.model),
    fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model),
    exxalg::ExxAlgorithm=AceExx(),
    callback=ScfDefaultCallback(; show_damping=false),
    seed=nothing,
) where {T}
    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end
    start_ns = time_ns()
    timeout_date = Dates.now() + maxtime
    seed = seed_task_local_rng!(seed, basis.comm_kpts)

    # We use a "generalised density" representation in the variable D/Din, that is adapted to
    # linear combinations (such as mixing or Anderson); see split_gdensity and pack_gdensity in
    # densities.jl for details.
    #
    function fixpoint_map(x, info)
        (;ψ, occupation) = x
        (;eigenvalues, εF, n_iter, converged, timedout) = info
        n_iter += 1

        energies, ham = energy_hamiltonian(basis, ψ, occupation;
                                           exxalg, eigenvalues, εF, ρ=compute_density(basis, ψ, occupation;nbandsalg.occupation_threshold), 
                                           nbandsalg.occupation_threshold)

        # Diagonalize `ham` to get the new state
        nextstate = next_orbitals(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                  occupation, miniter=1,
                                  tol=determine_diagtol(diagtolalg, info))

	    (;ψ, eigenvalues, occupation, εF) = nextstate
        ρ = compute_density(basis, ψ, occupation; nbandsalg.occupation_threshold)

        # Update info with results gathered so far
        info_next = (; ham, basis, converged, ρ, stage=:iterate, algorithm="SCF",
                       n_iter, nbandsalg.occupation_threshold,
                       seed, runtime_ns=time_ns() - start_ns, nextstate...,
                       diagonalization=[nextstate.diagonalization])

        # Compute the energy of the new state
        (; energies) = energy(basis, ψ, occupation;
                              exxalg, eigenvalues, εF, ρ,
                              nbandsalg.occupation_threshold)

        history_Etot = vcat(info.history_Etot, energies.total)
        history_Δρ   = vcat(info.history_Δρ, 1.0)


        info_next = merge(info_next, (; energies, history_Etot, history_Δρ,
                                        n_matvec=info.n_matvec + nextstate.n_matvec))

        converged = mpi_bcast(n_iter ≥ miniter && is_converged(info_next), basis.comm_kpts)
        timedout  = mpi_bcast(Dates.now() ≥ timeout_date,                  basis.comm_kpts)
        info_next = merge(info_next, (; converged, timedout))
        callback(info_next)

        (;ψ, occupation), info_next
    end

    if isnothing(ψ) || isnothing(occupation)
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=guess_density(basis),
                                           exxalg, eigenvalues, 
                                           nbandsalg.occupation_threshold)

        nextstate = next_orbitals(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                  occupation, miniter=1,
                                  tol=determine_diagtol(diagtolalg, info))

	(;ψ, eigenvalues, occupation, εF) = nextstate
    end

    info_init = (;basis, eigenvalues, εF,
                  n_iter=0, n_matvec=0, timedout=false, converged=false,
                  history_Etot=T[], history_Δρ=T[])

    # Convergence is flagged by is_converged inside the fixpoint_map.
    _, info = solver(fixpoint_map, (;ψ, occupation), info_init; maxiter)

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρ, so we return that. We also perform a last energy computation
    # to return a correct variational energy and to build a Hamiltonian without any compression
    # applied to the exchange operator.
    (;ψ,occupation,eigenvalues,εF,converged,ρ) = info
    energies, ham = energy_hamiltonian(basis, ψ, occupation; 
                                       exxalg=VanillaExx(),
                                       eigenvalues, εF, ρ, 
                                       nbandsalg.occupation_threshold)

    # Callback is run one last time with final state to allow callback to clean up
    scfres = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
                eigenvalues, occupation, εF, ρ,
                info.n_bands_converge, info.n_iter, info.n_matvec, ψ, info.diagonalization,
                stage=:finalize, info.history_Etot,
                info.timedout, is_converged, nbandsalg, fermialg, diagtolalg, solver,
                eigensolver, seed, runtime_ns=time_ns() - start_ns, algorithm="SCF")
    callback(scfres)
    scfres
end
