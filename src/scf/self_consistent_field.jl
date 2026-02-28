include("scf_callbacks.jl")
using Dates

"""
Transparently handle checkpointing by either returning kwargs for `self_consistent_field`,
which start checkpointing (if no checkpoint file is present) or that continue a checkpointed
run (if a checkpoint file can be loaded). `filename` is the location where the checkpoint
is saved, `save_ψ` determines whether orbitals are saved in the checkpoint as well.
The latter is discouraged, since generally slow.
See [Saving SCF results on disk and SCF checkpoints](@ref) for details how to use
this function in practice.
"""
function kwargs_scf_checkpoints(basis::AbstractBasis;
                                filename="dftk_scf_checkpoint.jld2",
                                callback=ScfDefaultCallback(),
                                diagtolalg::AdaptiveDiagtol=AdaptiveDiagtol(),
                                ρ=guess_density(basis),
                                τ=any(needs_τ, basis.terms) ? zero(ρ) : nothing,
                                ψ=nothing, save_ψ=false,
                                kwargs...)
    if isfile(filename)
        # Disable strict checking, since we can live with only the density data
        previous = load_scfres(filename, basis; skip_hamiltonian=true, strict=false)

        # If we can expect the guess to be good, tighten the diagtol.
        if !isnothing(previous.ρ)
            ρ = previous.ρ
            τ = previous.τ
            consistent_kpts = hasproperty(previous, :eigenvalues)
            if consistent_kpts && hasproperty(previous, :history_Δρ)
                diagtol_first = determine_diagtol(diagtolalg, previous)
            else
                diagtol_first = diagtolalg.diagtol_max
            end
            diagtolalg = AdaptiveDiagtol(; diagtol_first,
                                           diagtolalg.diagtol_max,
                                           diagtolalg.diagtol_min,
                                           diagtolalg.ratio_ρdiff)
        end
        ψ = something(previous.ψ, Some(ψ))
    end

    callback = callback ∘ ScfSaveCheckpoints(; filename, save_ψ)
    (; callback, diagtolalg, ψ, ρ, τ, kwargs...)
end


# Struct to store some options for forward-diff / reverse-diff response
# (unused in primal calculations)
@kwdef struct ResponseOptions
    verbose = true
end

"""
Obtain new density ρ by diagonalizing `ham`. Follows the policy imposed by the `bands`
data structure to determine and adjust the number of bands to be computed.
"""
function next_density(ham::Hamiltonian,
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
    if !increased_n_bands && minocc > nbandsalg.occupation_threshold
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`nbandsalg=AdaptiveBands(model)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`nbandsalg=AdaptiveBands(model; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    ρout = compute_density(ham.basis, eigres.X, occupation; nbandsalg.occupation_threshold)
    (; ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres,
     n_bands_converge, nbandsalg.occupation_threshold,
     n_matvec=mpi_sum(eigres.n_matvec, ham.basis.comm_kpts))
end


@doc raw"""
    self_consistent_field(basis; [tol, mixing, damping, ρ, ψ])

Solve the Kohn-Sham equations with a density-based SCF algorithm using damped, preconditioned
iterations where ``ρ_\text{next} = ρ_\text{in} + α P^{-1} (ρ_\text{out} - ρ_\text{in})``.

Overview of parameters:
- `ρ`:   Initial density
- `ψ`:   Initial orbitals
- `tol`: Tolerance for the density change (``\|ρ_\text{out} - ρ_\text{in}\|``)
  to flag convergence. Default is `1e-6`.
- `is_converged`: Convergence control callback. Typical objects passed here are
  `ScfConvergenceDensity(tol)` (the default), `ScfConvergenceEnergy(tol)` or `ScfConvergenceForce(tol)`.
- `miniter`: Minimal number of SCF iterations
- `maxiter`: Maximal number of SCF iterations
- `maxtime`: Maximal time to run the SCF for. If this is reached without
   convergence, the SCF stops.
- `mixing`: Mixing method, which determines the preconditioner ``P^{-1}`` in the above equation.
  Typical mixings are [`LdosMixing`](@ref), [`KerkerMixing`](@ref), [`SimpleMixing`](@ref)
  or [`DielectricMixing`](@ref). Default is `LdosMixing()`
- `damping`: Damping parameter ``α`` in the above equation. Default is `0.8`.
- `nbandsalg`: By default DFTK uses `nbandsalg=AdaptiveBands(model)`, which adaptively determines
  the number of bands to compute. If you want to influence this algorithm or use a predefined
  number of bands in each SCF step, pass a [`FixedBands`](@ref) or [`AdaptiveBands`](@ref).
  Beware that with non-zero temperature, the convergence of the SCF algorithm may be limited
  by the `default_occupation_threshold()` parameter. For highly accurate calculations we thus
  recommend increasing the `occupation_threshold` of the `AdaptiveBands`.
- `callback`: Function called at each SCF iteration. Usually takes care of printing the
  intermediate state.
"""
@timing function self_consistent_field(
    basis::PlaneWaveBasis{T};
    ρ=guess_density(basis),
    τ=any(needs_τ, basis.terms) ? zero(ρ) : nothing,
    hubbard_n=nothing,
    ψ=nothing,
    occupation=nothing,
    eigenvalues=nothing,
    tol=1e-6,
    is_converged=ScfConvergenceDensity(tol),
    miniter=0,
    maxiter=100,
    maxtime=Year(1),
    mixing=LdosMixing(),
    damping=0.8,
    solver=scf_anderson_solver(),
    eigensolver=lobpcg_hyper,
    diagtolalg=default_diagtolalg(basis; tol),
    nbandsalg::NbandsAlgorithm=AdaptiveBands(basis.model),
    fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model),
    callback=ScfDefaultCallback(; show_damping=false),
    compute_consistent_energies=true,
    seed=nothing,
    response=ResponseOptions(),  # Dummy here, only for AD
) where {T}
    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end
    start_ns = time_ns()
    timeout_date = Dates.now() + maxtime
    seed = seed_task_local_rng!(seed, basis.comm_kpts)

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin, info)
        (; ψ, occupation, eigenvalues, εF, n_iter, converged, timedout, τ, hubbard_n) = info
        n_iter += 1

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; 
                                           ρ=ρin, τ, hubbard_n, eigenvalues, εF, 
                                           occupation_threshold=nbandsalg.occupation_threshold)

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1,
                                 tol=determine_diagtol(diagtolalg, info))
        (; ψ, eigenvalues, occupation, εF, ρout) = nextstate
        Δρ = ρout - ρin

        # TODO Dirty hack. This should be solved more generally and τ and ρ should be put
        #      on the same footing. In the future such principles will also apply
        #      to other quantities. See discussion in
        #      https://github.com/JuliaMolSim/DFTK.jl/issues/1065
        if any(needs_τ, basis.terms)
            τ = compute_kinetic_energy_density(basis, ψ, occupation)
        end
        ihubbard = findfirst(t -> t isa TermHubbard, basis.terms)
        if !isnothing(ihubbard)
            hubbard_n = compute_hubbard_n(basis.terms[ihubbard], basis, ψ, occupation)
        end

        # Update info with results gathered so far
        info_next = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                       ρin, τ, hubbard_n, α=damping, n_iter, nbandsalg.occupation_threshold,
                       seed, runtime_ns=time_ns() - start_ns, nextstate...,
                       diagonalization=[nextstate.diagonalization])

        # Compute the energy of the new state
        if compute_consistent_energies
            (; energies) = energy(basis, ψ, occupation; 
                                  ρ=ρout, τ, hubbard_n, eigenvalues, εF,
                                  occupation_threshold=nbandsalg.occupation_threshold)
        end
        history_Etot = vcat(info.history_Etot, energies.total)
        history_Δρ = vcat(info.history_Δρ, norm(Δρ) * sqrt(basis.dvol))
        n_matvec = info.n_matvec + nextstate.n_matvec
        info_next = merge(info_next, (; energies, history_Etot, history_Δρ, n_matvec))

        # Apply mixing and pass it the full info as kwargs
        if basis.model.spin_polarization == :full
            # Precondition the charge density using the chosen mixer
            # We slice 1:1 to maintain the 4th dimension for the mixer's internal array shapes
            Δρ_charge = @view Δρ[:, :, :, 1:1]
            δρ_charge = mix_density(mixing, basis, Δρ_charge; info_next...)
            
            # Use simple mixing (no preconditioning) for the vector magnetization
            δρ_mag = @view Δρ[:, :, :, 2:4]
            
            # Recombine and apply damping
            δρ_mixed = cat(δρ_charge, δρ_mag; dims=4)
            ρnext = ρin .+ T(damping) .* δρ_mixed
        else
            ρnext = ρin .+ T(damping) .* mix_density(mixing, basis, Δρ; info_next...)
        end

        converged = n_iter ≥ miniter && is_converged(info_next)
        converged = MPI.bcast(converged, 0, MPI.COMM_WORLD)
        info_next = merge(info_next, (; converged))

        timedout = mpi_bcast(Dates.now() ≥ timeout_date, basis.comm_kpts)
        info_next = merge(info_next, (; timedout))

        callback(info_next)

        ρnext, info_next
    end

    info_init = (; ρin=ρ, τ, hubbard_n, ψ, occupation, eigenvalues, εF=nothing,
                   n_iter=0, n_matvec=0, timedout=false, converged=false,
                   history_Etot=T[], history_Δρ=T[])

    # Convergence is flagged by is_converged inside the fixpoint_map.
    _, info = solver(fixpoint_map, ρ, info_init; maxiter)

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy
    (; ρin, ρout, τ, hubbard_n, ψ, occupation, eigenvalues, εF, converged) = info
    energies, ham = energy_hamiltonian(basis, ψ, occupation; 
                                       ρ=ρout, τ, hubbard_n, eigenvalues, εF, 
                                       occupation_threshold=nbandsalg.occupation_threshold)

    # Callback is run one last time with final state to allow callback to clean up
    scfres = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
                ρ=ρout, τ, hubbard_n, α=damping, eigenvalues, occupation, εF,
                info.n_bands_converge, info.n_iter, info.n_matvec, ψ, info.diagonalization, 
                stage=:finalize, info.history_Δρ, info.history_Etot, info.timedout, mixing, 
                seed, runtime_ns=time_ns() - start_ns, algorithm="SCF")
    callback(scfres)
    scfres
end
