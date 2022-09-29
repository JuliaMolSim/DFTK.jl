include("scf_callbacks.jl")

# Struct to store some options for forward-diff / reverse-diff response
# (unused in primal calculations)
@kwdef struct ResponseOptions
    verbose = false
end

"""
Solve the Kohn-Sham equations with a SCF algorithm, starting at `ρ`.

- `bands`: By default DFTK uses `bands=AdaptiveBands(basis)`, which adaptively determines
  the number of bands to compute. If you want to influence this algorithm or use a predefined
  number of bands in each SCF step, pass a [`FixedBands`](@ref) or [`AdaptiveBands`](@ref).
"""
@timing function self_consistent_field(basis::PlaneWaveBasis{T};
                                       n_bands=nothing,    # TODO For backwards compatibility.
                                       n_ep_extra=nothing, # TODO For backwards compatibility.
                                       ρ=guess_density(basis),
                                       ψ=nothing,
                                       tol=1e-6,
                                       maxiter=100,
                                       solver=scf_nlsolve_solver(),
                                       eigensolver=lobpcg_hyper,
                                       determine_diagtol=ScfDiagtol(),
                                       damping=0.8,  # Damping parameter
                                       mixing=LdosMixing(),
                                       is_converged=ScfConvergenceEnergy(tol),
                                       bands=AdaptiveBands(basis),
                                       callback=ScfDefaultCallback(; show_damping=false),
                                       compute_consistent_energies=true,
                                       response=ResponseOptions(),  # Dummy here, only for AD
                                      ) where {T}
    if !isnothing(n_bands) || !isnothing(n_ep_extra)
        # TODO Backwards compatibility ... emulates exactly how bands worked before
        @warn("The options n_bands and n_ep_extra of self_consistent_field are deprecated. " *
              "Use `bands` instead.")
        n_bands = something(n_bands, FixedBands(basis.model).n_bands)
        bands = FixedBands(;n_bands, n_bands_compute=n_bands + something(n_ep_extra, 3))
    end

    # All these variables will get updated by fixpoint_map
    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
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

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin, eigenvalues, εF)

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, bands; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol=determine_diagtol(info))
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, bands.occupation_threshold,
                nextstate..., diagonalization=[nextstate.diagonalization])

        # Compute the energy of the new state
        if compute_consistent_energies
            energies, _ = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF)
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
    # inside the fixpoint_map.
    solver(fixpoint_map, ρout, maxiter; tol=eps(T))

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged, bands.occupation_threshold,
            ρ=ρout, α=damping, eigenvalues, occupation, εF, info.n_bands_converge,
            n_iter, ψ, info.diagonalization, stage=:finalize,
            algorithm="SCF", norm_Δρ)
    callback(info)
    info
end


"""
Obtain new density ρ by diagonalizing `ham`. Follows the policy imposed by the `bands`
data structure to determine and adjust the number of bands to be computed.
"""
function next_density(ham::Hamiltonian, bands=AdaptiveBands(ham.basis.model);
                      eigensolver=lobpcg_hyper, ψ=nothing, eigenvalues=nothing,
                      occupation=nothing, kwargs...)
    n_bands_converge, n_bands_compute = determine_n_bands(bands, occupation, eigenvalues, ψ)

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

    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute; ψguess=ψ,
                                     n_conv_check=n_bands_converge, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Check maximal occupation of the unconverged bands is sensible.
    occupation, εF = compute_occupation(ham.basis, eigres.λ)
    minocc = maximum(minimum, occupation)

    # TODO This is a bit hackish, but needed right now as we increase the number of bands
    #      to be computed only between SCF steps. Should be revisited once we have a better
    #      way to deal with such things in LOBPCG.
    if !increased_n_bands && minocc > bands.occupation_threshold
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`bands=AdaptiveBands(basis)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`bands=AdaptiveBands(basis; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    # TODO We should probably set occupation values below occupation_threshold explicitly
    #      to zero and ignore these bands in the density computation
    ρout = compute_density(ham.basis, eigres.X, occupation)
    (ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres,
     n_bands_converge, bands.occupation_threshold)
end


# Band adjustment during SCF
function default_n_bands(model)
    n_spin = model.n_spin_components
    min_n_bands = div(model.n_electrons, n_spin * filled_occupation(model), RoundUp)
    n_extra = iszero(model.temperature) ? 0 : ceil(Int, 0.15 * min_n_bands)
    min_n_bands + n_extra
end
default_occupation_threshold() = 1e-6

"""
In each SCF step converge exactly `n_bands_converge`, computing along the way exactly
`n_bands_compute` (usually a few more to ease convergence in systems with small gaps).
"""
struct FixedBands
    n_bands_converge::Int # Number of bands to converge
    n_bands_compute::Int  # Number of bands to compute (but not necessarily converge)
    occupation_threshold::Float64  # Threshold for orbital to be counted as occupied
end
function FixedBands(; n_bands_converge, n_bands_compute=n_bands_converge+3,
                    occupation_threshold=default_occupation_threshold())
    FixedBands(n_bands_converge, n_bands_compute, occupation_threshold)
end
function FixedBands(model::Model)
    n_bands_converge = default_n_bands(model)
    n_bands_converge += iszero(model.temperature) ? 0 : ceil(Int, 0.05 * n_bands_converge)
    FixedBands(; n_bands_converge, n_bands_compute=n_bands_converge + 3)
end
FixedBands(basis::PlaneWaveBasis; kwargs...) = FixedBands(basis.model; kwargs...)
function determine_n_bands(bands::FixedBands, occupation, eigenvalues, ψ)
    # TODO Could return different bands per k-Points
    (; n_bands_converge=bands.n_bands_converge, bands.n_bands_compute)
end


"""
Dynamically adapt number of bands to be converged to ensure that the orbitals of lowest
occupation are occupied to at most `occupation_threshold`. To obtain rapid convergence
of the eigensolver a gap between the eigenvalues of the last occupied orbital and the last
computed (but not converged) orbital of `gap_min` is ensured.
"""
@kwdef struct AdaptiveBands
    n_bands_converge::Int  # Minimal number of bands to converge
    n_bands_compute::Int   # Minimal number of bands to compute
    occupation_threshold::Float64 = default_occupation_threshold()
    gap_min::Float64 = 1e-3   # Minimal gap between converged and computed bands
end
function AdaptiveBands(model::Model; n_bands_converge=default_n_bands(model), kwargs...)
    n_extra = iszero(model.temperature) ? 3 : max(4, ceil(Int, 0.05 * n_bands_converge))
    AdaptiveBands(; n_bands_converge, n_bands_compute=n_bands_converge + n_extra, kwargs...)
end
AdaptiveBands(basis::PlaneWaveBasis; kwargs...) = AdaptiveBands(basis.model; kwargs...)

function determine_n_bands(bands::AdaptiveBands, occupation::Nothing, eigenvalues, ψ)
    if isnothing(ψ)
        n_bands_compute = bands.n_bands_compute
    else
        n_bands_compute = max(bands.n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
    end
    # Boost number of bands to converge to have more information around in the next step
    # and to thus make a better decision on the number of bands we actually care about.
    n_bands_converge = floor(Int, (bands.n_bands_converge + bands.n_bands_compute) / 2)
    (; n_bands_converge, n_bands_compute)
end
function determine_n_bands(bands::AdaptiveBands, occupation::AbstractVector,
                           eigenvalues::AbstractVector, ψ::AbstractVector)
    # TODO Could return different bands per k-Points

    # Determine number of bands to be actually converged
    n_bands_occ = maximum(occupation) do occk
        something(findlast(onk -> onk ≥ bands.occupation_threshold, occk), length(occk) + 1)
    end
    n_bands_converge = max(bands.n_bands_converge, n_bands_occ)

    # Determine number of bands to be computed
    n_bands_compute_ε = maximum(eigenvalues) do εk
        something(findlast(εnk -> εnk ≥ εk[n_bands_converge] + bands.gap_min, εk),
                  length(εk) + 1)
    end
    n_bands_compute = max(bands.n_bands_compute, n_bands_compute_ε, n_bands_converge + 3)
    if !isnothing(ψ)
        n_bands_compute = max(n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
    end
    (; n_bands_converge, n_bands_compute)
end
