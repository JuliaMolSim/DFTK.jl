# Densities (and potentials) are represented by arrays
# ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]

"""
    compute_density(ψ::BlochWaves, occupation::AbstractVector)

Compute the density for a wave function `ψ` discretized on the plane-wave grid `ψ.basis`,
where the individual k-points are occupied according to `occupation`.
`ψ` should contain one coefficient matrix per ``k``-point.
It is possible to ask only for occupations higher than a certain level to be computed by
using an optional `occupation_threshold`. By default all occupation numbers are considered.
"""
# TODO: We reduce all components for the density. Will need to be though again when we merge
# the components and the spins.
@views @timing function compute_density(ψ::BlochWaves{T, Tψ}, occupation;
                                        occupation_threshold=zero(T)) where {T, Tψ}
    S = promote_type(T, real(Tψ))
    # occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]

    basis = ψ.basis
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]
    if all(isempty, mask_occ)  # No non-zero occupations => return zero density
        ρ = zeros_like(basis.G_vectors, S, basis.fft_size..., basis.model.n_spin_components)
    else
        # we split the total iteration range (ik, n) in chunks, and parallelize over them
        ik_n = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]
        chunk_length = cld(length(ik_n), Threads.nthreads())

        # chunk-local variables
        ρ_chunklocal = map(1:Threads.nthreads()) do i
            zeros_like(basis.G_vectors, S, basis.fft_size..., basis.model.n_spin_components)
        end
        ψσnk_real_chunklocal = [zeros_like(basis.G_vectors, complex(S),
                                           basis.model.n_components, basis.fft_size...)
                                for _ = 1:Threads.nthreads()]

        @sync for (ichunk, chunk) in enumerate(Iterators.partition(ik_n, chunk_length))
            Threads.@spawn for (ik, n) in chunk  # spawn a task per chunk
                ρ_loc = ρ_chunklocal[ichunk]
                ψσnk_real = ψσnk_real_chunklocal[ichunk]
                kpt = basis.kpoints[ik]

                ifft!(ψσnk_real, basis, kpt, ψ[ik][:, :, n])
                for σ = 1:basis.model.n_components
                    ρ_loc[:, :, :, kpt.spin] .+= (occupation[ik][n] .* basis.kweights[ik]
                                                  .* abs2.(ψσnk_real[σ, :, :, :]))

                end
                synchronize_device(basis.architecture)
            end
        end

        ρ = sum(ρ_chunklocal)
    end

    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ; do_lowpass=false)

    # There can always be small negative densities, e.g. due to numerical fluctuations
    # in a vacuum region, so put some tolerance even if occupation_threshold == 0
    negtol = max(sqrt(eps(T)), 10occupation_threshold)
    minimum(ρ) < -negtol && @warn("Negative ρ detected", min_ρ=minimum(ρ))

    ρ
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(ψ::BlochWaves{T}, δψ, occupation,
                                   δoccupation=zero.(occupation);
                                   occupation_threshold=zero(T)) where {T}
    ForwardDiff.derivative(zero(T)) do ε
        ψ_ε   = [ψk   .+ ε .* δψk   for (ψk,   δψk)   in zip(ψ, δψ)]
        occ_ε = [occk .+ ε .* δocck for (occk, δocck) in zip(occupation, δoccupation)]
        compute_density(BlochWaves(ψ.basis, ψ_ε), occ_ε; occupation_threshold)
    end
end

@views @timing function compute_kinetic_energy_density(ψ::BlochWaves{T, Tψ},
                                                       occupation) where {T, Tψ}
    basis = ψ.basis
    @assert basis.model.n_components == 1
    TT = promote_type(T, real(Tψ))
    τ = similar(ψ[1], TT, (basis.fft_size..., basis.model.n_spin_components))
    τ .= 0
    dαψnk_real = zeros(complex(T), basis.fft_size)
    for (ik, kpt) in enumerate(basis.kpoints)
        G_plus_k = [[Gk[α] for Gk in Gplusk_vectors_cart(basis, kpt)] for α = 1:3]
        for n = 1:size(ψ[ik], 3), α = 1:3
            ifft!(dαψnk_real, basis, kpt, im .* G_plus_k[α] .* ψ[ik][1, :, n])
            @. τ[:, :, :, kpt.spin] += occupation[ik][n] * basis.kweights[ik] / 2 * abs2(dαψnk_real)
        end
    end
    mpi_sum!(τ, basis.comm_kpts)
    symmetrize_ρ(basis, τ; do_lowpass=false)
end

total_density(ρ) = dropdims(sum(ρ; dims=4); dims=4)
@views function spin_density(ρ)
    if size(ρ, 4) == 2
        ρ[:, :, :, 1] - ρ[:, :, :, 2]
    else
        zero(ρ[:, :, :])
    end
end

function ρ_from_total_and_spin(ρtot, ρspin=nothing)
    if ρspin === nothing
        # Val used to ensure inferability
        cat(ρtot; dims=Val(4))  # copy for consistency with other case
    else
        cat((ρtot .+ ρspin) ./ 2,
            (ρtot .- ρspin) ./ 2; dims=Val(4))
    end
end

function ρ_from_total(basis, ρtot::AbstractArray{T}) where {T}
    if basis.model.spin_polarization in (:none, :spinless)
        ρspin = nothing
    else
        ρspin = zeros_like(basis.G_vectors, T, basis.fft_size...)
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end
