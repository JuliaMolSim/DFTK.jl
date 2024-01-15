# Densities (and potentials) are represented by arrays
# ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]

"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per ``k``-point.
It is possible to ask only for occupations higher than a certain level to be computed by
using an optional `occupation_threshold`. By default all occupation numbers are considered.
"""
@views @timing function compute_density(basis::PlaneWaveBasis{T}, ψ, occupation;
                                        occupation_threshold=zero(T)) where {T}
    Tρ = promote_type(T, real(eltype(ψ[1])))

    # Occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        (; ρ=zeros_like(basis.G_vectors, Tρ, basis.fft_size..., basis.model.n_spin_components),
         ψnk_real=zeros_like(basis.G_vectors, complex(Tρ), basis.fft_size...))
    end
    # We split the total iteration range (ik, n) in chunks, and parallelize over them.
    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]

    storages = parallel_loop_over_range(allocate_local_storage, range) do storage, kn
        (ik, n) = kn
        kpt = basis.kpoints[ik]

        ifft!(storage.ψnk_real, basis, kpt, ψ[ik][:, n])
        storage.ρ[:, :, :, kpt.spin] .+= (occupation[ik][n] .* basis.kweights[ik]
                                              .* abs2.(storage.ψnk_real))

        synchronize_device(basis.architecture)
    end
    ρ = sum(getfield.(storages, :ρ))

    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ; do_lowpass=false)

    # There can always be small negative densities, e.g. due to numerical fluctuations
    # in a vacuum region, so put some tolerance even if occupation_threshold == 0
    negtol = max(sqrt(eps(T)), 10occupation_threshold)
    minimum(ρ) < -negtol && @warn("Negative ρ detected", min_ρ=minimum(ρ))

    ρ::AbstractArray{Tρ, 4}
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(basis::PlaneWaveBasis{T}, ψ, δψ, occupation,
                                   δoccupation=zero.(occupation);
                                   occupation_threshold=zero(T), q=zero(Vec3{T})) where {T}
    Tψ = promote_type(T, eltype(ψ[1]))
    # δρ is expected to be real when computations are not phonon-related.
    Tδρ = iszero(q) ? real(Tψ) : Tψ
    real_qzero = iszero(q) ? real : identity

    ordering(kdata) = kdata[k_to_equivalent_kpq_permutation(basis, q)]

    # occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        (; δρ=zeros_like(basis.G_vectors, Tδρ, basis.fft_size..., basis.model.n_spin_components),
          ψnk_real=zeros_like(basis.G_vectors, Tψ, basis.fft_size...),
         δψnk_real=zeros_like(basis.G_vectors, Tψ, basis.fft_size...))
    end
    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]

    storages = parallel_loop_over_range(allocate_local_storage, range) do storage, kn
        (ik, n) = kn

        kpt = basis.kpoints[ik]
        ifft!(storage.ψnk_real, basis, kpt, ψ[ik][:, n])
        # We return the δψk in the basis k+q which are associated to a displacement of the ψk.
        kpt_plus_q, δψk_plus_q = kpq_equivalent_blochwave_to_kpq(basis, kpt, q,
                                                                 ordering(δψ)[ik])
        # The perturbation of the density
        #   |ψ_nk|² is 2 ψ_{n,k} * δψ_{n,k+q}.
        ifft!(storage.δψnk_real, basis, kpt_plus_q, δψk_plus_q[:, n])

        storage.δρ[:, :, :, kpt.spin] .+= real_qzero(
            2 .* occupation[ik][n] .* basis.kweights[ik] .* conj(storage.ψnk_real)
                                                         .* storage.δψnk_real
                .+ δoccupation[ik][n] .* basis.kweights[ik] .* abs2.(storage.ψnk_real))

        synchronize_device(basis.architecture)
    end
    δρ = sum(getfield.(storages, :δρ))

    mpi_sum!(δρ, basis.comm_kpts)
    δρ = symmetrize_ρ(basis, δρ; do_lowpass=false)

    δρ
end

@views @timing function compute_kinetic_energy_density(basis::PlaneWaveBasis, ψ, occupation)
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    τ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    τ .= 0
    dαψnk_real = zeros(complex(eltype(basis)), basis.fft_size)
    for (ik, kpt) in enumerate(basis.kpoints)
        G_plus_k = [[p[α] for p in Gplusk_vectors_cart(basis, kpt)] for α = 1:3]
        for n = 1:size(ψ[ik], 2), α = 1:3
            ifft!(dαψnk_real, basis, kpt, im .* G_plus_k[α] .* ψ[ik][:, n])
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
