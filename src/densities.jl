# Densities (and potentials) are represented by arrays
# ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]

function _check_positive(ρ)
    minimum(ρ) < 0 && @warn("Negative ρ detected", min_ρ=minimum(ρ))
end
function _check_total_charge(dvol, ρ::AbstractArray{T}, N; tol=T(1e-10)) where {T}
    n_electrons = sum(ρ) * dvol
    if abs(n_electrons - N) > max(sqrt(eps(T)), tol)
        @warn("Mismatch in number of electrons", sum_ρ=n_electrons, N=N)
    end
end

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
    S = promote_type(T, real(eltype(ψ[1])))

    # we split the total iteration range (ik, n) in chunks, and parallelize over them
    mask_occ = map(occk -> findall(isless.(occupation_threshold, occk)), occupation)
    ik_n = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]
    chunk_length = cld(length(ik_n), Threads.nthreads())

    # chunk-local variables
    ρ_chunklocal = Array{S,4}[zeros(S, basis.fft_size..., basis.model.n_spin_components)
                               for _ = 1:Threads.nthreads()]
    ψnk_real_chunklocal = Array{complex(S),3}[zeros(complex(S), basis.fft_size)
                                               for _ = 1:Threads.nthreads()]

    @sync for (ichunk, chunk) in enumerate(Iterators.partition(ik_n, chunk_length))
        Threads.@spawn for (ik, n) in chunk  # spawn a task per chunk
            ψnk_real = ψnk_real_chunklocal[ichunk]
            ρ_loc = ρ_chunklocal[ichunk]

            kpt = basis.kpoints[ik]
            ifft!(ψnk_real, basis, kpt, ψ[ik][:, n])
            ρ_loc[:, :, :, kpt.spin] .+= occupation[ik][n] .* basis.kweights[ik] .* abs2.(ψnk_real)
        end
    end

    ρ = sum(ρ_chunklocal)
    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ; do_lowpass=false)

    _check_positive(ρ)
    n_elec_check = weighted_ksum(basis, sum.(occupation))
    _check_total_charge(basis.dvol, ρ, n_elec_check; tol=occupation_threshold)

    ρ
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(basis::PlaneWaveBasis{T}, ψ, δψ,
                                   occupation, δoccupation=zero.(occupation);
                                   occupation_threshold=zero(T)) where {T}
    ForwardDiff.derivative(zero(T)) do ε
        ψ_ε   = [ψk   .+ ε .* δψk   for (ψk,   δψk)   in zip(ψ, δψ)]
        occ_ε = [occk .+ ε .* δocck for (occk, δocck) in zip(occupation, δoccupation)]
        compute_density(basis, ψ_ε, occ_ε; occupation_threshold)
    end
end

@views @timing function compute_kinetic_energy_density(basis::PlaneWaveBasis, ψ, occupation)
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    τ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    τ .= 0
    dαψnk_real = zeros(complex(eltype(basis)), basis.fft_size)
    for (ik, kpt) in enumerate(basis.kpoints)
        G_plus_k = [[Gk[α] for Gk in Gplusk_vectors_cart(basis, kpt)] for α in 1:3]
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
