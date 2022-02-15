## Densities (and potentials) are represented by arrays
## ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]
using ForwardDiff

function _check_positive(ρ)
    minimum(ρ) < 0 && @warn("Negative ρ detected", min_ρ=minimum(ρ))
end
function _check_total_charge(dvol, ρ, N)
    n_electrons = sum(ρ) * dvol
    if abs(n_electrons - N) > sqrt(eps(eltype(ρ)))
        @warn("Mismatch in number of electrons", sum_ρ=n_electrons, N=N)
    end
end

"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per ``k``-point. 
"""
@views @timing function compute_density(basis::PlaneWaveBasis, ψ, occupation)
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    ρ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    ρ .= 0
    ψnk_real = zeros(complex(T), basis.fft_size)
    for ik = 1:length(basis.kpoints)
        kpt = basis.kpoints[ik]
        for n = 1:size(ψ[ik], 2)
            ψnk = @views ψ[ik][:, n]
            G_to_r!(ψnk_real, basis, kpt, ψnk)
            ρ[:, :, :, kpt.spin] .+= occupation[ik][n] .* basis.kweights[ik] .* abs2.(ψnk_real)
        end
    end
    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ; basis.symmetries)
    _check_positive(ρ)
    _check_total_charge(basis.dvol, ρ,
                        sum(basis.kweights[ik] * sum(occupation[ik]) for ik=1:length(basis.kpoints)))
    ρ
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(basis::PlaneWaveBasis, ψ, δψ, occupation, δoccupation=zero.(occupation))
    ForwardDiff.derivative(ε -> compute_density(basis,
                                                [ψ[ik] .+ ε .* δψ[ik] for ik = 1:length(ψ)],
                                                [occupation[ik] .+ ε .* δoccupation[ik] for ik = 1:length(ψ)]),
                           zero(eltype(basis)))
end

@views @timing function compute_kinetic_energy_density(basis::PlaneWaveBasis, ψ, occupation)
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    τ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    τ .= 0
    dαψnk_real = zeros(complex(eltype(basis)), basis.fft_size)
    for ik = 1:length(basis.kpoints)
        kpt = basis.kpoints[ik]
        G_plus_k = [[Gk[α] for Gk in Gplusk_vectors_cart(basis, kpt)] for α in 1:3]
        for n = 1:size(ψ[ik], 2), α = 1:3
            ψnk = @views ψ[ik][:, n]
            G_to_r!(dαψnk_real, basis, kpt, im .* G_plus_k[α] .* ψnk)
            τ[:, :, :, kpt.spin] .+= @. basis.kweights[ik] * occupation[ik][n] / 2 * abs2(dαψnk_real)
        end
    end
    mpi_sum!(τ, basis.comm_kpts)
    symmetrize_ρ(basis, τ; basis.symmetries)
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
