## Densities (and potentials) are represented by arrays
## ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]
#
# TODO - This file needs a bit of cleanup: There is code duplication in
#        compute_density, compute_δρ, compute_kinetic_energy_density
#      - Use symmetrization instead of explicit use of symmetry operators

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
Compute the partial density at the indicated ``k``-Point and return it (in Fourier space).
"""
function compute_partial_density!(ρ, basis, kpt, ψk, occupation)
    @assert length(occupation) == size(ψk, 2)

    # Build the partial density ρk_real for this k-point
    ρk_real = [zeros(eltype(basis), basis.fft_size) for it = 1:Threads.nthreads()]
    ψnk_real = [zeros(complex(eltype(basis)), basis.fft_size) for it = 1:Threads.nthreads()]
    Threads.@threads for n = 1:size(ψk, 2)
        ψnk = @views ψk[:, n]
        tid = Threads.threadid()
        G_to_r!(ψnk_real[tid], basis, kpt, ψnk)
        ρk_real[tid] .+= occupation[n] .* abs2.(ψnk_real[tid])
    end
    for it = 2:Threads.nthreads()
        ρk_real[1] .+= ρk_real[it]
    end
    ρk_real = ρk_real[1]

    # Check sanity of the density (positive and normalized)
    all(occupation .> 0) && _check_positive(ρk_real)
    _check_total_charge(basis.dvol, ρk_real, sum(occupation))

    # FFT and return
    r_to_G!(ρ, basis, ρk_real)
end


"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density and spin density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per ``k``-point. If the `Model` underlying the basis
is not collinear the spin density is `nothing`.
"""
@views @timing function compute_density(basis::PlaneWaveBasis, ψ, occupation)
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    ρ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    ρ .= 0
    ψnk_real = zeros(complex(eltype(basis)), basis.fft_size)
    for ik = 1:length(basis.kpoints)
        kpt = basis.kpoints[ik]
        for n = 1:size(ψ[ik], 2)
            ψnk = @views ψ[ik][:, n]
            G_to_r!(ψnk_real, basis, kpt, ψnk)
            ρ[:, :, :, kpt.spin] .+= occupation[ik][n] .* basis.kweights[ik] .* abs2.(ψnk_real)
        end
    end
    mpi_sum!(ρ, basis.comm_kpts)
    symmetrize_ρ(basis, ρ; basis.symmetries)
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(basis::PlaneWaveBasis, ψ, δψ, occupation, δoccupation=zero.(occupation))
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    δρ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    δρ .= 0
    ψnk_real = zeros(complex(eltype(basis)), basis.fft_size)
    δψnk_real = zeros(complex(eltype(basis)), basis.fft_size)
    for ik = 1:length(basis.kpoints)
        kpt = basis.kpoints[ik]
        for n = 1:size(ψ[ik], 2)
            ψnk = @views ψ[ik][:, n]
            δψnk = @views δψ[ik][:, n]
            G_to_r!(ψnk_real, basis, kpt, ψnk)
            G_to_r!(δψnk_real, basis, kpt, δψnk)
            δρ[:, :, :, kpt.spin] .+= occupation[ik][n] .* basis.kweights[ik] .*
                                        (conj.(ψnk_real) .* δψnk_real .+
                                        conj.(δψnk_real) .* ψnk_real) .+
                                        δoccupation[ik][n] .* basis.kweights[ik] .* abs2.(ψnk_real)
        end
    end
    mpi_sum!(δρ, basis.comm_kpts)
    symmetrize_ρ(basis, δρ; basis.symmetries)
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
