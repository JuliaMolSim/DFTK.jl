## Densities (and potentials) are represented by arrays
## ρ[ix,iy,iz,iσ] in real space, where iσ ∈ [1:n_spin_components]

"""
Compute the partial density at the indicated ``k``-Point and return it (in Fourier space).
"""
function compute_partial_density!(ρ, basis, kpt, ψk, occupation)
    @assert length(occupation) == size(ψk, 2)

    # Build the partial density ρk_real for this k-Point
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

    # Check sanity of the density (real, positive and normalized)
    T = real(eltype(ρk_real))
    if all(occupation .> 0)
        minimum(real(ρk_real)) < 0 && @warn("Negative ρ detected",
                                            min_ρ=minimum(real(ρk_real)))
    end
    n_electrons = sum(ρk_real) * basis.dvol
    if abs(n_electrons - sum(occupation)) > sqrt(eps(T))
        @warn("Mismatch in number of electrons", sum_ρ=n_electrons,
              sum_occupation=sum(occupation))
    end

    # FFT and return
    r_to_G!(ρ, basis, ρk_real)
end


"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density and spin density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-Points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per k-Point. If the `Model` underlying the basis
is not collinear the spin density is `nothing`.
"""
@views @timing function compute_density(basis::PlaneWaveBasis, ψ, occupation)
    n_k = length(basis.kpoints)
    n_spin = basis.model.n_spin_components

    # Sanity checks
    @assert n_k == length(ψ)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(G_vectors(basis.kpoints[ik])) == size(ψ[ik], 1)
        @assert length(occupation[ik]) == size(ψ[ik], 2)
    end
    @assert n_k > 0

    # Allocate an accumulator for ρ in each thread for each spin component
    ρaccus = [similar(view(ψ[1], :, 1), (basis.fft_size..., n_spin))
              for ithread in 1:Threads.nthreads()]

    # TODO Better load balancing ... the workload per kpoint depends also on
    #      the number of symmetry operations. We know heuristically that the Gamma
    #      point (first k-Point) has least symmetry operations, so we will put
    #      some extra workload there if things do not break even
    kpt_per_thread = [ifelse(i <= n_k, [i], Vector{Int}()) for i in 1:Threads.nthreads()]
    if n_k >= Threads.nthreads()
        kblock = floor(Int, length(basis.kpoints) / Threads.nthreads())
        kpt_per_thread = [collect(1:length(basis.kpoints) - (Threads.nthreads() - 1) * kblock)]
        for ithread in 2:Threads.nthreads()
            push!(kpt_per_thread, kpt_per_thread[end][end] .+ collect(1:kblock))
        end
        @assert kpt_per_thread[end][end] == length(basis.kpoints)
    end

    Threads.@threads for (ikpts, ρaccu) in collect(zip(kpt_per_thread, ρaccus))
        ρaccu .= 0
        ρ_k = similar(ψ[1][:, 1], basis.fft_size)
        for ik in ikpts
            kpt = basis.kpoints[ik]
            compute_partial_density!(ρ_k, basis, kpt, ψ[ik], occupation[ik])
            lowpass_for_symmetry!(ρ_k, basis)
            # accumulates all the symops of ρ_k into ρaccu
            accumulate_over_symmetries!(ρaccu[:, :, :, kpt.spin], ρ_k, basis, basis.ksymops[ik])
        end
    end

    # Count the number of k-points modulo spin
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    count = mpi_sum(count, basis.comm_kpts)
    ρ = sum(ρaccus) ./ count
    mpi_sum!(ρ, basis.comm_kpts)
    G_to_r(basis, ρ)
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views function compute_δρ(basis::PlaneWaveBasis{T}, ψ, δψ,
                           occupation, δoccupation=zero.(occupation)) where T
    n_spin = basis.model.n_spin_components

    δρ_fourier = zeros(complex(T), basis.fft_size..., n_spin)
    for (ik, kpt) in enumerate(basis.kpoints)
        δρk = zeros(T, basis.fft_size)
        for n = 1:size(ψ[ik], 2)
            ψnk_real = G_to_r(basis, kpt, ψ[ik][:, n])
            δψnk_real = G_to_r(basis, kpt, δψ[ik][:, n])
            δρk .+= (occupation[ik][n] .* (conj.(ψnk_real) .* δψnk_real .+
                                           conj.(δψnk_real) .* ψnk_real) .+
                     δoccupation[ik][n] .* abs2.(ψnk_real))
        end
        δρk_fourier = r_to_G(basis, complex(δρk))
        lowpass_for_symmetry!(δρk_fourier, basis)
        accumulate_over_symmetries!(δρ_fourier[:, :, :, kpt.spin], δρk_fourier,
                                    basis, basis.ksymops[ik])
    end
    mpi_sum!(δρ_fourier, basis.comm_kpts)
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    count = mpi_sum(count, basis.comm_kpts)
    δρ = G_to_r(basis, δρ_fourier) ./ count
    # Check sanity in the density, we should have ∫δρ = 0
    if abs(sum(δρ)) > sqrt(eps(T))
        @warn("Non-neutral δρ", sum_δρ=sum(δρ))
    end
    δρ
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
         # returns a copy for consistency with the other case
        copy(reshape(ρtot, (size(ρtot)..., n_spin)))
    else
        # Val to ensure inferability
        cat((ρtot .+ ρspin) ./ 2,
            (ρtot .- ρspin) ./ 2; dims=Val(4))
    end
end
