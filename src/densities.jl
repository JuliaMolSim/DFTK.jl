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
    real_checked(ρk_real)
    if all(occupation .> 0)
        minimum(real(ρk_real)) < 0 && @warn("Negative ρ detected",
                                            min_ρ=minimum(real(ρk_real)))
    end
    n_electrons = sum(ρk_real) * basis.model.unit_cell_volume / prod(basis.fft_size)
    if abs(n_electrons - sum(occupation)) > sqrt(eps(T))
        @warn("Mismatch in number of electrons", sum_ρ=n_electrons,
              sum_occupation=sum(occupation))
    end

    # FFT and return
    r_to_G!(ρ, basis, complex(ρk_real))
end


"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density and spin density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-Points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per k-Point. If the `Model` underlying the basis
is not collinear the spin density is `nothing`.
"""
@timing function compute_density(basis::PlaneWaveBasis, ψ, occupation)
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
    ρaccus = [[similar(view(ψ[1], :, 1), basis.fft_size) for iσ in 1:n_spin]
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
        for iσ in 1:n_spin
            ρaccu[iσ] .= 0
        end
        ρ_k = similar(ψ[1][:, 1], basis.fft_size)
        for ik in ikpts
            kpt = basis.kpoints[ik]
            compute_partial_density!(ρ_k, basis, kpt, ψ[ik], occupation[ik])
            lowpass_for_symmetry!(ρ_k, basis)
            # accumulates all the symops of ρ_k into ρaccu
            accumulate_over_symmetries!(ρaccu[kpt.spin], ρ_k, basis, basis.ksymops[ik])
        end
    end

    # Count the number of k-points modulo spin
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    count = mpi_sum(count, basis.comm_kpts)
    ρs = [sum(getindex.(ρaccus, iσ)) / count for iσ in 1:n_spin]
    mpi_sum!.(ρs, Ref(basis.comm_kpts))

    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)
    if basis.model.spin_polarization == :collinear
        ρtot  = from_fourier(basis, ρs[1] + ρs[2])  # up + down
        ρspin = from_fourier(basis, ρs[1] - ρs[2])  # up - down
    else
        ρtot  = from_fourier(basis, ρs[1])
        ρspin = nothing
    end

    ρtot, ρspin
end
