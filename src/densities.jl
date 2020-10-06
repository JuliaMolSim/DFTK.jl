"""
Compute the partial density at the indicated ``k``-Point and return it (in Fourier space).
"""
function compute_partial_density(basis, kpt, ψk, occupation)
    @assert length(occupation) == size(ψk, 2)

    # Build the partial density for this k-Point
    ρk_real = similar(ψk[:, 1], basis.fft_size)
    ρk_real .= 0
    for (n, ψnk) in enumerate(eachcol(ψk))
        ψnk_real = G_to_r(basis, kpt, ψnk)
        ρk_real .+= occupation[n] .* abs2.(ψnk_real)
    end

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
    r_to_G(basis, ρk_real)
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

    # Sanity checks
    @assert n_k == length(ψ)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(G_vectors(basis.kpoints[ik])) == size(ψ[ik], 1)
        @assert length(occupation[ik]) == size(ψ[ik], 2)
    end
    @assert n_k > 0

    # Allocate an accumulator for ρ in each thread for each spin component
    ρaccus = [Dict(σ => similar(ψ[1][:, 1], basis.fft_size)
                   for σ in spin_components(basis.model))
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
        for σ in spin_components(basis.model)
            ρaccu[σ] .= 0
        end
        for ik in ikpts
            kpt = basis.kpoints[ik]
            ρ_k = compute_partial_density(basis, kpt, ψ[ik], occupation[ik])
            # accumulates all the symops of ρ_k into ρaccu
            accumulate_over_symmetries!(ρaccu[kpt.spin], ρ_k, basis, basis.ksymops[ik])
        end
    end

    # Count the number of k-points modulo spin
    n_spin = basis.model.n_spin_components
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    ρs = Dict(σ => sum(getindex.(ρaccus, σ)) / count for σ in spin_components(basis.model))

    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)
    if basis.model.spin_polarization == :collinear
        ρtot  = from_fourier(basis, ρs[:up] + ρs[:down])
        ρspin = from_fourier(basis, ρs[:up] - ρs[:down])
    else
        σ     = only(spin_components(basis.model))
        ρtot  = from_fourier(basis, ρs[σ])
        ρspin = nothing
    end

    ρtot, ρspin
end
