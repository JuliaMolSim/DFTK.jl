"""
Compute the partial density at the indicated ``k``-Point and return it.
"""
function compute_partial_density(pw, kpt, Ψk, occupation)
    @assert length(occupation) == size(Ψk, 2)

    # Build the partial density for this k-Point
    ρk_real = similar(Ψk[:, 1], pw.fft_size)
    ρk_real .= 0
    for (ist, Ψik) in enumerate(eachcol(Ψk))
        Ψik_real = G_to_r(pw, kpt, Ψik)
        ρk_real .+= occupation[ist] .* abs2.(Ψik_real)
    end

    # Check sanity of the density (real, positive and normalized)
    T = real(eltype(ρk_real))
    check_real(ρk_real)
    if all(occupation .> 0)
        minimum(real(ρk_real)) < 0 && @warn("Negative ρ detected",
                                            min_ρ=minimum(real(ρk_real)))
    end
    n_electrons = sum(ρk_real) * pw.model.unit_cell_volume / prod(pw.fft_size)
    if abs(n_electrons - sum(occupation)) > sqrt(eps(T))
        @warn("Mismatch in number of electrons", sum_ρ=n_electrons,
              sum_occupation=sum(occupation))
    end

    # FFT and return
    r_to_G(pw, ρk_real)
end


"""
    compute_density(pw::PlaneWaveBasis, Psi::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `Psi` discretised on the plane-wave grid `pw`,
where the individual k-Points are occupied according to `occupation`. `Psi` should
be one coefficient matrix per k-Point.
"""
function compute_density(pw::PlaneWaveBasis, Psi::AbstractVector{VecT},
                         occupation::AbstractVector) where VecT
    T = eltype(VecT)
    n_k = length(pw.kpoints)

    # Sanity checks
    @assert n_k == length(Psi)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(G_vectors(pw.kpoints[ik])) == size(Psi[ik], 1)
        @assert length(occupation[ik]) == size(Psi[ik], 2)
    end
    @assert n_k > 0

    # Allocate an accumulator for ρ in each thread
    ρaccus = [similar(Psi[1][:, 1], pw.fft_size) for ithread in 1:Threads.nthreads()]

    # TODO Better load balancing ... the workload per kpoint depends also on
    #      the number of symmetry operations. We know heuristically that the Gamma
    #      point (first k-Point) has least symmetry operations, so we will put
    #      some extra workload there if things do not break even
    kpt_per_thread = [ifelse(i <= n_k, [i], Vector{Int}()) for i in 1:Threads.nthreads()]
    if n_k >= Threads.nthreads()
        kblock = floor(Int, length(pw.kpoints) / Threads.nthreads())
        kpt_per_thread = [collect(1:length(pw.kpoints) - (Threads.nthreads() - 1) * kblock)]
        for ithread in 2:Threads.nthreads()
            push!(kpt_per_thread, kpt_per_thread[end][end] .+ collect(1:kblock))
        end
        @assert kpt_per_thread[end][end] == length(pw.kpoints)
    end

    Gs = collect(G_vectors(pw))
    Threads.@threads for (ikpts, ρaccu) in collect(zip(kpt_per_thread, ρaccus))
        ρaccu .= 0
        for ik in ikpts
            ρ_k = compute_partial_density(pw, pw.kpoints[ik], Psi[ik], occupation[ik])
            _symmetrize_ρ!(ρaccu, ρ_k, pw, pw.ksymops[ik], Gs)
        end
    end

    count = sum(length(pw.ksymops[ik]) for ik in 1:length(pw.kpoints))
    from_fourier(pw, sum(ρaccus) / count; assume_real=true)
end

# For a given kpoint, accumulates the symmetrized versions of the
# density ρin into ρout. No normalization is performed
function _symmetrize_ρ!(ρaccu, ρin, basis, ksymops, Gs)
    T = real(eltype(ρin))
    for (S, τ) in ksymops
        invS = Mat3{Int}(inv(S))
        # Common special case, where ρin does not need to be processed
        if iszero(S - I) && iszero(τ)
            ρaccu .+= ρin
            continue
        end

        # Transform ρin -> to the partial density at S * k.
        #
        # Since the eigenfunctions of the Hamiltonian at k and Sk satisfy
        #      u_{Sk}(x) = u_{k}(S^{-1} (x - τ))
        # with Fourier transform
        #      ̂u_{Sk}(G) = e^{-i G \cdot τ} ̂u_k(S^{-1} G)
        # equivalently
        #      ̂ρ_{Sk}(G) = e^{-i G \cdot τ} ̂ρ_k(S^{-1} G)
        for (ig, G) in enumerate(Gs)
            igired = index_G_vectors(basis, invS * G)
            if igired !== nothing
                @inbounds ρaccu[ig] += cis(-2T(π) * dot(G, τ)) * ρin[igired]
            end
        end
    end  # (S, τ)
end


"""
Interpolate a function expressed in a basis `b_in` to a basis `b_out`
This interpolation uses a very basic real-space algorithm, and makes
a DWIM-y attempt to take into account the fact that b_out can be a supercell of b_in
"""
function interpolate_density(ρ_in::RealFourierArray, b_out::PlaneWaveBasis)
    ρ_out = interpolate_density(ρ_in.real, ρ_in.basis.fft_size, b_out.fft_size,
                                ρ_in.basis.model.lattice, b_out.model.lattice)
    from_real(b_out, ρ_out)
end

function interpolate_density(ρ_in::AbstractArray, grid_in, grid_out, lattice_in, lattice_out)
    T = real(eltype(ρ_in))
    # First, build supercell, array of 3 ints
    supercell = zeros(Int, 3)
    for i = 1:3
        if norm(lattice_in[:, i]) == 0
            @assert norm(lattice_out[:, i]) == 0
            supercell[i] = 1
        else
            supercell[i] = round(Int, norm(lattice_out[:, i]) / norm(lattice_in[:, i]))
        end
        if norm(lattice_out[:, i] - supercell[i]*lattice_in[:, i]) > .3*norm(lattice_out[:, i])
            @warn "In direction $i, the output lattice is very different from the input lattice"
        end
    end

    # ρ_in represents a periodic function, on a grid 0, 1/N, ... (N-1)/N
    grid_supercell = grid_in .* supercell
    ρ_in_supercell = similar(ρ_in, (grid_supercell...))
    for i = 1:supercell[1]
        for j = 1:supercell[2]
            for k = 1:supercell[3]
                ρ_in_supercell[
                    1 + (i-1)*grid_in[1] : i*grid_in[1],
                    1 + (j-1)*grid_in[2] : j*grid_in[2],
                    1 + (k-1)*grid_in[3] : k*grid_in[3]] = ρ_in
            end
        end
    end

    # interpolate ρ_in_supercell from grid grid_supercell to grid_out
    axes_in = (range(0, 1, length=grid_supercell[i]+1)[1:end-1] for i=1:3)
    itp = interpolate(ρ_in_supercell, BSpline(Quadratic(Periodic(OnCell()))))
    sitp = scale(itp, axes_in...)
    ρ_interp = extrapolate(sitp, Periodic())
    ρ_out = similar(ρ_in, grid_out)
    for i = 1:grid_out[1]
        for j = 1:grid_out[2]
            for k = 1:grid_out[3]
                ρ_out[i, j, k] = ρ_interp((i-1)/grid_out[1],
                                          (j-1)/grid_out[2],
                                          (k-1)/grid_out[3])
            end
        end
    end

    ρ_out
end
