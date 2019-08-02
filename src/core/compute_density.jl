



# TODO Documentation
function build_partial_density(pw, ik, Ψk, occupation)
    n_states = size(Ψk, 2)
    @assert n_states == length(occupation)

    # Fourier-transform the wave functions to real space
    Ψk_real = similar(Ψk[:, 1], size(pw.FFT)..., n_states)
    for ist in 1:n_states
        G_to_r!(pw, Ψk[:, ist], view(Ψk_real, :, :, :, ist), gcoords=pw.basis_wf[ik])
    end

    # TODO I am not quite sure why this is needed here
    #      maybe this points at an error in the normalisation of the
    #      Fourier transform
    Ψk_real /= sqrt(pw.unit_cell_volume)

    # Build the partial density for this k-Point
    ρk_real = similar(Ψk[:, 1], size(pw.FFT)...)
    ρk_real .= 0
    for ist in 1:n_states
        @. @views begin
            ρk_real += occupation[ist] * Ψk_real[:, :, :, ist] * conj(Ψk_real[:, :, :, ist])
        end
    end
    Ψk_real = nothing

    # Check ρk is real and positive and properly normalized
    T = real(eltype(ρk_real))
    @assert maximum(imag(ρk_real)) < 100 * eps(T)
    @assert minimum(real(ρk_real)) ≥ 0

    n_electrons = sum(ρk_real) * pw.unit_cell_volume / prod(size(pw.FFT))
    @assert abs(n_electrons - sum(occupation)) < sqrt(eps(T))

    ρk = similar(Ψk[:, 1], prod(pw.grid_size))
    r_to_G!(pw, ρk_real, ρk)
    ρk
end


"""
    compute_density(pw::PlaneWaveBasis, Psi::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `Psi` discretised on the plane-wave grid `pw`,
where the individual k-Points are occupied according to `occupation`. `Psi` should
be one coefficient matrix per k-Point.
"""
function compute_density(pw::PlaneWaveBasis, Psi::AbstractVector, occupation::AbstractVector)
    n_k = length(pw.kpoints)
    @assert n_k == length(Psi)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(pw.basis_wf[ik]) == size(Psi[ik], 1)
        @assert length(occupation[ik]) == size(Psi[ik], 2)
    end
    @assert n_k > 0

    # TODO Not sure this is reasonable
    @assert all(occupation[ik] == occupation[1] for ik in 1:n_k)

    function getindex_G(grid_size, G)  # This feels a little strange
        start = -ceil.(Int, (grid_size .- 1) ./ 2)
        stop  = floor.(Int, (grid_size .- 1) ./ 2)

        if all(start .<= G .<= stop)
            # With range of valid indices
            strides = [1, grid_size[1], grid_size[1] * grid_size[2]]
            sum((G .+ stop) .* strides) + 1
        else
            return 0  # Outside range of valid indices
        end
    end

    ρ = similar(Psi[1][:, 1], prod(pw.grid_size))
    ρ .= 0
    ρ_count = 0
    for (ik, k) in enumerate(pw.kpoints)
        ρ_k = build_partial_density(pw, ik, Psi[ik], occupation[ik])
        for (S, τ) in pw.ksymops[ik]
            ρ_count += 1
            # TODO If τ == [0,0,0] and if S == id
            #      this routine can be simplified or even skipped

            # Transform ρ_k -> to the partial density at S * k
            for (ig, G) in enumerate(basis_ρ(pw))
                igired = getindex_G(pw.grid_size, Vec3{Int}(inv(S) * G))
                if igired > 0
                    ρ[ig] += cis(2π * dot(G, τ)) * ρ_k[igired]
                end
            end
        end
    end

    ρ / ρ_count
end
