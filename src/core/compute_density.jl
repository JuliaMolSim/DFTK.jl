"""
Compute the partial density at the indicated ``k``-Point and return it.
"""
function compute_partial_density(pw, kpt, Ψk, occupation)
    n_states = size(Ψk, 2)
    @assert n_states == length(occupation)

    # Fourier-transform the wave functions to real space
    Ψk_real = similar(Ψk[:, 1], pw.fft_size..., n_states)
    for ist in 1:n_states
        G_to_r!(view(Ψk_real, :, :, :, ist), pw, kpt, Ψk[:, ist])
    end

    # TODO I am not quite sure why this is needed here
    #      maybe this points at an error in the normalisation of the
    #      Fourier transform
    Ψk_real /= sqrt(pw.model.unit_cell_volume)

    # Build the partial density for this k-Point
    ρk_real = similar(Ψk[:, 1], pw.fft_size)
    ρk_real .= 0
    for ist in 1:n_states
        @. @views begin
            ρk_real += occupation[ist] * Ψk_real[:, :, :, ist] * conj(Ψk_real[:, :, :, ist])
        end
    end
    Ψk_real = nothing

    # Check sanity of the density (real, positive and normalized)
    T = real(eltype(ρk_real))
    if maximum(imag(ρk_real)) > 100 * eps(T)
        @warn "Large norm(imag(ρ))" norm_imag=maximum(imag(ρk_real))
    end
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
    compute_density(pw::PlaneWaveModel, Psi::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `Psi` discretised on the plane-wave grid `pw`,
where the individual k-Points are occupied according to `occupation`. `Psi` should
be one coefficient matrix per k-Point.
"""
function compute_density(pw::PlaneWaveModel, Psi::AbstractVector, occupation::AbstractVector)
    n_k = length(pw.kpoints)
    @assert n_k == length(Psi)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(pw.kpoints[ik].basis) == size(Psi[ik], 1)
        @assert length(occupation[ik]) == size(Psi[ik], 2)
    end
    @assert n_k > 0

    ρ = similar(Psi[1][:, 1], pw.fft_size)
    ρ .= 0
    ρ_count = 0
    for (ik, kpt) in enumerate(pw.kpoints)
        ρ_k = compute_partial_density(pw, kpt, Psi[ik], occupation[ik])
        for (S, τ) in pw.ksymops[ik]
            ρ_count += 1
            # TODO If τ == [0,0,0] and if S == id
            #      this routine can be simplified or even skipped

            # Transform ρ_k -> to the partial density at S * k
            for (ig, G) in enumerate(basis_Cρ(pw))
                igired = index_Cρ(pw, Vec3{Int}(inv(S) * G))
                if igired !== nothing
                    ρ[ig] += cis(2π * dot(G, τ)) * ρ_k[igired]
                end
            end
        end
    end

    ρ / ρ_count
end
