"""
    compute_density(pw::PlaneWaveBasis, Psi::AbstractVector, occupation::AbstractVector;
                    tolerance_orthonormality)

Compute the density for a wave function `Psi` discretised on the plane-wave grid `pw`,
where the individual k-Points are occupied according to `occupation`. `Psi` should
be one coefficient matrix per k-Point. If `tolerance_orthonormality` is ≥ 0, some
orthonormality properties are verified explicitly.
"""
function compute_density(pw::PlaneWaveBasis, Psi::AbstractVector,
                         occupation::AbstractVector; tolerance_orthonormality=-1)
    # TODO This function could be made in-place

    T = eltype(pw.lattice)
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

    ρ_real = similar(Psi[1][:, 1], size(pw.FFT)...)
    ρ_real .= 0
    for ik in 1:n_k
        Ψ_k = Psi[ik]
        weight = pw.kweights[ik]
        n_states = size(Ψ_k, 2)

        # Fourier-transform the wave functions to real space
        Ψ_k_real = similar(ρ_real, size(pw.FFT)..., n_states)
        for istate in 1:n_states
            G_to_r!(pw, Ψ_k[:, istate], view(Ψ_k_real, :, :, :, istate),
                    gcoords=pw.basis_wf[ik])
        end

        # TODO I am not quite sure why this is needed here
        #      maybe this points at an error in the normalisation of the
        #      Fourier transform
        Ψ_k_real /= sqrt(pw.unit_cell_volume)

        if tolerance_orthonormality > 0
            # Check for orthonormality of the Ψ_k_reals
            n_fft = prod(size(pw.FFT))
            Ψ_k_real_mat = reshape(Ψ_k_real, n_fft, n_states)
            Ψ_k_real_overlap = adjoint(Ψ_k_real_mat) * Ψ_k_real_mat
            nondiag = Ψ_k_real_overlap - I * (n_fft / pw.unit_cell_volume)
            @assert maximum(abs.(nondiag)) < max(1000 * eps(T), tolerance_orthonormality)
            # TODO These assertions should go to a test case
        end

        # Add the density from this kpoint
        occ_k = occupation[ik]
        for istate in 1:n_states
            ρ_real .+= (weight * occ_k[istate]
                        * Ψ_k_real[:, :, :, istate] .* conj(Ψ_k_real[:, :, :, istate])
            )
        end
    end

    # Check ρ is real and positive and properly normalized
    @assert maximum(imag(ρ_real)) < 100 * eps(T)
    @assert minimum(real(ρ_real)) ≥ 0

    n_electrons = sum(ρ_real) * pw.unit_cell_volume / prod(size(pw.FFT))
    @assert abs(n_electrons - sum(occupation[1])) <  sqrt(eps(T))

    ρ = similar(Psi[1][:, 1], prod(pw.grid_size))
    r_to_G!(pw, ρ_real, ρ)
    return ρ
end
