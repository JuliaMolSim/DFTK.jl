## TODO switch this to a [:,:,:,α] representation?
@doc raw"""
Computes the *probability* (not charge) current, ``∑_n f_n \Im(ψ_n · ∇ψ_n)``.
"""
function compute_current(basis::PlaneWaveBasis, ψ, occupation)
    @assert length(basis.symmetries) == 1  # TODO lift this
    current = [zeros(eltype(basis), basis.fft_size) for _ = 1:3]
    for (ik, kpt) in enumerate(basis.kpoints)
        for (n, ψnk) in enumerate(eachslice(ψ[ik]; dims=3))
            ψnk_real = ifft(basis, kpt, ψnk)
            for σ = 1:basis.model.n_components
                ψσnk = ψnk[σ, :]
                ψσnk_real = ψnk_real[σ, :, :, :]
                for α = 1:3
                    dαψσnk = [im * q[α] for q in Gplusk_vectors_cart(basis, kpt)] .* ψσnk
                    dαψσnk_real = ifft(basis, kpt, dαψσnk)
                    current[α] .+= @. basis.kweights[ik] *
                                      occupation[ik][n] *
                                      imag(conj(ψσnk_real) * dαψσnk_real)
                end
            end
        end
    end
    mpi_sum!.(current, Ref(basis.comm_kpts))
    current
end
