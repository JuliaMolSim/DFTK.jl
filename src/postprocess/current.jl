## TODO switch this to a [:,:,:,α] representation?
@doc raw"""
Computes the *probability* (not charge) current, ``∑_n f_n \Im(ψ_n · ∇ψ_n)``.
"""
function compute_current(basis::PlaneWaveBasis, ψ, occupation)
    @assert length(basis.symmetries) == 1  # TODO lift this
    current = [zeros(eltype(basis), basis.fft_size) for α = 1:3]
    for (ik, kpt) in enumerate(basis.kpoints)
        for (n, ψnk) in enumerate(eachcol(ψ[ik]))
            ψnk_real = ifft(basis, kpt, ψnk)
            for α = 1:3
                dαψnk = [im * q[α] for q in Gplusk_vectors_cart(basis, kpt)] .* ψnk
                dαψnk_real = ifft(basis, kpt, dαψnk)
                current[α] .+= @. basis.kweights[ik] *
                                  occupation[ik][n] *
                                  imag(conj(ψnk_real) * dαψnk_real)
            end
        end
    end
    mpi_sum!.(current, Ref(basis.comm_kpts))
    current
end
