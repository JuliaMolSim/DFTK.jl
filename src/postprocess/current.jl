## TODO switch this to a [:,:,:,α] representation?
"""
Computes the *probability* (not charge) current, ∑ fn Im(ψn* ∇ψn)
"""
function compute_current(basis::PlaneWaveBasis, ψ, occupation)
    @assert all(symop -> length(symop) == 1, basis.ksymops) == 1  # TODO lift this
    current = [zeros(eltype(basis), basis.fft_size) for α = 1:3]
    for (ik, kpt) in enumerate(basis.kpoints)
        for (n, ψnk) in enumerate(eachcol(ψ[ik]))
            ψnk_real = G_to_r(basis, kpt, ψnk)
            for α = 1:3
                dαψnk = [im*(G+kpt.coordinate_cart)[α] for G in G_vectors_cart(kpt)] .* ψnk
                dαψnk_real = G_to_r(basis, kpt, dαψnk)
                current[α] .+= @. basis.kweights[ik] *
                                  occupation[ik][n] *
                                  imag(conj(ψnk_real) * dαψnk_real)
            end
        end
    end
    mpi_sum!.(current, Ref(basis.comm_kpts))
    current
end
