"""
Computes the kinetic energy density (ked), 1/2 ∑ fn |∇ψn|^2.
"""
function compute_kinetic_energy_density(basis::PlaneWaveBasis, ψ, occupation)
    @assert all(symop -> length(symop) == 1, basis.ksymops) == 1  # TODO lift this
    ked = zeros(eltype(basis), basis.fft_size)
    for (ik, kpt) in enumerate(basis.kpoints)
        for (n, ψnk) in enumerate(eachcol(ψ[ik]))
            for α = 1:3
                dαψnk = [im * q[α] for q in Gplusk_vectors_cart(basis, kpt)] .* ψnk
                dαψnk_real = G_to_r(basis, kpt, dαψnk)
                ked .+= @. 0.5 * basis.kweights[ik] *
                           occupation[ik][n] * 
                           real(conj(dαψnk_real) * dαψnk_real)
            end
        end
    end
    DFTK.mpi_sum!(ked, basis.comm_kpts)
    ked
end
