"""
Return the Fourier coefficients for ``δV_{q} ψ_{k-q}`` in the basis of `k`-points.
"""
function compute_δVψk(basis::PlaneWaveBasis, q, ψ, δV)
    equiv_kpoints_minus_q = k_to_kpq_mapping(basis, -q)
    ordering(kdata) = kdata[equiv_kpoints_minus_q]
    # First, express ψ_{[k-q]} in the basis of k-q points…
    ψ_shifted = [shift_kplusq(basis, kpt, -q, ψk)
                 for (kpt, ψk) in zip(basis.kpoints, ordering(ψ))]
    # … then perform the multiplication with δV in real space and get the Fourier
    # coefficients.
    δHψ = zero.(ψ)
    for (ik, kpt) in enumerate(basis.kpoints)
        kcoord_minus_q = kpt.coordinate - q
        kpt_minus_q = build_kpoints(basis, [kcoord_minus_q])[1]
        for n in 1:size(ψ[ik], 2)
            δHψ[ik][:, n] = fft(basis, kpt, ifft(basis, kpt_minus_q,
                                                 ψ_shifted[ik][:, n]) .* δV[:, :, :, kpt.spin])
        end
    end
    δHψ
end

