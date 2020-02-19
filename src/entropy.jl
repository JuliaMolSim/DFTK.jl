"""
Compute the temperature-dependent electronic entropy. The returned values is just
the entropy (i.e. no scaling by `-temperature` done).
"""
function compute_entropy(basis::PlaneWaveBasis, orben; εF=find_fermi_level(basis, orben),
                 temperature=basis.model.temperature)
    temperature == 0 && return 0.0
    @assert basis.model.spin_polarisation in (:none, :spinless)
    smearing = basis.model.smearing
    filled_occ = filled_occupation(basis.model)
    Sk = sum.(Smearing.entropy.(smearing, (εk .- εF) ./ temperature) for εk in orben)
    sum(basis.kweights .* filled_occ .* Sk)
end
