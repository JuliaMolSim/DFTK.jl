"""
Compute the temperature-dependent electronic entropy term `-TS`.
"""
function compute_entropy_term(basis::PlaneWaveBasis, orben; εF=find_fermi_level(basis, orben),
                              temperature=basis.model.temperature)
    temperature == 0 && return 0.0
    @assert basis.model.spin_polarisation in (:none, :spinless)
    smearing = basis.model.smearing
    filled_occ = filled_occupation(basis.model)
    Sk = sum.(Smearing.entropy.(smearing, (εk .- εF) ./ temperature) for εk in orben)
    -temperature * sum(basis.kweights .* filled_occ .* Sk)
end
