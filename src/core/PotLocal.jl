"""
Class for holding the values of a local potential,
like the local part of a pseudopotential
"""
struct PotLocal{T<:AbstractArray}
    basis::PlaneWaveBasis
    values_real::T
end
apply_real!(out, op::PotLocal, in) = (out .= op.values_real .* in)

function update_energies_real!(energies, op::PotLocal, ρ_real)
    pw = op.basis
    dVol = pw.unit_cell_volume / prod(size(pw.FFT))
    energies[:PotLocal] = real(sum(ρ_real .* op.values_real) * dVol)
    energies
end
