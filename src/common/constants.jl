# Some useful conversion constants.
const dalton_to_au    = austrip(1u"u")  # unit of atomic mass
const hartree_to_cm⁻¹ = ustrip(u"cm^-1", 1u"bohr^-1") ./ austrip(1u"c0") / 2π
