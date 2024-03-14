# Phonons modes are usually given in cm⁻¹.
const hartree_to_cm⁻¹ = ustrip(u"cm^-1", 1u"bohr^-1") ./ austrip(1u"c0") / 2π
