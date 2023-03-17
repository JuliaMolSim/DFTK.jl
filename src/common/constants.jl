"""The default search location for Pseudopotential data files"""
DFTK_DATADIR = joinpath(dirname(pathof(DFTK)), "..", "data")

# Some useful conversion constants.
dalton_to_amu   = austrip(1u"u")
hartree_to_cm⁻¹ = ustrip(u"cm^-1", 1u"bohr^-1") ./ austrip(1u"c0") / 2π
