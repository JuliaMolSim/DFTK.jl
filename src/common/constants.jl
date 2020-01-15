"""The default search location for Pseudopotential data files"""
DFTK_DATADIR = joinpath(dirname(pathof(DFTK)), "..", "data")

# Commonly used constants. The factors convert from the respective unit to AU
module units
    eV = 0.03674932248  # Hartree
    Ǎ = 1.8897261254578281  # Bohr
end
