

# Commonly used constants. The factors convert from the respective unit
# to atomic units
module units
    Ha = 1                   # Hartree -> Hartree
    eV = 0.03674932248       # electron volt -> Hartree
    Ǎ  = 1.8897261254578281  # Ǎngström -> Bohr
    K  = 3.166810508e-6      # Kelvin -> Hartree
    Ry = 0.5                 # Rydberg -> Hartree
end

"""
    unit(symbol)
Get the factor converting from the unit `symbol` to atomic units.
E.g. `unit(:eV)` returns the conversion factor from electron volts to Hartree.
"""
unit_to_au(sym::Symbol) = getfield(units, sym)
