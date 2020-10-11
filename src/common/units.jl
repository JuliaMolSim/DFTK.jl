using Unitful
import Unitful.uconvert
using UnitfulAtomic


# Commonly used constants. The factors convert from the respective unit
# to atomic units
module units
    const Ha = 1                   # Hartree -> Hartree
    const eV = 0.03674932248       # electron volt -> Hartree
    const Å  = 1.8897261254578281  # Ǎngström -> Bohr
    const K  = 3.166810508e-6      # Kelvin -> Hartree
    const Ry = 0.5                 # Rydberg -> Hartree
    const Ǎ  = Å                   # Ǎngström -> Bohr  (deprecated)
end

"""
    unit_to_ao(symbol)
Get the factor converting from the unit `symbol` to atomic units.
E.g. `unit_to_au(:eV)` returns the conversion factor from electron volts to Hartree.
"""
unit_to_au(symbol::Symbol) = getfield(units, symbol)

# Convert a temperature to an energy via the Boltzmann constant.
function uconvert(a::Unitful.EnergyUnits, x::Unitful.Quantity{T,Unitful.𝚯,U}) where {T <: Real, U <: Unitful.Units}
    uconvert(a, Unitful.k * x)
end

# Convert a value to an energy in atomic units.
# If given a number, assume it is already in atomic units (pass through directly).
function to_energy(x::Unitful.Quantity)::Real
    ustrip(uconvert(UnitfulAtomic.Eh_au, x))
end
to_energy(x::Real)::Real = x
