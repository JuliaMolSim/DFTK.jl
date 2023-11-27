import PeriodicTable
using AtomsBase

# Alias to avoid similarity of elements and Element in DFTK module namespace
periodic_table = PeriodicTable.elements

# Data structure for chemical element and the potential model via which
# they interact with electrons. A compensating charge background is
# always assumed. It is assumed that each implementing struct
# defines at least the functions `local_potential_fourier` and `local_potential_real`.
# Very likely `charge_nuclear` and `charge_ionic` need to be defined as well.
abstract type Element end

"""Return the total nuclear charge of an atom type"""
charge_nuclear(::Element) = 0

"""Chemical symbol corresponding to an element"""
AtomsBase.atomic_symbol(::Element) = :X

"""Return the total ionic charge of an atom type (nuclear charge - core electrons)"""
charge_ionic(el::Element) = charge_nuclear(el)

"""Return the atomic mass in a.u. of an atom type"""
atomic_mass(el::Element) = ustrip(periodic_table[charge_nuclear(el)].atomic_mass) * dalton_to_au

"""Return the number of valence electrons"""
n_elec_valence(el::Element) = charge_ionic(el)

"""Return the number of core electrons"""
n_elec_core(el::Element) = charge_nuclear(el) - charge_ionic(el)

"""Check presence of model core charge density (non-linear core correction)."""
has_core_density(::Element) = false
# The preceeding functions are fallback implementations that should be altered as needed.

# Fall back to the Gaussian table for Elements without pseudopotentials
function valence_charge_density_fourier(el::Element, q::T)::T where {T <: Real}
    gaussian_valence_charge_density_fourier(el, q)
end

"""Gaussian valence charge density using Abinit's coefficient table, in Fourier space."""
function gaussian_valence_charge_density_fourier(el::Element, q::T)::T where {T <: Real}
    charge_ionic(el) * exp(-(q * atom_decay_length(el))^2)
end

function core_charge_density_fourier(::Element, ::T)::T where {T <: Real}
    error("Abstract elements do not necesesarily provide core charge density.")
end

# Fallback print function:
Base.show(io::IO, el::Element) = print(io, "$(typeof(el))($(atomic_symbol(el)))")


struct ElementCoulomb <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
end
charge_ionic(el::ElementCoulomb)   = el.Z
charge_nuclear(el::ElementCoulomb) = el.Z
AtomsBase.atomic_symbol(el::ElementCoulomb) = el.symbol

"""
Element interacting with electrons via a bare Coulomb potential
(for all-electron calculations)
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
ElementCoulomb(key) = ElementCoulomb(periodic_table[key].number, Symbol(periodic_table[key].symbol))


function local_potential_fourier(el::ElementCoulomb, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    # General atom => Use default Coulomb potential
    # We use int_{R^3} -Z/r e^{-i q⋅x} = -4π Z / |q|^2
    return -4T(π) * el.Z / q^2
end

local_potential_real(el::ElementCoulomb, r::Real) = -el.Z / r


struct ElementPsp <: Element
    Z::Int         # Nuclear charge
    symbol         # Element symbol
    psp            # Pseudopotential data structure
end
function Base.show(io::IO, el::ElementPsp)
    pspid = isempty(el.psp.identifier) ? "custom" : el.psp.identifier
    print(io, "ElementPsp($(el.symbol); psp=\"$pspid\")")
end

"""
Element interacting with electrons via a pseudopotential model.
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
function ElementPsp(key; psp)
    ElementPsp(periodic_table[key].number, Symbol(periodic_table[key].symbol), psp)
end

charge_ionic(el::ElementPsp) = charge_ionic(el.psp)
charge_nuclear(el::ElementPsp) = el.Z
has_core_density(el::ElementPsp) = has_core_density(el.psp)
AtomsBase.atomic_symbol(el::ElementPsp) = el.symbol

function local_potential_fourier(el::ElementPsp, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    eval_psp_local_fourier(el.psp, q)
end

function local_potential_real(el::ElementPsp, r::Real)
    return eval_psp_local_real(el.psp, r)
end

function valence_charge_density_fourier(el::ElementPsp, q::T) where {T <: Real}
    if has_valence_density(el.psp)
        eval_psp_density_valence_fourier(el.psp, q)
    else
        gaussian_valence_charge_density_fourier(el, q)
    end
end

function core_charge_density_fourier(el::ElementPsp, q::T) where {T <: Real}
    eval_psp_density_core_fourier(el.psp, q)
end

struct ElementCohenBergstresser <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    V_sym   # Map |G|^2 (in units of (2π / lattice_constant)^2) to form factors
    lattice_constant  # Lattice constant (in Bohr) which is assumed
end
charge_ionic(el::ElementCohenBergstresser)   = 4
charge_nuclear(el::ElementCohenBergstresser) = el.Z
AtomsBase.atomic_symbol(el::ElementCohenBergstresser) = el.symbol

"""
Element where the interaction with electrons is modelled
as in [CohenBergstresser1966](https://doi.org/10.1103/PhysRev.141.789).
Only the homonuclear lattices of the diamond structure
are implemented (i.e. Si, Ge, Sn).

`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
function ElementCohenBergstresser(key; lattice_constant=nothing)
    # Form factors from Cohen-Bergstresser paper Table 2
    # Lattice constants from Table 1
    data = Dict(:Si => (; form_factors=Dict( 3 => -0.21u"Ry",
                                             8 =>  0.04u"Ry",
                                            11 =>  0.08u"Ry"),
                        lattice_constant=5.43u"Å"),
                :Ge => (; form_factors=Dict( 3 => -0.23u"Ry",
                                             8 =>  0.01u"Ry",
                                            11 =>  0.06u"Ry"),
                        lattice_constant=5.66u"Å"),
                :Sn => (; form_factors=Dict( 3 => -0.20u"Ry",
                                             8 =>  0.00u"Ry",
                                            11 =>  0.04u"Ry"),
                        lattice_constant=6.49u"Å"),
            )

    symbol = Symbol(periodic_table[key].symbol)
    if !(symbol in keys(data))
        error("Cohen-Bergstresser potential not implemented for element $symbol.")
    end
    isnothing(lattice_constant) && (lattice_constant = data[symbol].lattice_constant)
    lattice_constant = austrip(lattice_constant)

    # Unit-cell volume of the primitive lattice (used in DFTK):
    unit_cell_volume = det(lattice_constant / 2 .* [[0 1 1]; [1 0 1]; [1 1 0]])

    # The form factors in the Cohen-Bergstresser paper Table 2 are
    # with respect to normalized planewaves (i.e. not plain Fourier coefficients)
    # and are already symmetrized into a sin-cos basis (see derivation p. 141)
    # => Scale by Ω / 2 to get them into the DFTK convention
    V_sym = Dict(key => austrip(value) * unit_cell_volume / 2
                 for (key, value) in pairs(data[symbol].form_factors))

    ElementCohenBergstresser(periodic_table[key].number, symbol, V_sym, lattice_constant)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background

    # Get |q|^2 in units of (2π / lattice_constant)^2
    qsq_pi = Int(round(q^2 / (2π / el.lattice_constant)^2, digits=2))
    T(get(el.V_sym, qsq_pi, 0.0))
end


struct ElementGaussian <: Element
    α                               # Prefactor
    L                               # Width of the Gaussian nucleus
    symbol::Symbol  # Element symbol
end
AtomsBase.atomic_symbol(el::ElementGaussian) = el.symbol

"""
Element interacting with electrons via a Gaussian potential.
Symbol is non-mandatory.
"""
function ElementGaussian(α, L; symbol=:X)
    ElementGaussian(α, L, symbol)
end
function local_potential_real(el::ElementGaussian, r)
    -el.α / (√(2π) * el.L) * exp(- (r / el.L)^2 / 2)
end
function local_potential_fourier(el::ElementGaussian, q::Real)
    -el.α * exp(- (q * el.L)^2 / 2)  # = ∫_ℝ³ V(x) exp(-ix⋅q) dx
end
