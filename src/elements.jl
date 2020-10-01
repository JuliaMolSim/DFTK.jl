import PeriodicTable

# Alias to avoid similarity of elements and Element in DFTK module namespace
periodic_table = PeriodicTable.elements

# Data structure for chemical element and the potential model via which
# they interact with electrons. A compensating charge background is
# always assumed. It is assumed that each implementing struct has the
# entity Z (for the nuclear charge).
abstract type Element end

"""Return the total nuclear charge of an atom type"""
charge_nuclear(el::Element) = el.Z

"""Return the total ionic charge of an atom type (nuclear charge - core electrons)"""
charge_ionic(::Element) = error("Implement charge_ionic")

"""Return the number of valence electrons"""
n_elec_valence(el::Element) = charge_ionic(el)

"""Return the number of core electrons"""
n_elec_core(el::Element) = charge_nuclear(el) - charge_ionic(el)

"""Return the initial magnetic moment of the element"""
magnetic_moment(::Element) = zeros(3)

"""Radial local potential, in Fourier space: V(q) = int_{R^3} V(x) e^{-iqx} dx."""
function local_potential_fourier(el::Element, q::AbstractVector)
    local_potential_fourier(el, norm(q))
end

"""Radial local potential, in real space."""
function local_potential_real(el::Element, q::AbstractVector)
    local_potential_real(el, norm(q))
end

_normalise_magnetic_moment(::Nothing) = zeros(3)
_normalise_magnetic_moment(mm::Number) = Float64[0, 0, mm]
_normalise_magnetic_moment(mm::AbstractVector) = Vec3{Float64}(mm)

struct ElementCoulomb <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    magnetic_moment::Vec3{Float64}
    #         Initial electron-spin magnetic moment
    #         for building an SCF guess (in units of μ_B / 2)
end
charge_ionic(el::ElementCoulomb) = el.Z
magnetic_moment(el::ElementCoulomb) = el.magnetic_moment

"""
Element interacting with electrons via a bare Coulomb potential
(for all-electron calculations)
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
function ElementCoulomb(key; magnetic_moment=nothing)
    ElementCoulomb(periodic_table[key].number, Symbol(periodic_table[key].symbol),
                   _normalise_magnetic_moment(magnetic_moment))
end


function local_potential_fourier(el::ElementCoulomb, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    # General atom => Use default Coulomb potential
    # We use int_R^3 1/r e^{-i q⋅x} = 4π / |q|^2
    return -4T(π) * el.Z / q^2
end

local_potential_real(el::ElementCoulomb, r::Real) = -el.Z / r


struct ElementPsp <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    psp     # Pseudopotential data structure
    magnetic_moment::Vec3{Float64}
    #         Initial electron-spin magnetic moment
    #         for building an SCF guess (in units of μ_B / 2)
end

"""
Element interacting with electrons via a pseudopotential model.
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
function ElementPsp(key; psp, magnetic_moment=nothing)
    ElementPsp(periodic_table[key].number,
               Symbol(periodic_table[key].symbol),
               psp,
               _normalise_magnetic_moment(magnetic_moment))
end
charge_ionic(el::ElementPsp) = el.psp.Zion
magnetic_moment(el::ElementPsp) = el.magnetic_moment

function local_potential_fourier(el::ElementPsp, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    eval_psp_local_fourier(el.psp, q)
end

function local_potential_real(el::ElementPsp, r::Real)
    # Use local part of pseudopotential defined in Element object
    return eval_psp_local_real(el.psp, r)
end


struct ElementCohenBergstresser <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    V_sym   # Map |G|^2 (in units of (2π / lattice_constant)^2) to form factors
    lattice_constant  # Lattice constant (in Bohr) which is assumed
end
charge_ionic(el::ElementCohenBergstresser) = 2

"""
Element where the interaction with electrons is modelled
as in [CohenBergstresser1966] (DOI 10.1103/PhysRev.141.789).
Only the homonuclear lattices of the diamond structure
are implemented (i.e. Si, Ge, Sn).

`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
function ElementCohenBergstresser(key; lattice_constant=nothing)
    # Form factors from Cohen-Bergstresser paper Table 2, converted to Bohr
    # Lattice constants from Table 1, converted to Bohr
    data = Dict(:Si => (form_factors=Dict( 3 => -0.21 * units.Ry,
                                           8 =>  0.04 * units.Ry,
                                          11 =>  0.08 * units.Ry),
                        lattice_constant=5.43 * units.Ǎ),
                :Ge => (form_factors=Dict( 3 => -0.23 * units.Ry,
                                           8 =>  0.01 * units.Ry,
                                          11 =>  0.06 * units.Ry),
                        lattice_constant=5.66 * units.Ǎ),
                :Sn => (form_factors=Dict( 3 => -0.20 * units.Ry,
                                           8 =>  0.00 * units.Ry,
                                          11 =>  0.04 * units.Ry),
                        lattice_constant=6.49 * units.Ǎ),
            )

    symbol = Symbol(periodic_table[key].symbol)
    if !(symbol in keys(data))
        error("Cohen-Bergstresser potential not implemented for element " *
              "$(element.symbol).")
    end
    isnothing(lattice_constant) && (lattice_constant = data[symbol].lattice_constant)

    # Unit-cell volume of the primitive lattice (used in DFTK):
    unit_cell_volume = det(lattice_constant / 2 .* [[0 1 1]; [1 0 1]; [1 1 0]])

    # The form factors in the Cohen-Bergstresser paper Table 2 are
    # with respect to normalized planewaves (i.e. not plain Fourier coefficients)
    # and are already symmetrized into a sin-cos basis (see derivation p. 141)
    # => Scale by Ω / 2 to get them into the DFTK convention
    V_sym = Dict(key => value * unit_cell_volume / 2
                 for (key, value) in pairs(data[symbol].form_factors))

    ElementCohenBergstresser(periodic_table[key].number, symbol, V_sym, lattice_constant)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background

    # Get |q|^2 in units of (2π / lattice_constant)^2
    qsq_pi = Int(round(q^2 / (2π / el.lattice_constant)^2, digits=2))
    T(get(el.V_sym, qsq_pi, 0.0))
end
