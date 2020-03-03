import PeriodicTable

# Alias to avoid similarity of elements and Element in DFTK module namespace
periodic_table = PeriodicTable.elements

# Data structure for chemical element and the potential model via which
# they interact with electrons. A compensating charge background is
# always assumed.

abstract type Element end

"""Return the total nuclear charge of an atom type"""
charge_nuclear(el::Element) = el.Z

"""Return the total ionic charge of an atom type (nuclear charge - core electrons)"""
charge_ionic(el::Element) = error("Implement charge_ionic")

"""Return the number of valence electrons"""
n_elec_valence(el::Element) = charge_ionic(el)

"""Return the number of core electrons"""
n_elec_core(el::Element) = charge_nuclear(el) - charge_ionic(el)

function local_potential_fourier(el::Element, q::AbstractVector)
    local_potential_fourier(el, norm(q))
end

function local_potential_real(el::Element, q::AbstractVector)
    local_potential_real(el, norm(q))
end

struct ElementAllElectron <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
end
charge_ionic(el::ElementAllElectron) = el.Z

"""
Element interacting with electrons via a bare Coulomb potential
(for all-electron calculations)
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
function ElementAllElectron(key)
    ElementAllElectron(periodic_table[key].number, Symbol(periodic_table[key].symbol))
end


"""Radial local potential, in Fourier space: V(q) = int_{R^3} V(x) e^{-iqx} dx."""
function local_potential_fourier(el::ElementAllElectron, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    # General atom => Use default Coulomb potential
    # We use int_R^3 1/r e^{-i q⋅x} = 4π / |q|^2
    return -4T(π) * el.Z / q^2
end

"""Radial local potential, in real space."""
local_potential_real(el::ElementAllElectron, r::Real) = -el.Z / r


struct ElementPsp <: Element
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    psp     # Pseudopotential data structure
end

"""
Element interacting with electrons via a pseudopotential model.
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or an element name (e.g. `"silicon"`)
"""
ElementPsp(key; psp) = ElementPsp(periodic_table[key].number,
                                  Symbol(periodic_table[key].symbol), psp)
charge_ionic(el::ElementPsp) = el.psp.Zion

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
    element = periodic_table[key]

    # Form factors from Cohen-Bergstresser paper Table 2, converted to Bohr
    # Lattice constants from Table 1, converted to Bohr
    V_sym_paper = Dict{Int64,Float64}()
    if element.symbol == "Si"
        V_sym_paper[3]  = -0.21 * units.Ry
        V_sym_paper[8]  =  0.04 * units.Ry
        V_sym_paper[11] =  0.08 * units.Ry
        isnothing(lattice_constant) && (lattice_constant = 5.43 * units.Ǎ)
    elseif element.symbol == "Ge"
        V_sym_paper[3]  = -0.23 * units.Ry
        V_sym_paper[8]  =  0.01 * units.Ry
        V_sym_paper[11] =  0.06 * units.Ry
        isnothing(lattice_constant) && (lattice_constant = 5.66 * units.Ǎ)
    elseif element.symbol == "Sn"
        V_sym_paper[3]  = -0.20 * units.Ry
        V_sym_paper[8]  =  0.00 * units.Ry
        V_sym_paper[11] =  0.04 * units.Ry
        isnothing(lattice_constant) && (lattice_constant = 6.49 * units.Ǎ)
    else
        error("Cohen-Bergstresser potential not implemented for element " *
              "$(element.symbol).")
    end

    # Unit-cell volume of the primitive lattice (used in DFTK):
    unit_cell_volume = det(lattice_constant / 2 .* [[0 1 1]; [1 0 1]; [1 1 0]])

    # The form factors in the Cohen-Bergstresser paper Table 2 are
    # with respect to non-normalised planewaves and are already
    # symmetrised into a sin-cos basis (see derivation p. 141)
    # => Scale by Ω / 2
    V_sym = Dict(key => value * unit_cell_volume / 2
                 for (key, value) in pairs(V_sym_paper))

    ElementCohenBergstresser(element.number, Symbol(element.symbol),
                             V_sym, lattice_constant)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background

    # Number of digits to keep in rounding (depending on element type)
    digits = 10
    eps(T) > 1e-13 && (digits = 3)
    eps(T) > 1e-6 && (digits = 2)

    # Get |G|^2 in units of (2π / lattice_constant)^2
    Gsq_pi = Int(round(q^2 / (2π / el.lattice_constant)^2, digits=digits))
    T(get(el.V_sym, Gsq_pi, 0.0))
end
