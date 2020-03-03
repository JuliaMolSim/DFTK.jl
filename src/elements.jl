import PeriodicTable

# Alias to avoid similarity of elements and Element in DFTK module namespace
periodic_table = PeriodicTable.elements

# Data structure for chemical element and the potential model via which
# they interact with electrons. A compensating charge background is
# always assumed.

abstract type AbstractElement end

"""Return the total nuclear charge of an atom type"""
charge_nuclear(el::AbstractElement) = el.Z

"""Return the total ionic charge of an atom type (nuclear charge - core electrons)"""
charge_ionic(el::AbstractElement) = error("Implement charge_ionic")

"""Return the number of valence electrons"""
n_elec_valence(el::AbstractElement) = charge_ionic(el)

"""Return the number of core electrons"""
n_elec_core(el::AbstractElement) = charge_nuclear(el) - charge_ionic(el)

function local_potential_fourier(el::AbstractElement, q::AbstractVector)
    local_potential_fourier(el, norm(q))
end

function local_potential_real(el::AbstractElement, q::AbstractVector)
    local_potential_real(el, norm(q))
end


"""
Element interacting with electrons via a bare Coulomb potential
(for all-electron calculations)
"""
struct ElementAllElectron <: AbstractElement
    Z::Int  # Nuclear charge
    symbol  # Element symbol
end
charge_ionic(el::ElementAllElectron) = el.Z

function ElementAllElectron(key)
    ElementAllElectron(periodic_table[key].number, periodic_table[key].symbol)
end


"""Radial local potential, in Fourier space: V(q) = int_{R^3} V(x) e^{-iqx} dx."""
function local_potential_fourier(el::ElementAllElectron, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    # General atom => Use default Coulomb potential
    # We use int_R^3 1/r e^{-i q⋅x} = 4π / |q|^2
    return -4T(π) * charge_nuclear(el) / q^2
end

"""Radial local potential, in real space."""
local_potential_real(el::ElementAllElectron, r::Real) = 1 / r


"""
Element interacting with electrons via a pseudopotential model
"""
struct ElementPsp <: AbstractElement
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    psp     # Pseudopotential data structure
end
ElementPsp(key; psp) = ElementPsp(periodic_table[key].number, periodic_table[key].symbol, psp)
charge_ionic(el::ElementPsp) = el.psp.Zion

function local_potential_fourier(el::ElementPsp, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    eval_psp_local_fourier(el.psp, q)
end

function local_potential_real(el::ElementPsp, r::Real)
    # Use local part of pseudopotential defined in Element object
    return eval_psp_local_real(el.psp, r)
end


"""
Element where the interaction with electrons is modelled
as in [CohenBergstresser1966] (DOI 10.1103/PhysRev.141.789)

Only the homonuclear lattices of the diamond structure
are implemented (i.e. Si, Ge, Sn).
"""
struct ElementCohenBergstresser <: AbstractElement
    Z::Int  # Nuclear charge
    symbol  # Element symbol
    V_sym   # Map |G|^2 (in units of (2π / lattice_constant)^2) to form factors
    lattice_constant  # Lattice constant (in Bohr) which is assumed
end
charge_ionic(el::ElementCohenBergstresser) = 2

function ElementCohenBergstresser(key, lattice_constant=nothing)
    # Form factors from Cohen-Bergstresser paper Table 2, converted to Bohr
    # Lattice constants from Table 1, converted to Bohr
    V_sym_paper = Dict{Int64,Float64}()
    if Z == 14      # Si
        V_sym_paper[3]  = -0.21 * units.Ry
        V_sym_paper[8]  =  0.04 * units.Ry
        V_sym_paper[11] =  0.08 * units.Ry
        isnothing(lattice_constant) && (lattice_constant = 5.43 * units.Ǎ)
    elseif Z == 32  # Ge
        V_sym_paper[3]  = -0.23 * units.Ry
        V_sym_paper[8]  =  0.01 * units.Ry
        V_sym_paper[11] =  0.06 * units.Ry
        isnothing(lattice_constant) && (lattice_constant = 5.66 * units.Ǎ)
    elseif Z == 50  # Sn
        V_sym_paper[3]  = -0.20 * units.Ry
        V_sym_paper[8]  =  0.00 * units.Ry
        V_sym_paper[11] =  0.04 * units.Ry
        isnothing(lattice_constant) && (lattice_constant = 6.49 * units.Ǎ)
    else
        error("Cohen-Bergstresser potential for Z == $Z not implemented.")
    end

    # Unit-cell volume of the primitive lattice (used in DFTK):
    unit_cell_volume = det(a / 2 .* [[0 1 1]; [1 0 1]; [1 1 0]])

    # The form factors in the Cohen-Bergstresser paper Table 2 are
    # with respect to non-normalised planewaves and are already
    # symmetrised into a sin-cos basis (see derivation p. 141)
    # => Scale by Ω / 2
    V_sym = Dict(key => value * unit_cell_volume / 2
                 for (key, value) in pairs(V_sym_paper))

    ElementCohenBergstresser(periodic_table[key].number, periodic_table[key].symbol,
                             V_sym, lattice_constant)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background

    # Get |G|^2 in units of (2π / lattice_constant)^2
    Gsq_pi = Int(round((q / (2π / el.lattice_constant))^2, digits=12))
    T(get(el.V_sym, Gsq_pi, 0.0))
end
