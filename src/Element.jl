# Chemical element
#      should contain field Znuc, nuclear change
#
# By default an Element is a physical atom, attracting Electrons
# via the Coulomb potential. For all elements a compensating charge
# background is assumed.
abstract struct AbstractElement end

"""Return the total nuclear charge of an atom type"""
charge_nuclear(el::AbstractElement) = el.Znuc

"""Return the total ionic charge of an atom type (nuclear charge - core electrons)"""
charge_ionic(el::AbstractElement) = charge_ionic(el)

"""Return the number of valence electrons"""
n_elec_valence(el::AbstractElement) = charge_ionic(el)

"""Return the number of core electrons"""
n_elec_core(el::AbstractElement) = charge_nuclear(el) - charge_ionic(el)

"""Radial local potential, in Fourier space: V(q) = int_{R^3} V(x) e^{-iqx} dx."""
function local_potential_fourier(el::AbstractElement, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    # General atom => Use default Coulomb potential
    # We use int_R^3 1/r e^{-i q⋅x} = 4π / |q|^2
    return -4T(π) * charge_nuclear(el) / q^2
end

"""Radial local potential, in real space."""
local_potential_real(el::AbstractElement, r::Real) = 1 / r

"""
Element interacting with electrons via a bare Coulomb potential
(for all-electron calculations)
"""
struct Element <: AbstractElement
    Znuc::Int
end


"""
Element interacting with electrons via a Pseudopotential
"""
struct PspElement <: AbstractElement
    Znuc::Int
    psp  # Pseudopotential data structure
end
charge_ionic(el::PspElement) = el.psp.Zion

function local_potential_fourier(el::PspElement, q::T) where {T <: Real}
    q == 0 && return zero(T)  # Compensating charge background
    eval_psp_local_fourier(el.psp, q)
end

function local_potential_real(el::PspElement, r::Real)
    # Use local part of pseudopotential defined in Element object
    return eval_psp_local_real(el.psp, r)
end
