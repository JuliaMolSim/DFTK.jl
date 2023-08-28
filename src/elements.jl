import PeriodicTable
using AtomsBase
using AtomicPotentials

# Alias to avoid similarity of elements and Element in DFTK module namespace
periodic_table = PeriodicTable.elements

# Data structure for chemical element and the potential model via which they interact with
# electrons.
struct Element{S<:EvaluationSpace,PT<:AtomicPotential{S}}
    Z::Int
    symbol::Symbol
    potential::PT
end
function Element(symbol::Symbol, potential)
    Z = periodic_table[symbol].number
    return Element(Z, symbol, potential)
end

function Base.show(io::IO, el::Element)
    quantities = [key for key in propertynames(el.potential) if hasquantity(el.potential, key)]
    print(io, "Element($(el.symbol), [$(join(String.(quantities), ", "))])")
end

"""Return the total nuclear charge of an atom type"""
nuclear_charge(el::Element) = el.Z

"""Chemical symbol corresponding to an element"""
AtomsBase.atomic_symbol(el::Element) = el.symbol

AtomicPotentials.ionic_charge(el::Element) = ionic_charge(el.potential.local_potential)

AtomicPotentials.hasquantity(el::Element, name) = hasquantity(el.potential, name)

# The preceeding functions are fallback implementations that should be altered as needed.

"""Return the number of valence electrons"""
n_elec_valence(el::Element) = round(Int, ionic_charge(el))

"""Return the number of core electrons"""
n_elec_core(el::Element) = round(Int, nuclear_charge(el) - ionic_charge(el))
