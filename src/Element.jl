# Chemical element
struct Element
    Znuc::Int
    psp
end

"""
    Element(Znuc; psp=nothing)

- `Znuc`: Nuclear charge (single atom)
- `psp`:  Pseudopotential to be used (or nothing for all-electron treatment)
"""
Element(Znuc; psp=nothing) = Element(Znuc, psp)


"""Return the total nuclear charge of an atom type"""
charge_nuclear(attype::Element) = attype.Znuc

"""Return the total ionic charge of an atom type (nuclear charge - core electrons)"""
charge_ionic(attype::Element) = attype.psp === nothing ? attype.Znuc : attype.psp.Zion::Int

"""Return the number of valence electrons"""
n_elec_valence(attype::Element) = charge_ionic(attype)

"""Return the number of core electrons"""
n_elec_core(attype::Element) = charge_nuclear(attype) - charge_ionic(attype)
