struct Species
    Znuc::Int
    psp
end

"""
    Species(Znuc; psp=nothing)

Structure collecting information about a species in a structure or crystal.
- `Znuc`: Nuclear charge (single atom) or accumulated nuclear change (composition)
- `psp`:  Pseudopotential to be used (or nothing for all-electron treatment)
"""
Species(Znuc; psp=nothing) = Species(Znuc, psp)


"""Return the total nuclear charge of a species"""
charge_nuclear(spec::Species) = spec.Znuc

"""Return the total ionic charge of a species (nuclear charge - core electrons)"""
charge_ionic(spec::Species) = spec.psp === nothing ? spec.Znuc : spec.psp.Zion::Int

"""Return the number of valence electrons of a species"""
n_elec_valence(spec::Species) = charge_ionic(spec)

"""Return the numebr of core electrons of a species"""
n_elec_core(spec::Species) = charge_nuclear(spec) - charge_ionic(spec)
