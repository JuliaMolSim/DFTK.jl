"""
    energy_nuclear_psp_correction(lattice, composition...)

Compute the correction term for properly modelling the interaction of
the pseudopotential core with the compensating background charge in `energy_nuclear_ewald`.
"""
function energy_nuclear_psp_correction(lattice, composition...)
    T = eltype(lattice)

    # Total number of explicitly treated (i.e. valence) electrons
    n_electrons = sum(n_elec_valence(spec) for (spec, positions) in composition
                      for pos in positions)

    correction_per_cell::T = sum(
        length(positions) * eval_psp_energy_correction(spec.psp, n_electrons)
        for (spec, positions) in composition
        if spec.psp !== nothing
    )

    correction_per_cell / det(lattice)
end

"""
    energy_nuclear_ewald(lattice, composition...; η=nothing)

Compute nuclear-nuclear repulsion energy per unit cell in a uniform background of
negative charge following the Ewald summation procedure. The `lattice` should contain
the lattice vectors as columns and `composition` are pairs mapping from a `Species` object
to a list of positions (given as fractional coordinates). The function assumes all species
are modelled as point charges. If pseudopotentials are used, one needs to additionally
compute the `energy_nuclear_psp_correction` to get the correct energy.
"""
function energy_nuclear_ewald(lattice, composition...; η=nothing)
    charges = [charge_ionic(spec) for (spec, positions) in composition for pos in positions]
    positions = [pos for (elem, positions) in composition for pos in positions]
    energy_ewald(lattice, charges, positions; η=η)
end
