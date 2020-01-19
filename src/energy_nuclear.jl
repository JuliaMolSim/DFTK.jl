"""
    energy_nuclear_psp_correction(model)

Compute the correction term for properly modelling the interaction of
the pseudopotential core with the compensating background charge in `energy_nuclear_ewald`.
"""
energy_nuclear_psp_correction(model::Model) = energy_nuclear_psp_correction(model.lattice,
                                                                            model.atoms)
function energy_nuclear_psp_correction(lattice, atoms)
    T = eltype(lattice)

    isempty(atoms) && return T(0)

    # Total number of explicitly treated (i.e. valence) electrons
    n_electrons = sum(n_elec_valence(type) for (type, positions) in atoms
                      for pos in positions)

    correction_per_cell = sum(
        length(positions) * eval_psp_energy_correction(T, type.psp, n_electrons)
        for (type, positions) in atoms
        if type.psp !== nothing
    )

    correction_per_cell / abs(det(lattice))
end

"""
    energy_nuclear_ewald(model; η=nothing)

Compute nuclear-nuclear repulsion energy per unit cell in a uniform background of
negative charge following the Ewald summation procedure. The function assumes all typeies
are modelled as point charges. If pseudopotentials are used, one needs to additionally
compute the `energy_nuclear_psp_correction` to get the correct energy.
"""
energy_nuclear_ewald(model::Model; kwargs...) = energy_nuclear_ewald(model.lattice, model.atoms; kwargs...)
function energy_nuclear_ewald(lattice, atoms; kwargs...)
    charges = [charge_ionic(type) for (type, positions) in atoms for pos in positions]
    positions = [pos for (elem, positions) in atoms for pos in positions]
    energy_ewald(lattice, charges, positions; kwargs...)
end
