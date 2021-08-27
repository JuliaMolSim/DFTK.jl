"""
Pseudopotential correction energy. TODO discuss the need for this.
"""
struct PspCorrection end
(::PspCorrection)(basis) = TermPspCorrection(basis)

struct TermPspCorrection <: Term
    basis::PlaneWaveBasis
    energy::Real  # precomputed energy
end
function TermPspCorrection(basis::PlaneWaveBasis)
    model = basis.model
    if model.n_dim != 3 && any(attype isa ElementPsp for (attype, _) in model.atoms)
        error("The use of pseudopotentials is only sensible for 3D systems.")
    end

    TermPspCorrection(basis, energy_psp_correction(model))
end

function ene_ops(term::TermPspCorrection, ψ, occ; kwargs...)
    ops = [NoopOperator(term.basis, kpoint) for kpoint in term.basis.kpoints]
    (E=term.energy, ops=ops)
end

"""
    energy_psp_correction(model)
Compute the correction term for properly modelling the interaction of the pseudopotential
core with the compensating background charge induced by the `Ewald` term.
"""
energy_psp_correction(model::Model) = energy_psp_correction(model.lattice, model.atoms)
function energy_psp_correction(lattice, atoms)
    T = eltype(lattice)

    # Early return for cases without atoms or psp atoms
    any(attype isa ElementPsp for (attype, _) in atoms) || return T(0)

    # Total number of explicitly treated (i.e. valence) electrons
    n_electrons = sum(length(positions) * n_elec_valence(attype)
                      for (attype, positions) in atoms)

    correction_per_cell = sum(
        length(positions) * eval_psp_energy_correction(T, attype.psp, n_electrons)
        for (attype, positions) in atoms
        if attype isa ElementPsp
    )

    correction_per_cell / compute_unit_cell_volume(lattice)
end
