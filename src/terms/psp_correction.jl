"""
Pseudopotential correction energy. TODO discuss the need for this.
"""
struct PspCorrection end
(::PspCorrection)(basis) = TermPspCorrection(basis)

struct TermPspCorrection{T} <: Term
    energy::T  # precomputed energy
end
function TermPspCorrection(basis::PlaneWaveBasis)
    model = basis.model
    if model.n_dim != 3 &&
        any(hasquantity(atom, :non_local_potential) for atom in model.atoms)
        error("The use of pseudopotentials is only sensible for 3D systems.")
    end
    TermPspCorrection(energy_psp_correction(model))
end

function ene_ops(term::TermPspCorrection, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

"""
Compute the correction term for properly modeling the interaction of the pseudopotential
core with the compensating background charge induced by the `Ewald` term.
"""
function energy_psp_correction(model::Model{T}) where {T}
    species = [model.atoms[first(group)] for group in model.atom_groups]
    correction_per_cell = sum(zip(species, map(length, model.atom_groups))) do (el, n)
         n * T(energy_correction(el.potential))
    end
    total_n_elec_valence = sum(n_elec_valence, model.atoms)
    total_n_elec_valence * correction_per_cell / compute_unit_cell_volume(model.lattice)
end
