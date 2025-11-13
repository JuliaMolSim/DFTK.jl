"""
Pseudopotential correction energy. TODO discuss the need for this.
"""
struct PspCorrection end
(::PspCorrection)(basis) = TermPspCorrection(basis)

struct TermPspCorrection{T} <: TermLinear
    energy::T  # precomputed energy
end
function TermPspCorrection(basis::PlaneWaveBasis)
    model = basis.model
    if model.n_dim != 3 && any(attype isa ElementPsp for attype in model.atoms)
        error("The use of pseudopotentials is only sensible for 3D systems.")
    end
    TermPspCorrection(energy_psp_correction(model))
end

function ene_ops(term::TermPspCorrection, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (; E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

"""
Compute the correction term for properly modelling the interaction of the pseudopotential
core with the compensating background charge induced by the `Ewald` term.
"""
function energy_psp_correction(lattice::AbstractMatrix{T}, atoms, atom_groups) where {T}
    correction_per_cell_and_electron = sum(atom_groups) do group
        length(group) * eval_psp_energy_correction(T, atoms[first(group)])::T
    end
    n_electrons::Int = n_electrons_from_atoms(atoms)
    correction_per_cell_and_electron * n_electrons / compute_unit_cell_volume(lattice)
end
function energy_psp_correction(model::Model)
    energy_psp_correction(model.lattice, model.atoms, model.atom_groups)
end
