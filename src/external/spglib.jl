# Routines for interaction with spglib
#
# Note that these routines should be generalised to be compatible with AtomsBase structures
# and not live in DFTK, but we keep them for now.
import Spglib

function spglib_cell(lattice, atom_groups, positions, magnetic_moments)
    magnetic_moments = normalize_magnetic_moment.(magnetic_moments)

    spg_atoms     = Int[]
    spg_magmoms   = Float64[]
    spg_positions = Vector{Float64}[]

    arbitrary_spin = false
    for (igroup, indices) in enumerate(atom_groups), iatom in indices
        push!(spg_atoms, igroup)
        push!(spg_positions, positions[iatom])

        if isempty(magnetic_moments)
            magmom = zeros(3)
        else
            magmom = magnetic_moments[iatom]
            !iszero(magmom[1:2]) && (arbitrary_spin = true)
        end
        push!(spg_magmoms, magmom[3])
    end
    @assert !arbitrary_spin
    Spglib.SpglibCell(lattice, spg_positions, spg_atoms, spg_magmoms)
end
function spglib_cell(system::AbstractSystem)
    parsed = parse_system(system)
    atom_groups = [findall(Ref(pot) .== parsed.atoms) for pot in Set(parsed.atoms)]
    spglib_cell(parsed.lattice, atom_groups, parsed.positions, parsed.magnetic_moments)
end
spglib_cell(model::Model, magnetic_moments) = spglib_cell(atomic_system(model, magnetic_moments))

"""
Returns crystallographic conventional cell according to the International Table of
Crystallography Vol A (ITA) in case `primitive=false`. If `primitive=true`
the primitive lattice is returned in the convention of the reference work of
Cracknell, Davies, Miller, and Love (CDML). Of note this has minor differences to
the primitive setting choice made in the ITA.
"""
function spglib_standardize_cell(lattice::AbstractArray{T}, atom_groups, positions,
                                 magnetic_moments=[];
                                 correct_symmetry=true, primitive=false,
                                 tol_symmetry=SYMMETRY_TOLERANCE) where {T}
    # TODO For time-reversal symmetry see the discussion in PR 496.
    #      https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554
    #      Essentially this does not influence the standardisation,
    #      but it only influences the kpath.
    cell = spglib_cell(lattice, atom_groups, positions, magnetic_moments)
    std_cell = Spglib.standardize_cell(cell, tol_symmetry; to_primitive=primitive,
                                       no_idealize=!correct_symmetry)

    lattice   = Matrix{T}(std_cell.lattice)
    positions = Vec3{T}.(std_cell.positions)
    magnetic_moments = normalize_magnetic_moment.(std_cell.magmoms)
    (; lattice, atom_groups, positions, magnetic_moments)
end
function spglib_standardize_cell(model::Model, magnetic_moments=[]; kwargs...)
    spglib_standardize_cell(model.lattice, model.atom_groups, model.positions,
                            magnetic_moments; kwargs...)
end

function spglib_dataset(system::AbstractSystem; tol_symmetry=SYMMETRY_TOLERANCE)
    # Get spacegroup number according to International Tables for Crystallography (ITA)
    # TODO Time-reversal symmetry disabled? (not yet available in DFTK)
    Spglib.get_dataset(spglib_cell(system), tol_symmetry)
end
