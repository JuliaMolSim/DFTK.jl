# Routines for interaction with spglib
# Note: spglib/C uses the row-major convention, thus we need to perform transposes
#       between julia and spglib (https://spglib.github.io/spglib/variable.html)
#       In contrast, Spglib.jl follows the spglib/python convention, which is the one used
#       in DFTK. So, when calling Spglib functions, we do not perform transposes.
import Spglib
const SPGLIB = spglib_jll.libsymspg

function spglib_get_error_message()
    error_code = ccall((:spg_get_error_code, SPGLIB), Cint, ())
    return unsafe_string(ccall((:spg_get_error_message, SPGLIB), Cstring, (Cint,), error_code))
end

"""
Convert the DFTK atoms datastructure into a tuple of datastructures for use with spglib.
`positions` contains positions per atom, `numbers` contains the mapping atom
to a unique number for each indistinguishable element, `spins` contains
the ``z``-component of the initial magnetic moment on each atom, `mapping` contains the
mapping of the `numbers` to the element objects in DFTK and `collinear` whether
the atoms mark a case of collinear spin or not. Notice that if `collinear` is false
then `spins` is garbage.
"""
function spglib_atoms(atoms, magnetic_moments)
    n_attypes = isempty(atoms) ? 0 : sum(length(positions) for (typ, positions) in atoms)
    spg_numbers = Vector{Cint}(undef, n_attypes)
    spg_spins = Vector{Cdouble}(undef, n_attypes)
    spg_positions = Matrix{Cdouble}(undef, 3, n_attypes)

    arbitrary_spin = false
    offset = 0
    nextnumber = 1
    mapping = Dict{Int, Any}()
    for (iatom, (el, positions)) in enumerate(atoms)
        mapping[nextnumber] = el

        # Default to zero magnetic moment unless this is a case of collinear magnetism
        for (ipos, pos) in enumerate(positions)
            # assign the same number to all elements with this position
            spg_numbers[offset + ipos] = nextnumber
            spg_positions[:, offset + ipos] .= pos

            if !isempty(magnetic_moments)
                magmom = magnetic_moments[iatom][2][ipos]
                spg_spins[offset + ipos] = magmom[3]
                !iszero(magmom[1:2]) && (arbitrary_spin = true)
            end
        end
        offset += length(positions)
        nextnumber += 1
    end

    collinear = !isempty(magnetic_moments) && !arbitrary_spin && !all(iszero, spg_spins)
    (positions=spg_positions, numbers=spg_numbers, spins=spg_spins,
     mapping=mapping, collinear=collinear)
end

function spglib_cell(lattice, atoms, magnetic_moments)
    # Convert lattice and atoms to spglib and keep the mapping between our atoms
    # and spglibs atoms
    positions, numbers, spins, atommapping, collinear = spglib_atoms(atoms, magnetic_moments)
    (; cell=Spglib.Cell(lattice, positions, numbers, spins), atommapping, collinear)
end


@timing function spglib_get_symmetry(lattice, atoms, magnetic_moments=[]; tol_symmetry=SYMMETRY_TOLERANCE)
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision

    if isempty(atoms)
        # spglib doesn't like no atoms, so we default to
        # no symmetries (even though there are lots)
        return [Mat3{Int}(I)], [Vec3(zeros(3))]
    end

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_positions, spg_numbers, spg_spins, _, collinear = spglib_atoms(atoms, magnetic_moments)

    # Maximal number of symmetry operations spglib searches for
    max_ops = max(384, 50 * length(spg_numbers))
    spg_rotations    = Array{Cint}(undef, 3, 3, max_ops)
    spg_translations = Array{Cdouble}(undef, 3, max_ops)
    if collinear
        spg_equivalent_atoms = Array{Cint}(undef, max_ops)
        spg_n_ops = ccall((:spg_get_symmetry_with_collinear_spin, SPGLIB), Cint,
                          (Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Cint, Ptr{Cdouble},
                           Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cdouble),
                          spg_rotations, spg_translations, spg_equivalent_atoms, max_ops, copy(lattice'),
                          spg_positions, spg_numbers, spg_spins, Cint(length(spg_numbers)), tol_symmetry)
    else
        spg_n_ops = ccall((:spg_get_symmetry, SPGLIB), Cint,
            (Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint, Cdouble),
            spg_rotations, spg_translations, max_ops, copy(lattice'), spg_positions, spg_numbers,
            Cint(length(spg_numbers)), tol_symmetry)
    end

    # If spglib does not find symmetries give an error
    if spg_n_ops == 0
        err_message = spglib_get_error_message()
        error("spglib failed to get the symmetries. Check your lattice, use a " *
              "uniform BZ mesh or disable symmetries. Spglib reported : " * err_message)
    end

    # Note: Transposes are performed to convert between spglib row-major to julia column-major
    Ws = [Mat3{Int}(spg_rotations[:, :, i]') for i in 1:spg_n_ops]
    ws = [rationalize.(Vec3{Float64}(spg_translations[:, i]), tol=tol_symmetry)
               for i in 1:spg_n_ops]

    # Checks: (A W A^{-1}) is unitary
    for W in Ws
        Scart = lattice * W * inv(lattice)  # Form S in cartesian coords
        if maximum(abs, Scart'Scart - I) > tol_symmetry
            error("spglib returned bad symmetries: Non-unitary rotation matrix.")
        end
    end

    # Check (W, w) maps atoms to equivalent atoms in the lattice
    for (W, w) in zip(Ws, ws)
        for (elem, positions) in atoms
            for coord in positions
                diffs = [rationalize.(W * coord + w - pos, tol=5*tol_symmetry)
                         for pos in positions]

                # If all elements of a difference in diffs is integer, then
                # W * coord + w and pos are equivalent lattice positions
                if !any(all(isinteger, d) for d in diffs)
                    error("spglib returned bad symmetries: Cannot map the atom at position " *
                          "$coord to another atom of the same element under the symmetry " *
                          "operation (W, w):\n" *
                          "($W, $w)")
                end
            end
        end
    end

    Ws, ws
end


function spglib_get_stabilized_reciprocal_mesh(kgrid_size, rotations::Vector;
                                               is_shift=Vec3(0, 0, 0),
                                               is_time_reversal=false,
                                               qpoints=[Vec3(0.0, 0.0, 0.0)])
    spg_rotations = cat([copy(Cint.(S')) for S in rotations]..., dims=3)
    nkpt = prod(kgrid_size)
    mapping = Vector{Cint}(undef, nkpt)
    grid_address = Matrix{Cint}(undef, 3, nkpt)

    nrot = length(rotations)
    n_kpts = ccall((:spg_get_stabilized_reciprocal_mesh, SPGLIB), Cint,
      (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}),
      grid_address, mapping, [Cint.(kgrid_size)...], [Cint.(is_shift)...], Cint(is_time_reversal),
      Cint(nrot), spg_rotations, Cint(length(qpoints)), Vec3{Float64}.(qpoints))

    return n_kpts, Int.(mapping), [Vec3{Int}(grid_address[:, i]) for i in 1:nkpt]
end


"""
Returns crystallographic conventional cell according to the International Table of
Crystallography Vol A (ITA) in case `primitive=false`. If `primitive=true`
the primitive lattice is returned in the convention of the reference work of
Cracknell, Davies, Miller, and Love (CDML). Of note this has minor differences to
the primitive setting choice made in the ITA.
"""
function spglib_standardize_cell(lattice::AbstractArray{T}, atoms, magnetic_moments=[];
                                 correct_symmetry=true,
                                 primitive=false,
                                 tol_symmetry=1e-5) where {T}
    # TODO For time-reversal symmetry see the discussion in PR 496.
    #      https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554
    #      Essentially this does not influence the standardisation,
    #      but it only influences the kpath.
    cell, atommapping, _ = spglib_cell(lattice, atoms, magnetic_moments)
    std_cell = Spglib.standardize_cell(cell; to_primitive=primitive,
                                       symprec=tol_symmetry,
                                       no_idealize=!correct_symmetry)

    newatoms = map(unique(std_cell.types)) do iatom
        ipos = findall(isequal(iatom), std_cell.types)
        atommapping[iatom] => Vec3{T}.(eachcol(std_cell.positions[:, ipos]))
    end
    newmagmoms = map(unique(std_cell.types)) do iatom
        ipos = findall(isequal(iatom), std_cell.types)
        atommapping[iatom] => normalize_magnetic_moment.(std_cell.magmoms[ipos])
    end
    (; lattice=Matrix{T}(std_cell.lattice), atoms=newatoms, magnetic_moments=newmagmoms)
end
function spglib_standardize_cell(model::Model; kwargs...)
    # TODO This ignores magnetic moments
    spglib_standardize_cell(model.lattice, model.atoms; kwargs...)
end


function spglib_spacegroup_number(model, magnetic_moments=[]; tol_symmetry=1e-5)
    # Get spacegroup number according to International Tables for Crystallography (ITA)
    # TODO Time-reversal symmetry disabled? (not yet available in DFTK)
    # TODO Are magnetic moments passed?
    cell, _ = spglib_cell(model.lattice, model.atoms, magnetic_moments)
    Spglib.get_spacegroup_number(cell, tol_symmetry)
end
