"""
Returns a vector containing the number of unit cells on each axis in the supercell.
"""
function get_supercell_size(basis::PlaneWaveBasis)
    shift = basis.kshift
    isnothing(shift) && (return basis.kgrid)
    weights_shift = shift .+ eltype(shift).(shift .==0)
    Int64.(basis.kgrid ./ weights_shift)
end

@doc raw"""
Returns ``k+G`` vectors of an given basis as ``G`` vectors of the supercell fft_grid,
in reduced coordinates.
"""
Gplusk_vectors_in_supercell(basis::PlaneWaveBasis, kpt::Kpoint) =
   map(Gpk -> round.(Int64, Gpk .* get_supercell_size(basis)), Gplusk_vectors(basis, kpt))

@doc raw"""
Construct a plane-wave basis whose unit cell is the supercell associated to
an input basis ``kgrid``. All other parameters are modified so that the respective physical
systems associated to both basis are equivalent.
"""
function cell_to_supercell(basis::PlaneWaveBasis)
    model = basis.model

    # Compute supercell model and basis parameters
    supercell_size = get_supercell_size(basis)
    lattice_supercell = supercell_size' .* model.lattice
    fft_size_supercell = basis.fft_size .* supercell_size

    # Compute atoms reduced coordinates in the supercell
    atom_supercell = []; nx, ny, nz = supercell_size
    supercell_normalization(coord) = coord ./ [nx, ny, nz]
    for atom in model.atoms
        atom_coords_supercell = vcat([supercell_normalization.(atom[2] .+ Ref([i;j;k]))
                                          for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1]...)
        append!(atom_supercell, [atom[1] => atom_coords_supercell])
    end

    # Assemble new model and new basis
    model_supercell = Model(lattice_supercell, atoms=atom_supercell,
                            terms=model.term_types, symmetries = false)
    PlaneWaveBasis(model_supercell, basis.Ecut, fft_size_supercell,
                   basis.variational, [zeros(Int64, 3)],
                   [[one(SymOp)]], # Single point symmetry
                   [1,1,1],        # k_grid = Γ point only
                   zeros(Int64, 3), [one(SymOp)], basis.comm_kpts,
                   )
end

@doc raw"""
Re-organize Bloch waves computed in a given basis as Bloch waves of the associated
supercell basis.
The output ``ψ_supercell`` have a single component at ``Γ``-point, such that 
``ψ_supercell[Γ][:,k+n]`` contains ``ψ[k][:,n]``, within normalization on the supercell.
"""
function cell_to_supercell(ψ, basis::PlaneWaveBasis{T},
                           basis_supercell::PlaneWaveBasis{T}) where {T<:Real}
    # Ensure that the basis is unfolded.
    (prod(basis.kgrid) != length(basis.kpoints)) && (error("basis must be unfolded"))

    num_kpG = sum(size.(ψ,1)); num_bands = size(ψ[1],2);
    # Maps k+G vector of initial basis to a G vector of basis_supercell
    cell_supercell_mapping(kpt) = index_G_vectors.(basis_supercell,
                Ref(basis_supercell.kpoints[1]), Gplusk_vectors_in_supercell(basis, kpt))

    # Transfer all ψ[k] independantly and return the hcat of all blocs
    ψ_out_blocs = []
    for (ik, kpt) in enumerate(basis.kpoints)
        ψk_supercell = zeros(ComplexF64, num_kpG, num_bands)
        ψk_supercell[cell_supercell_mapping(kpt), :] .= ψ[ik]
        push!(ψ_out_blocs, ψk_supercell)
    end
    # Note that each column is normalize since each ψ[ik][:,n] is.
    hcat(ψ_out_blocs...)
end

@doc raw"""
Transpose all data from a given self-consistent-field result from unit cell
to supercell conventions. The parameters to adapt are the following:
 - ``basis_supercell`` and ``ψ_supercell`` are computed by the routines above.
 - The supercell occupations vector is the concatenation of all input occupations vectors.
 - The supercell density is computed with supercell occupations and ``ψ_supercell``.
 - Supercell energies are the multiplication of input energies by the number of
   unit cells in the supercell.
Other parameters stay untouched.
"""
function cell_to_supercell(scfres::NamedTuple)
    # Transfer from cell to supercell need unfolded symmetries.
    scfres_unfold = unfold_bz(scfres)
    basis = scfres_unfold.basis; ψ = scfres_unfold.ψ

    # Compute supercell basis, ψ, occupations and ρ
    basis_supercell = cell_to_supercell(basis)
    ψ_supercell = [cell_to_supercell(ψ, basis, basis_supercell)]
    occ_supercell = [vcat(scfres_unfold.occupation...)]
    ρ_supercell = compute_density(basis_supercell, ψ_supercell, occ_supercell)

    # Supercell Energies
    eigvalues_supercell = [vcat(scfres_unfold.eigenvalues...)]
    n_unit_cells = prod(get_supercell_size(basis))
    energies_supercell = [n_unit_cells*value
                          for (key,value) in scfres_unfold.energies.energies]
    E_supercell = Energies(basis.model.term_types, energies_supercell)

    merge(scfres, (;basis=basis_supercell, ψ=ψ_supercell, energies=E_supercell,
                   ρ=ρ_supercell, eigenvalues=eigvalues_supercell,
                   occupation=occ_supercell)
          )
end
