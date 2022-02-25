"""
Returns an array containing the number of unit cell on each axis in the supercell.
"""
function supercell_size(basis::PlaneWaveBasis)
    shift = basis.kshift
    weights_shift = shift .+ eltype(shift).(shift .==0)
    Int64.(basis.kgrid ./ weights_shift)
end

@doc raw"""
Return all ``k+G`` vectors given by `Gplusk_vectors` as a G vector of
the supercell fft_grid in reduced coordinates.
"""
Gplusk_vectors_in_supercell(basis::PlaneWaveBasis, kpt::Kpoint) =
    map(Gpk -> round.(Int64, Gpk .* supercell_size(basis)), Gplusk_vectors(basis, kpt))

@doc raw"""
Construct a PlaneWaveBasis adapted to fft and ifft in the supercell. Its fft_grid contains
all the ``k+G`` vectors of the initial given basis.

This amounts to take a single ``k`` point and multiply each fft_size component by the number
of unit cell in the supercell, which is equal to ``k_grid`` with a correction due to an
eventual ``k_shift``.
"""
function cell_to_supercell(basis::PlaneWaveBasis)
    model = basis.model

    # Compute supercell model and basis parameters
    size_supercell = supercell_size(basis)
    supercell = size_supercell' .* model.lattice
    fft_size_supercell = basis.fft_size .* size_supercell

    # Compute atoms reduced coordinates in the supercell
    atom_supercell = []; nx, ny, nz = size_supercell
    supercell_normalization(coord) = coord ./ [nx, ny, nz]
    for atom in basis.model.atoms
        atom_coords_supercell = vcat([supercell_normalization.(atom[2] .+ Ref([i;j;k]))
                                          for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1]...)
        append!(atom_supercell, [atom[1] => atom_coords_supercell])
    end

    # Assemble new model and new basis
    model_supercell = Model(supercell, atoms=atom_supercell, terms=model.term_types,
                            symmetries = [one(SymOp)])
    PlaneWaveBasis(model_supercell, basis.Ecut, fft_size_supercell,
                   basis.variational, [zeros(Int64, 3)],
                   [[one(SymOp)]], # Single point symmetry
                   [1,1,1], # k_grid = Γ point only
                   zeros(Int64, 3), [one(SymOp)], basis.comm_kpts,
                   )
end

@doc raw"""
Convert an array of format ``[ψ(1), ψ(2), ... ψ(n_kpts)]`` into a 3 dimensional
tensor adapted to ifft in the supercell.
A ψ[ik][:,n] vector is mapped to ψ_supercell[1][:,(num_kpt*(ik-1)+n)] 
"""
function cell_to_supercell(ψ, basis::PlaneWaveBasis{T},
                           basis_supercell::PlaneWaveBasis{T}) where {T<:Real}
    num_kpG = sum(size.(ψ,1)); num_bands = size(ψ[1],2);
    # Maps k+G vector of initial basis to a G vector of basis_supercell
    cell_supercell_mapping(kpt) = index_G_vectors.(basis_supercell,
                Ref(basis_supercell.kpoints[1]), Gplusk_vectors_in_supercell(basis, kpt))
    # Transfer all ψk independantly and return the hcat of all blocs
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
to supercell convention. The parameters to adapt are the following:

 - ``basis_supercell`` is computed by ``cell_to_supercell(basis)``
 - ``ψ_supercell`` have a single component at Γ-point, which is the concatenation of 
   all ``ψ_k``, and each ``ψ_nk_supercell`` is normalized over the supercell
 - occupations ...
 -
 - energies are multiplied by the number of unit cells in the supercell

Other parameters stay untouched.
"""
function cell_to_supercell(scfres::NamedTuple)
    basis = scfres.basis; ψ = scfres.ψ

    # Compute supercell basis, ψ and ρ
    basis_supercell = cell_to_supercell(basis)
    ψ_supercell = [cell_to_supercell(ψ, basis, basis_supercell)]
    occ_supercell = [vcat(scfres.occupation...)]
    ρ_supercell = compute_density(basis_supercell, ψ_supercell, occ_supercell)

    # Supercell Energies
    eigvalues_supercell = [vcat(scfres.eigenvalues...)]
    n_unit_cells = prod(supercell_size(basis))
    energies_supercell = [n_unit_cells*value for (key,value) in scfres.energies.energies]
    E_supercell = Energies(basis.model.term_types, energies_supercell)

    merge(scfres, (;basis=basis_supercell, ψ=ψ_supercell, energies=E_supercell,
                   ρ=ρ_supercell, eigenvalues=eigvalues_supercell,
                   occupation=occ_supercell)
          )
end

# Old cell to supercell mapping
    # num_kpG = sum(size.(ψ,1)); num_bands = size(ψ[1],2)
    # ψ_supercell = zeros(ComplexF64, num_kpG, num_bands)
    # Γ_point = only(basis_supercell.kpoints)
    # for (ik, kpt) in enumerate(basis.kpoints)
    #     id_kpG_supercell = DFTK.index_G_vectors.(basis_supercell, Ref(Γ_point),
    #                                              Gplusk_vectors_in_supercell(basis, kpt))
    #     ψ_supercell[id_kpG_supercell, :] .= hcat(eachcol(scfres.ψ[ik])...)
    # end
    # # Normalize over the supercell
    # ψ_supercell = hcat([ψn_supercell ./ norm(ψn_supercell)
    #                     for ψn_supercell in eachcol(ψ_supercell)]...)
