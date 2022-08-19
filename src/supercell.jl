@doc raw"""
Construct a plane-wave basis whose unit cell is the supercell associated to
an input basis ``kgrid``. All other parameters are modified so that the respective physical
systems associated to both basis are equivalent.
"""
function cell_to_supercell(basis::PlaneWaveBasis)
    @assert(iszero(basis.kshift))
    model = basis.model

    # Compute supercell model and basis parameters
    supercell_size = Int64.(basis.kgrid) # Renaming for clarity
    lattice_supercell = reduce(hcat, supercell_size .* eachcol(model.lattice))
    fft_size_supercell = basis.fft_size .* supercell_size

    # Compute atoms reduced coordinates in the supercell
    atoms_supercell = eltype(model.atoms)[]
    positions_supercell = eltype(model.positions)[]
    nx, ny, nz = supercell_size

    for (atom, position) in zip(model.atoms, model.positions)
        append!(positions_supercell, [(position .+ [i;j;k]) ./ [nx, ny, nz]
                                      for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1])
        append!(atoms_supercell, vcat([atom for _ in 1:nx*ny*nz]...))
    end
    # Assemble new model and new basis
    model_supercell = Model(lattice_supercell, atoms_supercell, positions_supercell;
                            terms=model.term_types, symmetries=false)
    PlaneWaveBasis(model_supercell, basis.Ecut, fft_size_supercell,
                   basis.variational,
                   [zeros(Float64, 3)],  # kcoords
                   [one(Float64)],       # kweights
                   [1,1,1],              # kgrid = Γ point only
                   zeros(Int64, 3),      # kshift
                   false,                # single point symmetry
                   basis.comm_kpts,
                   )
end

@doc raw"""
Maps all ``k+G`` vectors of an given basis as ``G`` vectors of the supercell basis,
in reduced coordinates.
"""
function Gplusk_vectors_in_supercell(basis::PlaneWaveBasis,
                                     basis_supercell::PlaneWaveBasis, kpt::Kpoint)
    inv_recip_superlattice = compute_inverse_lattice(basis_supercell.model.recip_lattice)
    map(Gpk -> round.(Int64, inv_recip_superlattice*Gpk), Gplusk_vectors_cart(basis, kpt))
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

    num_kpG = sum(size.(ψ,1))
    num_bands = size(ψ[1],2)

    # Maps k+G vector of initial basis to a G vector of basis_supercell
    cell_supercell_mapping(kpt) = index_G_vectors.(basis_supercell,
                                Ref(basis_supercell.kpoints[1]),
                                Gplusk_vectors_in_supercell(basis, basis_supercell, kpt))

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
    basis = scfres_unfold.basis
    ψ = scfres_unfold.ψ

    # Compute supercell basis, ψ, occupations and ρ
    basis_supercell = cell_to_supercell(basis)
    ψ_supercell = [cell_to_supercell(ψ, basis, basis_supercell)]
    occ_supercell = [vcat(scfres_unfold.occupation...)]
    ρ_supercell = compute_density(basis_supercell, ψ_supercell, occ_supercell)

    # Supercell Energies
    eigvalues_supercell = [vcat(scfres_unfold.eigenvalues...)]
    n_unit_cells = prod(Int64.(basis.kgrid))
    energies_supercell = [n_unit_cells*value
                          for (key,value) in scfres_unfold.energies.energies]
    E_supercell = Energies(basis.model.term_types, energies_supercell)

    merge(scfres, (;basis=basis_supercell, ψ=ψ_supercell, energies=E_supercell,
                   ρ=ρ_supercell, eigenvalues=eigvalues_supercell,
                   occupation=occ_supercell)
          )
end
