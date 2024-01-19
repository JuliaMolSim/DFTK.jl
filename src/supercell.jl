"""
Construct a supercell of size `supercell_size` from a unit cell described by its `lattice`,
`atoms` and their `positions`.
"""
function create_supercell(lattice, atoms, positions, supercell_size)
    lattice_supercell = stack(supercell_size .* eachcol(lattice))

    # Compute atoms reduced coordinates in the supercell
    atoms_supercell = eltype(atoms)[]
    positions_supercell = eltype(positions)[]
    nx, ny, nz = supercell_size

    for (atom, position) in zip(atoms, positions)
        append!(positions_supercell, [(position .+ [i;j;k]) ./ [nx, ny, nz]
                                      for i = 0:nx-1, j = 0:ny-1, k = 0:nz-1])
        append!(atoms_supercell, reduce(vcat, [atom for _ = 1:nx*ny*nz]))
    end

    (; lattice=lattice_supercell, atoms=atoms_supercell, positions=positions_supercell)
end

@doc raw"""
Construct a plane-wave basis whose unit cell is the supercell associated to
an input basis ``k``-grid. All other parameters are modified so that the respective physical
systems associated to both basis are equivalent.
"""
function cell_to_supercell(basis::PlaneWaveBasis{T}) where {T}
    @assert basis.kgrid isa MonkhorstPack
    iszero(basis.kgrid.kshift) || error("Only kshift of 0 implemented.")
    model = basis.model

    # Compute supercell model and basis parameters
    supercell_size = basis.kgrid.kgrid_size
    supercell = create_supercell(model.lattice, model.atoms, model.positions, supercell_size)
    supercell_fft_size = Tuple{Int,Int,Int}(basis.fft_size .* supercell_size)
    if isnothing(model.n_electrons)
        n_electrons_supercell = nothing
    else
        n_electrons_supercell = prod(supercell_size) * model.n_electrons
    end
    supercell_kgrid = MonkhorstPack([1, 1, 1], basis.kgrid.kshift)

    # Assemble new model and new basis
    supercell_model = Model(model; supercell.lattice, supercell.atoms, supercell.positions,
                            n_electrons=n_electrons_supercell, symmetries=false)
    symmetries_respect_rgrid = true
    use_symmetries_for_kpoint_reduction = true
    PlaneWaveBasis(supercell_model, basis.Ecut, supercell_fft_size,
                   basis.variational, supercell_kgrid, symmetries_respect_rgrid,
                   use_symmetries_for_kpoint_reduction,
                   basis.comm_kpts, basis.architecture)
end

@doc raw"""
Maps all ``k+G`` vectors of an given basis as ``G`` vectors of the supercell basis,
in reduced coordinates.
"""
function Gplusk_vectors_in_supercell(basis::PlaneWaveBasis, basis_supercell::PlaneWaveBasis,
                                     kpt::Kpoint)
    inv_recip_superlattice = compute_inverse_lattice(basis_supercell.model.recip_lattice)
    map(Gpk -> round.(Int64, inv_recip_superlattice * Gpk), Gplusk_vectors_cart(basis, kpt))
end

@doc raw"""
Re-organize Bloch waves computed in a given basis as Bloch waves of the associated
supercell basis.
The output `ψ_supercell` have a single component at ``Γ``-point, such that
`ψ_supercell[Γ][:, k+n]` contains `ψ[k][:, n]`, within normalization on the supercell.
"""
function cell_to_supercell(ψ, basis::PlaneWaveBasis{T},
                           basis_supercell::PlaneWaveBasis{T}) where {T <: Real}
    # Ensure that the basis is unfolded.
    length(basis.kgrid) != length(basis.kpoints) && error("basis must be unfolded")

    num_kpG   = sum(size.(ψ, 1))
    num_bands = size(ψ[1], 2)

    # Maps k+G vector of initial basis to a G vector of basis_supercell
    function cell_supercell_mapping(kpt)
        index_G_vectors.(basis_supercell, Ref(basis_supercell.kpoints[1]),
                         Gplusk_vectors_in_supercell(basis, basis_supercell, kpt))
    end

    # Transfer all ψ[k] independantly and return the hcat of all blocs
    ψ_out_blocs = []
    for (ik, kpt) in enumerate(basis.kpoints)
        ψk_supercell = zeros(complex(T), num_kpG, num_bands)
        ψk_supercell[cell_supercell_mapping(kpt), :] .= ψ[ik]
        push!(ψ_out_blocs, ψk_supercell)
    end
    # Note that each column is normalize since each ψ[ik][:,n] is.
    reduce(hcat, ψ_out_blocs)
end

@doc raw"""
Transpose all data from a given self-consistent-field result from unit cell
to supercell conventions. The parameters to adapt are the following:
 - `basis_supercell` and `ψ_supercell` are computed by the routines above.
 - The supercell occupations vector is the concatenation of all input occupations vectors.
 - The supercell density is computed with supercell occupations and `ψ_supercell`.
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
    ψ_supercell     = [cell_to_supercell(ψ, basis, basis_supercell)]
    eigs_supercell  = [reduce(vcat, scfres_unfold.eigenvalues)]
    occ_supercell   = compute_occupation(basis_supercell, eigs_supercell, scfres.εF).occupation
    ρ_supercell     = compute_density(basis_supercell, ψ_supercell, occ_supercell;
                                      scfres.occupation_threshold)

    # Supercell Energies
    Eham_supercell = energy_hamiltonian(basis_supercell, ψ_supercell, occ_supercell;
                                        ρ=ρ_supercell, eigenvalues=eigs_supercell, scfres.εF)

    merge(scfres, (; ham=Eham_supercell.ham, basis=basis_supercell, ψ=ψ_supercell,
                   energies=Eham_supercell.energies, ρ=ρ_supercell,
                   eigenvalues=eigs_supercell, occupation=occ_supercell))
end
