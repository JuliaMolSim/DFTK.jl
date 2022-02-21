"""
    Returns an array containing the number of unit cell on each axis in the supercell.
"""
function supercell_size(basis::PlaneWaveBasis)
    shift = basis.kshift
    weights_shift = shift .+ eltype(shift).(shift .==0)
    basis.kgrid ./ weights_shift
end

@doc raw"""
    For a given k point `k` return all `k+G` for `G` in `G_vectors(k)` as a G vector of
    the supercell in reduced coordinates.
"""
function kpG_reduced_supercell(basis::PlaneWaveBasis, kpt::Kpoint)
    nbr_unit_cells =  supercell_size(basis)
    map(Gpk -> round.(Int64, Gpk .* nbr_unit_cells), [G .+ kpt.coordinate
                                             for G in G_vectors(basis, kpt)])
end

@doc raw"""
    Construct a PlaneWaveBasis adapted to fft and ifft in the supercell.
    Its fft_grid contains all the `k+G` vectors of the initial given basis.
    This amounts to take a single `k` point and multiply each fft_size
    component by the number of unit cell in the supercell, which is
    equal to `k_grid` with a correction due to an eventual `k_shift`.
"""
function cell_to_supercell(basis::PlaneWaveBasis)
    size_supercell = supercell_size(basis)
    # New basis and model parameters
    model = basis.model
    supercell = ( model.lattice' .* size_supercell )'
    fft_size_supercell = Int64.(basis.fft_size .* size_supercell)
    
    # Compute atoms coordinates in the supercell
    atom_supercell = []; nx, ny, nz = size_supercell
    for atom in basis.model.atoms
        atom_coords_supercell = vcat([atom[2] .+ Ref(basis.model.lattice*[i;j;k])
                                      for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1]...)
        append!(atom_supercell, [atom[1] => atom_coords_supercell])
    end
        
    # Assemble new model and new basis
    model_supercell = Model(supercell, atoms=atom_supercell)
    PlaneWaveBasis(model_supercell, basis.Ecut, fft_size_supercell,
                   basis.variational, [zeros(Int64, 3)],
                   [[one(SymOp)]], # only works with "unfolded" basis
                   [1,1,1], # k_grid = Γ point only
                   zeros(Int64, 3), basis.symmetries, basis.comm_kpts)
end

"""
    Convert an array of format `[ψ(1), ψ(2), ... ψ(n_kpts)]` into a 3 dimensional
    tensor adapted to ifft in the supercell.
"""
function cell_to_supercell(basis::PlaneWaveBasis, basis_supercell::PlaneWaveBasis,
                           ψ_fourier)
    ψ_fourier_supercell = zeros(ComplexF64, Tuple(basis_supercell.fft_size))
    for (ik, kpt) in enumerate(basis.kpoints)
        ikpG = DFTK.index_G_vectors.(basis_supercell, kpG_reduced_supercell(basis, kpt))
        ψ_fourier_supercell[ikpG] .= ψ_fourier[ik]
    end
    ψ_fourier_supercell
end

function cell_to_supercell(scfres::NamedTuple)
    basis = scfres.basis; ψ = scfres.ψ
    basis_supercell = cell_to_supercell(basis)
    
    # Compute ψ viewed as a single table in the supercell fft_grid
    @show num_kpG = sum(size(ψk,1) for ψk in ψ); num_bands = size(ψ[1],2)
    ψ_supercell = zeros(ComplexF64, num_kpG, num_bands)
    Γ_supercell = only(basis_supercell.kpoints)
    for (ik, kpt) in enumerate(basis.kpoints)
        id_kpG_supercell = DFTK.index_G_vectors.(basis_supercell, Ref(Γ_supercell),
                                                 kpG_reduced_supercell(basis, kpt))
        ψ_supercell[id_kpG_supercell, :] .= hcat(eachcol(scfres.ψ[ik])...)
    end
    # Tmp have to adapt also the density
    scfres_supercell = merge(scfres, (;ψ=[ψ_supercell], basis=basis_supercell))
end
