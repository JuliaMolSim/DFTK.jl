import WriteVTK: vtk_grid, vtk_save

@doc raw"""
    save_scfres(filename, scfres, Val(:vtk))

The function takes in the VTK filename and the scfres structure and stores into a VTK file.

Parameters
- `save_ψ`: Store the orbitals or not. By default they are not stored.
- `save_ldos`: Store the LDOS or not. By default the LDOS is computed and stored.

Grid Values:
- ``\rho`` -> Density in real space
- ``\psi \_k(i)\_band(j)\_real`` -> Real values of Bloch waves in real space
- ``\psi \_k(i)\_band(j)\_imag`` -> Imaginary values of Bloch waves in real space
- ``\rho spin`` -> Real value of ρspin are stored if ρspin in present

MetaData:
- energies, eigenvalues, Fermi level and occupations.
"""
function save_scfres_master(filename::AbstractString, scfres::NamedTuple, ::Val{:vts};
                            save_ψ=false, save_ldos=true, extra_data=Dict{String,Any}())
    !mpi_master() && error(
        "This function should only be called on MPI master after the k-point data has " *
        "been gathered with `gather_kpts`."
    )

    # Initialzing the VTK Grid
    basis = scfres.basis
    grid = zeros(3, basis.fft_size...)
    for (idcs, r) in zip(CartesianIndices(basis.fft_size), r_vectors_cart(basis))
        grid[:, Tuple(idcs)...] = r
    end
    vtkfile = vtk_grid(filename, grid)

    # Storing the bloch waves
    if save_ψ
        for ik in 1:length(basis.kpoints)
            for iband in 1:size(scfres.ψ[1])[2]
                ψnk_real = G_to_r(basis, basis.kpoints[ik], scfres.ψ[ik][:, iband])
                vtkfile["ψ_k$(ik)_band$(iband)_real"] = real.(ψnk_real)
                vtkfile["ψ_k$(ik)_band$(iband)_imag"] = imag.(ψnk_real)
            end
        end
    end

    # Storing the  density in real space
    vtkfile["ρ"] = scfres.ρ.real

    # Storing ρspin if it is present.
    isnothing(scfres.ρspin)  || (vtkfile["ρspin"] = scfres.ρspin.real)

    # Storing the energy components
    for key in keys(scfres.energies)
        vtkfile["energy_$(key)"] = scfres.energies[key]
    end

    # Storing the Fermi level
    vtkfile["εF"] = scfres.εF

    # Storing the EigenValues as a matrix
    vtkfile["eigenvalues"] = hcat(scfres.eigenvalues...)

    # Storing the Occupation as a matrix
    vtkfile["occupation"] = hcat(scfres.occupation...)

    for (key, value) in pairs(extra_data)
        vtkfile[key] = value
    end

    if scfres.basis.model.temperature > 0 && save_ldos
        for σ in 1:scfres.basis.model.n_spin_components
            ldos = compute_ldos(scfres.εF, scfres.ham.basis, scfres.eigenvalues,
                                scfres.ψ, spins=[σ])
            vtkfile["LDOS_spin$σ"] = ldos
        end
    end
    only(vtk_save(vtkfile))
end
