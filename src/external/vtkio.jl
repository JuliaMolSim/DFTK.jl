function save_scfres_master(filename::AbstractString, scfres::NamedTuple, ::Val{:vts};
                            save_ψ=false, extra_data=Dict{String,Any}())
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
    vtkfile = WriteVTK.vtk_grid(filename, grid)

    # Storing the bloch waves
    if save_ψ
        for ik in 1:length(basis.kpoints)
            for iband in 1:size(scfres.ψ[1])[2]
                ψnk_real = G_to_r(basis, basis.kpoints[ik], scfres.ψ[ik][:, iband]; assume_real=Val(true))
                vtkfile["ψ_k$(ik)_band$(iband)_real"] = real.(ψnk_real)
                vtkfile["ψ_k$(ik)_band$(iband)_imag"] = imag.(ψnk_real)
            end
        end
    end

    # Storing the  density in real space
    vtkfile["ρ"] = total_density(scfres.ρ)

    # Storing ρspin if it is present.
    isnothing(spin_density(scfres.ρ))  || (vtkfile["ρspin"] = spin_density(scfres.ρ))

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

    only(WriteVTK.vtk_save(vtkfile))
end
