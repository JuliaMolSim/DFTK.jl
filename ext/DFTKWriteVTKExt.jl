module DFTKWriteVTKExt
using WriteVTK
using DFTK

function DFTK.save_scfres_master(filename::AbstractString, scfres::NamedTuple, ::Val{:vts};
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
        for ik = 1:length(basis.kpoints)
            for iband = 1:size(scfres.ψ[ik], 3)
                ψnk_real = ifft(basis, basis.kpoints[ik], scfres.ψ[ik][:, :, iband])
                for σ = 1:basis.model.n_components
                    vtkfile["σ$(σ)_ψ_k$(ik)_band$(iband)_real"] = real.(ψnk_real[σ, :, :, :])
                    vtkfile["σ$(σ)_ψ_k$(ik)_band$(iband)_imag"] = imag.(ψnk_real[σ, :, :, :])
                end
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

end
