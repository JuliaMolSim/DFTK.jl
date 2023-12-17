module DFTKWriteVTKExt
using WriteVTK
using DFTK
using DFTK: gather_kpts

function DFTK.save_scfres(::Val{:vts}, filename::AbstractString, scfres::NamedTuple;
                          save_ψ=false, extra_data=Dict{String,Any}())
    # Initialise the grid and open the file on master
    vtkfile = nothing
    if mpi_master()
        grid = zeros(3, scfres.basis.fft_size...)
        for (idcs, r) in zip(CartesianIndices(scfres.basis.fft_size),
                             r_vectors_cart(scfres.basis))
            grid[:, Tuple(idcs)...] = r
        end
        vtkfile = WriteVTK.vtk_grid(filename, grid)
    end

    if save_ψ
        bands = gather_kpts(scfres.ψ, scfres.basis)
        basis = gather_kpts(scfres.basis)
        if mpi_master()
            for ik = 1:length(basis.kgrid), n = 1:size(bands[ik], 2)
                ψnk_real = ifft(basis, basis.kpoints[ik], bands[ik][:, n])
                prefix = @sprintf "ψ_k%03i_n%03i" ik n
                vtkfile["$(prefix)_real"] = real.(ψnk_real)
                vtkfile["$(prefix)_imag"] = imag.(ψnk_real)
            end
        end
    end

    eigenvalues = gather_kpts(scfres.eigenvalues, scfres.basis)
    occupation  = gather_kpts(scfres.occupation,  scfres.basis)
    if mpi_master()
        ρtotal, ρspin = total_density(scfres.ρ), spin_density(scfres.ρ)
        vtkfile["ρtotal"] = ρtotal
        if !isnothing(ρspin)
            vtkfile["ρspin"]  = ρspin
        end

        for key in keys(scfres.energies)
            vtkfile["energy_$(key)"] = scfres.energies[key]
        end
        vtkfile["εF"] = scfres.εF
        vtkfile["eigenvalues"] = reduce(hcat, scfres.eigenvalues)
        vtkfile["occupation"]  = reduce(hcat, scfres.occupation)

        for (key, value) in pairs(extra_data)
            vtkfile[key] = value
        end

        WriteVTK.vtk_save(vtkfile)
    end

    MPI.Barrier(MPI.COMM_WORLD)
    nothing
end

end
