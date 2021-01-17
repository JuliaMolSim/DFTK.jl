import WriteVTK: vtk_grid, vtk_save

# Uses WriteVTK.jl to convert scfres structure to VTk file format

@doc raw"""
    save_scfres(filename, scfres, Val(:vtk))

The function takes in the VTK filename and the scfres structure and stores into a VTK file.

Grid Values:

- ``\rho`` -> Density in real space
- ``\psi \_k(i)\_band(j)\_real`` -> Real values of Bloch waves in real space
- ``\psi \_k(i)\_band(j)\_imag`` -> Imaginary values of Bloch waves in real space
- ``\rho spin`` -> Real value of ρspin are stored if ρspin in present

MetaData:
Energy , EigenValues, Fermi Level and occupation.
"""
function save_scfres(filename::AbstractString, scfres::NamedTuple, format::Val{:vts})
    # Initialzing the VTK Grid
    basis = scfres.basis
    grid = zeros(3, basis.fft_size...)
    for (idcs, r) in zip(CartesianIndices(basis.fft_size), r_vectors_cart(basis))
        grid[:, Tuple(idcs)...] = r
    end
    vtkfile = vtk_grid(filename, grid)
    
    # Storing the Bloch Waves in Real space
    for ik in 1:length(basis.kpoints)
        for iband in 1:size(scfres.ψ[1])[2]
            ψnk_real = G_to_r(basis, basis.kpoints[ik], scfres.ψ[ik][:,iband])
            vtkfile["ψ_k$(ik)_band$(iband)_real"] = real.(ψnk_real)
            vtkfile["ψ_k$(ik)_band$(iband)_imag"] = imag.(ψnk_real)
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

    only(vtk_save(vtkfile))
end
