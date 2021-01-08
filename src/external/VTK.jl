import WriteVTK: vtk_grid, vtk_save

# Uses WriteVTK.jl to convert scfres structure to VTk file format

"""
    save_scfres(filename, scfres)

The function takes in the VTK filename and the scfres structure and stores into a VTK file.

Grid Values:

- ρ -> Density in real space
- ψreal\\_i\\_j -> Real values of Bloch waves in real space, where i and j are Kpoint and EigenVector indexes respectively
- ψimag\\_i\\_j -> Imaginary values of Bloch waves in real space, where i and j are Kpoint and EigenVector indexes respectively
- ρspin -> Real value of ρspin are stored if ρspin in present

MetaData:
Energy , EigenValues, Fermi Level and occupation.
"""
function save_scfres(filename::AbstractString, scfres::NamedTuple)
    # Initialzing the VTK Grid
    basis = scfres.basis
    grid = zeros(3, basis.fft_size...)
    for (idcs, r) in zip(CartesianIndices(basis.fft_size), r_vectors_cart(basis))
        grid[:, Tuple(idcs)...] = r
    end
    vtkfile = vtk_grid(filename, grid)
    
    # Storing the Bloch Waves in Real space
    for i in 1:length(basis.kpoints)
        for j in 1:size(scfres.ψ[1])[2]
            vtkfile["ψ_real_$(i)_$(j)"] = real.(G_to_r(basis, basis.kpoints[i], scfres.ψ[i][:,j]))
            vtkfile["ψ_imag_$(i)_$(j)"] = imag.(G_to_r(basis, basis.kpoints[i], scfres.ψ[i][:,j]))
        end
    end

    # Storing the  density in real space
    vtkfile["ρ"] = scfres.ρ.real
    
    # Storing ρspin if it is present.
    isnothing(scfres.ρspin)  || (vtkfile["ρspin_real"] = scfres.ρspin.real)
    
    # Storing the energy components
    for key in keys(scfres.energies)
        vtkfile["energy_$key"] = scfres.energies[key]
    end
    
    # Storing the Fermi level 
    vtkfile["fermi_level"] =  scfres.εF

    # Storing the EigenValues as a matrix
    vtkfile["eigen_values"] = hcat(scfres.eigenvalues...)

    # Storing the Occupation as a matrix
    vtkfile["occupation"] = hcat(scfres.occupation...)

    vtk_save(vtkfile)[1]
end