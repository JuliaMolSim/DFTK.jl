# Uses WriteVTK.jl to convert scfres structure to VTk file format
import WriteVTK: vtk_grid, vtk_save

"""
    save_scfres(VTKFileName, scfres)

The function takes in the VTK filename and the scfres structure and stores into a VTK file.
Grid Values:
ρ_real -> real values of scfers.ρ
ψreal_i_j -> Real values of Bloch waves in real space, where i and j are Kpoint and EigenVector indexes respectively
ρspin -> Real of ρspin are stored if ρspin in present
MetaData:
Energy and occupation.
"""
function save_scfres(VTKFileName::AbstractString, scfres)
    # Initialzing the VTK Grid
    basis = scfres.basis
    basis_r_co = r_vectors_cart(basis)
    gridsize = size(basis_r_co)
    grid = zeros(3, gridsize...)
    for (idcs, r) in zip(CartesianIndices(gridsize), basis_r_co)
        grid[:, Tuple(idcs)...] = r
    end
    vtkfile = vtk_grid(VTKFileName, grid)
    
    # Storing the Bloch Waves in Real space
    for i in 1:length(basis.kpoints)
        for j in 1:7
            vtkfile["ψreal_$(i)_$(j)"] = real.(G_to_r(basis, basis.kpoints[i], scfres.ψ[i][:,j]))
        end
    end

    # Storing the Real component of density
    vtkfile["ρ_real"] = scfres.ρ.real
    
    # Storing ρspin if it is present.
    isnothing(scfres.ρspin)  || (vtkfile["ρspin_real"] = scfres.ρspin.real)
    
    # Storing the energy components
    for i in keys(scfres.energies)
        vtkfile["Energy_$i"] = scfres.energies[i]
    end

    # Storing the Occupation as a matrix
    occupation_mat = hcat(scfres.occupation...)
    vtkfile["Occupation"] = occupation_mat

    out = vtk_save(vtkfile)
end