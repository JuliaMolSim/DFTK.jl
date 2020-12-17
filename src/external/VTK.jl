# Uses WriteVTK.jl to convert various structures and matrices to VTk file format
using WriteVTK

"""
    EnergyToVTK(VTKFileName::String, energy::Energies{::Number})

Stores the Energy structure in a VTK image file and returns the name of the file.
"""
function EnergyToVTK(VTKFileName::String, energy::Energies{<:Number})

    vtkfile = vtk_grid(VTKFileName, 1, 1)
    for i in keys(energy)
        vtkfile[i] = energy.energies[i]
    end
    out = vtk_save(vtkfile)
    return out[1]
end

"""
    DensityToVtk(VTKFileName::String, ρ)

Stores the energy density in a VTK image file and returns the name of the file.
The density is stored in 3 submatrices : Real, Real components of Fourier, Imaginary
components of Fourier.
"""
function DensityToVtk(VTKFileName::String, ρ::RealFourierArray{T, <: AbstractArray{T, 3}, 
                                             <: AbstractArray{Complex{T}, 3}}) where T<:Real

    out = vtk_write_array(VTKFileName,(ρ.real, Base.getproperty.(ρ.fourier, :re),
                          Base.getproperty.(ρ.fourier, :im)), 
                            ("Real", "Fourier Real", "Fourier Complex"))
    return out[1]
end

"""
    EigenValuesToVTK(VTKFileName::String, Eigenvalues)

Stores the eigen values in a VTK image file and returns the name of the file.
"""
function EigenValuesToVTK(VTKFileName::String, Eigenvalues::Array{Array{T,1},1}) where T<:Real
    eigen_mat = hcat(Eigenvalues...)
    out = vtk_write_array(VTKFileName, eigen_mat, "Eigen")
    return out[1]
end

"""
    WavesToVTK(VTKFIleName::String, wave)   

Stores the wave values in a VTK image file and returns the name of the file.
If the sizes of wave vectors for each each value is not equal, the wave matrix
will have size = (e, w, n) where e = No of eigen values, w = max no of values for
each eigen value, n =  no of Kpoints). The matrix is stored as two submatrix,
one containing the real part and the other containing the imaginary part.
"""
function WavesToVTK(VTKFileName::String, wave)

    mat_size = (maximum(x->size(x)[1], wave), size(wave[1])[2], length(wave))
    wave_mat = zeros(ComplexF64, mat_size)
    @views for i = 1:length(wave)
        (x, y) = size(wave[i])
        for m in 1:x, n in 1:y
            wave_mat[m,n,i] = wave[i][m, n]
        end
    end
    out = vtk_write_array(VTKFileName, (Base.getproperty.(wave_mat, :re), 
            Base.getproperty.(wave_mat, :im)), ("Wave Real", "Wave Imag."))
    return out[1]
end
"""
    OccupationToVTK(VTKFileName::String, occupation)

Stores the occupation values in a VTK image file and returns the file name.
"""
function OccupationToVTK(VTKFileName::String, occupation::Array{Array{T,1},1}) where T<:Real
   
    occupation_mat = hcat(occupation...)
    out = vtk_write_array(VTKFileName, occupation_mat, "Occupation")
    return out[1]
end