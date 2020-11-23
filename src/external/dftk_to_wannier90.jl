using ProgressMeter
using LinearAlgebra
using Dates
using Printf

## TODO
# add handling of the SCDM routine as an alternative guess

include("win_guess_utils.jl")

"""
    Create a .win file for Wannier90, compatible with the system studied with DFTK.
    Options have to be compeleted by hand.
    Parameters can also be added as kwargs argument : e.g. num_iter = 500.
"""
function dftk2wan_win_file(prefix::String, basis::PlaneWaveBasis, scfres, kgrid,
                           num_wann::Integer;
                           bands_plot = false,
                           kwargs...
                           )
    
    # Check kgrids
    if (prod(kgrid) != size(scfres.ψ,1))
        error("The given kgrid doesn't match the one used for the scf calculation")
    end    

    # Write in file   
    open("$prefix.win", "w") do f

        write(f,"! $prefix.win file generated by DFTK at $(now())"*"\n"^2)
        
        # Required parameters
        num_bands = size(scfres.ψ[1],2)
        write(f,@sprintf("%-20s %s %5s \n","num_bands","=",num_bands))
        write(f,@sprintf("%-20s %s %5s \n","num_wann","=",num_wann))
        write(f,"\n")
        
        # Optional parameters in kwargs
        for (key,value) in kwargs
            write(f,@sprintf("%-20s %s   %-30s \n",key,"=",value))
        end
        write(f,"\n"^2)

        # Messages for the user
        write(f,"!"^20*" Complete with optional parameters :"*"\n"^2)

        # System
        write(f,"!"^20*" System"*"\n"^2)

        # Unit cell block
        write(f,"begin unit_cell_cart"*"\n"*"bohr"*"\n")
        unit_cell = transpose(basis.model.lattice) # cell vectors are in raws in Wannier90   
        for vec in eachrow(unit_cell)
            write(f,@sprintf("%10.6f %10.6f %10.6f \n",vec[1],vec[2],vec[3]))
        end
        write(f,"end unit_cell_cart"*"\n"^2)

        # Atoms block
        atoms = basis.model.atoms
        write(f,"begin atoms_frac"*"\n")
        for i in 1:length(atoms)                  # loop over different elements 
            for coord in atoms[i][2]              # loop over atoms of the same element
                write(f,@sprintf("%-2s %10.6f %10.6f %10.6f \n",
                                 atoms[i][1].symbol,coord[1],coord[2],coord[3]) )
            end
        end
        write(f,"end atoms_frac"*"\n"^2)

        # Projection block : message for the user.
        write(f,"""
        ! Add a projection block only with the option : guess = "win" in 
        ! dftk2wan_wannierization_files.
        ! Otherwise, DFTK uses by default gaussian guesses or 
        ! the SCDM method if guess = "SCDM" (not yet implemented).


        """)

        # k points
        write(f,"!"^20*" k_points"*"\n"^2)
        
        # Bands_plot block
        if bands_plot
            trash,labels,path = high_symmetry_kpath(basis.model)
            write(f,"begin kpoint_path"*"\n")
            for i in 1:size(path[1],1)-1
                # write path section A -> B
                A,B = path[1][i],path[1][i+1]
                if A == "\\Gamma"
                    A = "Γ"
                end
                if B == "\\Gamma"
                    B = "Γ"
                end
                # kpoints are writen in the file with normalized coordinates
                k_A = round.(DFTK.normalize_kpoint_coordinate.(get!(labels,A,1)),
                             digits = 5)
                k_B = round.(DFTK.normalize_kpoint_coordinate.(get!(labels,B,1)),
                             digits = 5)
                write(f,@sprintf("%s %10.6f %10.6f %10.6f  %s %10.6f %10.6f %10.6f \n",
                                 A,k_A[1],k_A[2],k_A[3],B,k_B[1],k_B[2],k_B[3]) )
            end
            write(f,"end kpoint_path"*"\n")
            write(f,"bands_plot = T"*"\n"^2)
        end
        
        # Mp grid
        if kgrid[1]*kgrid[2]*kgrid[3] !== size(scfres.ψ,1)
            error("Given kgrid doesn't match the kgrid used for scf calculation")
        end
        write(f,"mp_grid : $(kgrid[1]) $(kgrid[2]) $(kgrid[3])"*"\n"^2)
        
        # kpoints block
        write(f,"begin kpoints"*"\n")
        for i in 1:size(scfres.ψ,1)
            coord =  basis.kpoints[i].coordinate
            write(f,@sprintf("%10.6f %10.6f %10.6f \n",coord[1],coord[2],coord[3]))
        end
        write(f,"end kpoints"*"\n"^2)
    end

    return nothing
    
end


"""
Read the .nnkp file provided by the preprocessing routine of Wannier90 
(i.e. "wannier90.x -pp prefix")
Returns: 
1) the array 'nn_kpts' of k points, they respective nearest neighbors and associated
   shifing vectors (non zero if the neighbour of is located in another cell).
2) the number 'nn_num' of neighbors per k point. 
2) the array 'projs' of projections. Each line contains the informations for one 
   projection :  [center],[quantum numbers: l,mr,r],[ref. z_axis],[ref. x_axis],α = Z/a
"""
function read_nnkp_file(prefix::String)

    # Check the presence of the file
    if !isfile("$prefix.nnkp")
        error("file $prefix.nnkp not found.")   
    end

    # Read the file
    file = open("$prefix.nnkp", "r")
    @info "Reading nnkp file", file 
    ln = [string(l) for l in eachline(file)]
    close(file)

    # Extract nnkp block
    i_nn_kpts = findall(x-> endswith(x,"nnkpts"),ln)  # 1st and last line of the block                
    @assert size(i_nn_kpts,1) == 2
    nn_num = parse(Int,ln[i_nn_kpts[1]+1])          # number of neighbors per k points
    nn_kpts =  [parse.(Int,split(l,' ',keepempty = false))
                for l in ln[i_nn_kpts[1]+2:i_nn_kpts[2]-1] ]
    

    # Extract projections block : NEEDED FOR WIN GUESSES
    i_projs = findall(x-> endswith(x,"projections"),ln) # 1st and last line of the block   
    @assert size(i_projs,1) == 2
    raw_projs = split.(ln[i_projs[1]+1:i_projs[2]-1],' ',keepempty = false)

    n_projs = parse(Int,only(popfirst!(raw_projs)))     # number of projections
    @assert(n_projs == size(raw_projs,1)/2)

    # Reshape so that one line gives all infos about one projection
    raw_projs = reshape(raw_projs,(2,n_projs))
    # Parse in the format proj = [ [[center],[l,mr,r],[z_axis],[x_axis], α], ... ]
    projs = []
    for j in 1:n_projs
        center = parse.(Float64,raw_projs[1,j][1:3])
        quantum_numbers = parse.(Int,raw_projs[1,j][4:end])
        z_axis = parse.(Float64,raw_projs[2,j][1:3])
        x_axis = parse.(Float64,raw_projs[2,j][4:6])
        α = parse(Float64,raw_projs[2,j][7])
        push!(projs, [center,quantum_numbers,z_axis,x_axis,α])
    end
    
    nn_kpts,nn_num,projs    
end



"""
Take scfres, the result of a self_consistent_field calculation and writes eigenvalues
(in eV) in a format readable by Wannier90.
"""
function generate_eig_file(prefix::String, scfres)
    open("$prefix.eig", "w") do f
        for (k,e_k) in enumerate(scfres.eigenvalues)
            for (n,e_kn) in enumerate(e_k) 
                write(f,@sprintf("%3i  %3i   %25.18f \n",n,k,(1/DFTK.units.eV)*e_kn))
            end
        end
    end
    return nothing
end


@doc raw"""
 Computes the matrix ``[M^{k,b}]_{m,n} = \langle u_{m,k} | u_{n,k+b} \rangle`` 
 for given k, kpb = k+b.
 
 Return in the format of a (n_bands^2, 2) matrix, where each line is one overlap, 
 and columns are real and imaginary parts.

 G_shift is the "shifting" vector, correction due to the periodicity conditions
 imposed on k -> ψ_k. 
 It is non zero if kpb is taken in another unit cell of the reciprocal lattice.
 We use here that : ``u_{n(k + G_shift)}(r) = e^{-i*\langle G_shift,r \rangle} u_{nk}``
"""
function overlap_Mmn_k_kpb(basis::PlaneWaveBasis, ψ, ik::Integer, ikpb::Integer,
                           n_bands::Integer;
                           G_shift = [0,0,0])

    Mkb = zeros(ComplexF64,(n_bands,n_bands))
    # Search for common Fourier modes and their resp. indices in bloch states k and kpb
    k = basis.kpoints[ik]; kpb = basis.kpoints[ikpb] #renaming for clarity
    map_fourier_modes = [ (iGk,DFTK.index_G_vectors(basis, kpb, Gk+G_shift))
                          for (iGk,Gk) in enumerate(G_vectors(k))
                          if !isnothing(DFTK.index_G_vectors(basis,kpb,Gk+G_shift)) ]

    # Compute overlaps
    for n in 1:n_bands
        for m in 1:n_bands
            # Select the coefficient in right order
            Gk_coeffs = @view ψ[ik][[i[1] for i in map_fourier_modes],m] 
            Gkpb_coeffs = @view ψ[ikpb][[i[2] for i in map_fourier_modes],n]
            Mkb[m,n] = dot(Gk_coeffs,Gkpb_coeffs)
        end
    end    

    Mkb
end


"""
Iteratively use the preceeding function on each k and kpb to generate the whole .mmn file.
"""
function generate_mmn_file(prefix::String, basis::PlaneWaveBasis, ψ,
                           nn_kpts, nn_num::Integer)
    # general parameters
    n_bands = size(ψ[1],2)
    k_size = length(ψ)
    
    progress = Progress(size(nn_kpts,1), desc="Computing Mmn overlaps : ")
    
    # Small function for the sake of clarity
    read_nn_kpts(n) = nn_kpts[n][1],nn_kpts[n][2],nn_kpts[n][3:end]

    # Write file
    open("$prefix.mmn", "w") do f
        write(f,"Generated by DFTK at $(now())"*"\n")
        write(f,"$n_bands  $k_size  $nn_num"*"\n")

        # Loop over all (k_points, nearest_neighbour, shif_vector)
        for i_nnkp in 1:size(nn_kpts,1)
            # Write the label of the matrix
            k,kpb,shift = read_nn_kpts(i_nnkp)
            write(f,@sprintf("%i  %i  %i  %i  %i \n",
                             k,kpb,shift[1],shift[2],shift[3]) )
            # The matrix itself
            Mkb = overlap_Mmn_k_kpb(basis, ψ, k, kpb, n_bands,; G_shift = shift)
            for ovlp in Mkb
                write(f, @sprintf("%22.18f %22.18f \n",real(ovlp),imag(ovlp)) )
            end
            next!(progress)
        end
    end
    
end


@doc raw"""
Compute the matrix ``[A_k]_{m,n} = \langle ψ_m^k | g^{\text{per}}_n \rangle`` 

``g^{per}_n`` are periodized gaussians whose respective centers are given as an
 (n_bands,1) array [ [center 1], ... ].
Centers are to be given in lattice coordinates.

The dot product is computed in the Fourier space. 

Given an orbital ``g_n``, the periodized orbital is defined by :
 ``g^{per}_n=  \sum\limits_{R in lattice} g_n( ⋅ - R)``. 
``g^{per}_n`` is not explicitly computed. Its Fourier coefficient at any G 
is given by the value of the Fourier transform of ``g_n`` in G.
"""
function A_k_matrix_gaussian_guesses(basis::PlaneWaveBasis, ψ, k::Integer,
                                     n_bands::Integer, n_wann::Integer;
                                     centers = [], projs = [])
    
    # associate a center with the fourier transform of the corresponding gaussian
    guess_fourier(center) = xi ->  exp(-im*dot(xi,center) - dot(xi,xi)/4) 
    
    G_cart =[G for (iG,G) in enumerate(G_vectors_cart(basis))]    
    # Indices of the Fourier modes of the bloch states in the general FFT_grid for given k
    index = [DFTK.index_G_vectors(basis,G) for G in G_vectors(basis.kpoints[k])]                    
  
    A_k = zeros(Complex,(n_bands,n_wann))
    
    # Compute A_k
    for n in 1:n_wann
        fourier_gn = guess_fourier(basis.model.lattice*transpose(centers[n]))
        # functions are l^2 normalized in Fourier, in DFTK conventions.
        norm_gn_per = norm(fourier_gn.(G_cart),2)
        
        # Fourier coeffs of gn_per for G_vectors in common with ψm
        coeffs_gn_per = fourier_gn.(G_cart[index])./ norm_gn_per     
        # Compute overlap
        for m in 1:n_bands
            coeffs_ψm = ψ[k][:,m]
            A_k[m,n] = dot(coeffs_ψm,coeffs_gn_per)
        end  
    end
    
    A_k
end


"""
Use the preceding functions on every k to generate the .amn file 
"""
function generate_amn_file(prefix::String,basis::PlaneWaveBasis,ψ, n_wann::Integer;
                           projs=[], centers=[], guess="")
    # Select guess
    if guess == "win"
        compute_A_k = A_k_matrix_win_guesses
        # Check if the right number of projection is given...
        @assert n_wann == size(projs,1)  
        
    elseif guess == "gaussian"
        compute_A_k = A_k_matrix_gaussian_guesses
        @assert n_wann == size(centers, 1)   # ... same for the number of centers.
        
    end

    # general parameters
    n_bands = size(ψ[1],2)
    k_size = size(ψ,1)

    progress = Progress(k_size, desc="Computing Amn overlaps : ")

    # write file
    open("$prefix.amn", "w") do f
        # Header
        write(f,"Generated by DFTK at $(now())"*"\n")
        write(f,"$n_bands   $k_size   $n_wann \n")

        # Matrices
        for k in 1:k_size
            A_k = compute_A_k(basis,ψ,k,n_bands,n_wann;
                              centers = centers, projs = projs)
            for n in 1:size(A_k,2)
                for m in 1:size(A_k,1)
                    write(f,@sprintf("%3i %3i %3i  %22.18f %22.18f \n",
                                     m,n,k,real(A_k[m,n]),imag(A_k[m,n])) )
                end
            end
            next!(progress)
        end      
    end         

end


"""
Use the above functions to read the nnkp file and generate the 
.eig, .amn and .mmn files needed by Wannier90.
"""
function dftk2wan_wannierization_files(prefix::String, basis::PlaneWaveBasis,
                                       scfres, n_wann::Integer;
                                       write_amn = true,
                                       write_mmn = true,
                                       write_eig = true,
                                       guess = "gaussian",
                                       centers = [])

    # Check for errors
    @assert guess ∈ ("gaussian","SCDM","win")
    
    if guess == "SCDM"
        error("SCDM not yet implemented")
    end

    @info "Guess = $guess"

    # Generate random centers for gaussian guesses if none are given
    # The centers are in lattice coordinate in [-1,1]^3.
    if (guess == "gaussian") & isempty(centers)
        for i in 1:n_wann
            push!(centers, 1 .- 2 .*rand(1,3))   
        end
    end
    
    # Read the .nnkp file
    ψ = scfres.ψ
    nn_kpts,nn_num,projs = read_nnkp_file(prefix)
    
    # Generate_files
    if write_eig
        generate_eig_file("Si",scfres)
    end
    
    if write_amn
        generate_amn_file("Si", basis, ψ, n_wann;
                          centers = centers, projs = projs, guess = guess)
    end
    
    if write_mmn
        generate_mmn_file("Si", basis, ψ, nn_kpts, nn_num)
    end

end
