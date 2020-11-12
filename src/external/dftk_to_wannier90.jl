using ProgressMeter
using LinearAlgebra
using Dates


@doc raw"""
    Create a .win file for Wannier90, compatible with the system studied with DFTK.
    Options have to be compeleted by hand.

    TODO 1 -  Implement a structure containing all options for Wannier90 calculations ?
    TODO 2 -  Generate kpoint_path with DFTK for band diagram ploting. In wannier90 :
    "
    begin kpoint_path
    L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
    G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
    X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
    K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
    end kpoint_path
    bands_plot =T
    "

    TODO 3 - add handling of the SCDM routine as an alternative for bond centered gaussian guesses

    TODO 4 -  handle the cases were num_wann != num_band.

"""
function dftk2wan_win_file(prefix::String,basis::PlaneWaveBasis,scfres;
                           kgrid = [4,4,4],
                           num_iter = 500,
                           num_print_cycles = 50)
    
    num_bands = size(scfres.ψ[1],2)
    num_wann = num_bands      # TODO handle the cases were num_wann != num_band.
    if (prod(kgrid) != size(scfres.ψ,1))
        error("The given kgrid doesn't match the one used for the scf calculation")
    end    

    ## Write in file   
    open("$prefix.win","w") do f
        
        ## General parameters
        write(f,"num_bands "*(" "^9)*"= $num_bands"*"\n")
        write(f,"num_wann "*(" "^10)*"= $num_bands"*"\n"^2)

        write(f,"num_iter "*(" "^10)*"= $num_iter"*"\n")
        write(f,"num_print_cycles "*(" "^2)*"= $num_print_cycles"*"\n"^2)

        ## Messages for the user
        write(f,"!"^20*" Complete with optional parameters :"*"\n"^2)
        write(f,"!"^20*" System "*"\n"^2)
        
        ## Unit cell block
        write(f,"begin unit_cell_cart"*"\n"*"bohr"*"\n")
        unit_cell = transpose(basis.model.lattice) #unit cell in raws for Wannier90
        for i in 1:3
            for j in 1:3
                write(f,"$(unit_cell[i,j]) ")
            end
            write(f,"\n")
        end
        write(f,"end unit_cell_cart"*"\n"^2)

        ## Atoms bloc
        atoms = basis.model.atoms
        
        write(f,"begin atoms_frac"*"\n")
        #loop over different elements
        for i in 1:length(atoms)
            symbol = String(atoms[i][1].symbol)
            #loop over atoms of the same element
            for coord in atoms[i][2]
                write(f,symbol*"  "*"$(coord[1])  $(coord[2])  $(coord[3])"*"\n")  
            end
        end
        write(f,"end atoms_frac"*"\n"^2)

        ## Message fo the user on projection bloc
        write(f,"! Projection bloc not needed"*"\n")
        write(f,"! By default, DFTK uses gaussian guesses on specified centers. See documentation."*"\n")
        write(f,"! Otherwise use 'SCDM = true' in arguments of template_win_file"*"\n"^2)

        ## Mp grid
        if kgrid[1]*kgrid[2]*kgrid[3] !== size(scfres.ψ,1)
            error("Given kgrid doesn't match the kgrid used for scf calculation")
        end
        write(f,"mp_grid : $(kgrid[1]) $(kgrid[2]) $(kgrid[3])"*"\n"^2)

        ## kpoints bloc
        write(f,"begin kpoints"*"\n")
        for i in 1:size(scfres.ψ,1)
            for x in basis.kpoints[i].coordinate
                write(f,"$x ")
            end
            write(f,"\n")
        end
        write(f,"end kpoints"*"\n"^2)

        @info "File generated.",f
    end
    
    
end


"""
Read the .nnkp file provided by the preprocessing routine of Wannier90 (i.e. "wannier90.x -pp prefix")
Returns: 
1) the array of k points, they respective neirest neighbours and associated shifing vectors (non zero if the neighbour of is located in another cell).
2) OLD !!! the array of projections. Each line contains the informations for one projection :  [center],[quantum numbers: l,mr,r],[ref. z_axis],[ref. x_axis],α = Z/a
"""
function read_nnkp_file(prefix::String)
    ## Read file
    if !isfile("$prefix.nnkp")
        error("file $prefix.nnkp not found.")    # Check the presence of the file
    end
    
    file = open("$prefix.nnkp")
    @info "Reading nnkp file", file 
    ln = [string(l) for l in eachline(file)]
    close(file)


    ## Extract nnkp block 
    i_nn_kpts = findall(x-> endswith(x,"nnkpts"),ln)                               #Indices of the first and last line of the nnkpts block
    @assert size(i_nn_kpts,1) == 2
    char_to_vec(line) = [parse.(Int,x) for x in split(line,' ',keepempty=false)]
    nn_num = parse(Int64,ln[i_nn_kpts[1]+1])                                       #number of neighbours per k points
    nn_kpts =  [ char_to_vec(l) for l in ln[i_nn_kpts[1]+2:i_nn_kpts[2]-1] ]



    ## Extract projections block : NOT NEEDED ANYMORE
    i_projs = findall(x-> endswith(x,"projections"),ln)                                  #Indices of the first and last line of the projection block
    @assert size(i_projs,1) == 2
    raw_projs = [split(ln[i],' ',keepempty = false) for i in i_projs[1]+1:i_projs[2]-1]  #data in strings

    # reshape so that one line gives all infos about one projection
    n_projs = parse(Int,only(popfirst!(raw_projs)))
    @assert(n_projs == size(raw_projs,1)/2)
    raw_projs = reshape(raw_projs,(2,n_projs))

    # Parse in the format proj = [ [[center],[l,mr,r],[z_axis],[x_axis], α], ... ]
    projs = []
    for j in 1:n_projs
        center = [parse(Float64,x) for x in raw_projs[1,j][1:3]]
        quantum_numbers = [parse(Int,x) for x in raw_projs[1,j][4:end] ]
        z_axis = [parse(Float64,x) for x in raw_projs[2,j][1:3]]
        x_axis = [parse(Float64,x) for x in raw_projs[2,j][4:6]]
        α = parse(Float64,raw_projs[2,j][7])
        push!(projs, [center,quantum_numbers,z_axis,x_axis,α])
    end
    
    nn_kpts,nn_num,projs    
end




##!!!!!!!!!! !!!! !  !              eig file              !  ! !!!! !!!!!!!!!!##

"""
Take scf_res, the result of a self_consistent_field calculation and writes eigenvalues in a format readable by Wannier90.
"""
function generate_eig_file(prefix::String,scf_res)
    #energies have to be in EV
    Ha_to_Ev = 27.2114

    eigvalues = scf_res.eigenvalues
    eigvalues .*= Ha_to_Ev
    k_size = size(eigvalues,1)
    n_bands = size(eigvalues[1],1)

    #write file
    open("$prefix.eig","w") do f
        for k in 1:k_size
            for n in 1:n_bands
                write(f,"  $n  $k  $(eigvalues[k][n])"*"\n")
            end
        end
    end

    return nothing
    
end


##!!!!!!!!!! !!!! !  !              mmn file              !  ! !!!! !!!!!!!!!!##

@doc raw"""
 Computes the matrix ``[M^{k,b}]_{m,n} = \langle u_{m,k} | u_{n,k+b} \rangle`` for given k, kpb = k+b.
 
 Returns in the format of a (n_bands^2, 2) matrix, where each line is one overlap, and columns are real and imaginary parts.

 K_shift is the "shifting" vector correction due to the periodicity conditions imposed on k -> ψ_k.
 We use here that : ``u_{n(k + K_shift)}(r) = e^{-i*\langle K_shift,r \rangle} u_{nk}``
"""
function overlap_Mmn_k_kpb(pw_basis::PlaneWaveBasis,ψ,k,kpb,n_bands; K_shift = [0,0,0])
    Mkb = zeros(Float64,n_bands*n_bands,2)
    current_line = 0 #Manual count necessary to match the format of .mmn files.

    for n in 1:n_bands
        for m in 1:n_bands
            
            ovlp = 0im
            
            #Extract Fourier coeffs and corresponding vectors in reciprocal lattice
            Gk_coeffs = ψ[k][:,m] 
            Gk_vec = G_vectors(pw_basis.kpoints[k])
            Gkpb_coeffs = ψ[kpb][:,n]
            Gkpb_vec = [ G - K_shift for G in G_vectors(pw_basis.kpoints[kpb]) ] # Don't forget the shift, see the DOC block
            
            #Search for common Fourier modes, corresponding indices are written in map_fourier_mode
            map_fourier_modes = []
            for G1 in Gk_vec
                for G2 in Gkpb_vec
                    if  G1 == G2
                        iG1 = only(findall(x-> x==G1,Gk_vec))
                        iG2 = only(findall(x-> x==G2,Gkpb_vec))
                        push!(map_fourier_modes,[iG1,iG2])
                    end
                end
            end
            
            #Compute the overlap for mn
            for (i,j) in map_fourier_modes
                ovlp += conj(Gk_coeffs[i])*Gkpb_coeffs[j]
            end
            
            current_line += 1 #Go to the next line
            Mkb[current_line,:] = [real(ovlp),imag(ovlp)]

        end
    end

    Mkb

 end



"""
Iteratively use the preceeding function on each k and kpb to generate the whole .mmn file.
"""
function generate_mmn_file(prefix::String,pw_basis::PlaneWaveBasis,ψ, nn_kpts, nn_num)
    #general parameters
    n_bands = size(ψ[1],2)
    k_size = length(ψ)
    
    progress = Progress(only(size(nn_kpts)),desc = "Computing Mmn overlaps : ")
    
    #Small function for the sake of clarity
    read_nn_kpts(n) = nn_kpts[n][1],nn_kpts[n][2],nn_kpts[n][3:end]

    #Write file
    open("$prefix.mmn","w") do f
        write(f,"Generated by DFTK at ",string(now()),"\n")
        write(f,"$n_bands  $k_size  $nn_num"*"\n")
        
        for i_nnkp in 1:only(size(nn_kpts)) #Loop over all (k_points, neirest_neighbour, shif_vector)
            #Label of the matrix
            k,kpb,shift = read_nn_kpts(i_nnkp)
            write(f,string(k)*"  "*string(kpb)*"  "*string(shift[1])*"  "*string(shift[2])*"  "*string(shift[3])*"\n")   
            #Overlaps
            Mkb = overlap_Mmn_k_kpb(pw_basis,ψ,k,kpb,n_bands,; K_shift = shift)
            for i in 1:n_bands*n_bands
                write(f, "$(string(Mkb[i,1]))  $(string(Mkb[i,2]))"*"\n")
            end
            next!(progress)
        end
    end

end


"""
Given a k point, provide the indices of the corresponding G_vectors in the general FFT grid.
"""
function map_k_G_vectors(pw_basis::PlaneWaveBasis,k,fft_grid)
    map = []
    for G in G_vectors(pw_basis.kpoints[k])
        iG = only( findall(x -> x==G, fft_grid) )
        push!(map,iG)
    end
    map
    
end


@doc raw"""
Compute the matrix ``[A_k]_{m,n} = \langle ψ_m^k | g^{\text{per}}_n \rangle`` where ``g^{per}_n`` are periodized gaussians whose respective centers are given as an (n_bands,1) array [ [center 1], ... ].

The dot product is computed in the Fourier space. 

Given a gaussian ``g_n``, the periodized gaussian is defined by : ``g^{per}_n=  \sum\limits_{R in lattice} g_n( ⋅ - R)``. 
``g^{per}_n`` is not explicitly computed. Its Fourier coefficient at ant G is given by the value of the Fourier transform of ``g_n`` in G.
"""
function A_k_matrix_gaussian_guesses(pw_basis::PlaneWaveBasis,ψ,k,centers)

    ## Before calculation 
    guess_fourier(center) = xi ->  exp(-im*dot(xi,center) - dot(xi,xi)/4)  #associate a center with the fourier transform of the corresponding gaussian
    n_bands = size(ψ[1][1,:],1)
    n_guess = size(centers,1)

    fft_grid = [G for (iG,G) in enumerate(G_vectors(pw_basis)) ]    #FFT grid in recip lattice coordinates
    G_cart =[ pw_basis.model.recip_lattice * G for G in fft_grid ]  #FFT grid in cartesian coordinates
    index = map_k_G_vectors(pw_basis,k,fft_grid)                    #Indices of the Fourier modes of the bloch states in the general FFT_grid for given k

    #Initialize output
    A_k = zeros(Complex,(n_bands,n_guess))

    ## Compute A_k
    for n in 1:n_guess

        ## Compute fourier coeffs of g_per_n
        fourier_gn = guess_fourier(centers[n])
        norm_g_per_n = norm([fourier_gn(G) for G in G_cart],2)                          # functions are l^2 normalized in Fourier, in DFTK conventions.
        coeffs_g_per_n = [ fourier_gn(G_cart[iG]) for iG in index ]  ./ norm_g_per_n    # Fourier coeffs of gn for G_vectors in common with ψm
        
        for m in 1:n_bands
            coeffs_ψm = ψ[k][:,m]
            A_k[m,n] = dot(coeffs_ψm,coeffs_g_per_n)                                    #The first argument is conjugated with the Julia "dot" function
        end
        
    end

    A_k
end


"""
Use the preceding function on every k to generate the .amn file 
"""
function generate_amn_file(prefix::String,pw_basis::PlaneWaveBasis,ψ ; centers = [])
    # general parameters
    n_bands = size(ψ[1],2)
    k_size = length(ψ)

    progress = Progress(k_size,desc = "Computing Amn overlaps : ")

    ## write file
    open("$prefix.amn","w") do f
        write(f,"Generated by DFTK at ",string(now()),"\n")
        write(f,string(n_bands)*"   "*string(k_size)*"   "*string(n_bands)*"\n") #TODO num_wan pour le dernier
        for k in 1:k_size
            A_k = A_k_matrix_gaussian_guesses(pw_basis,ψ,k,centers)
            for m in 1:size(A_k,1)
                for n in 1:size(A_k,2)
                    write(f,"$m  $n  $k  $(real(A_k[m,n]))  $(imag(A_k[m,n]))"*"\n")
                end
            end
            next!(progress)
        end
         
    end         

end


"""
Use the above functions to read the nnkp file and generate the .eig, .amn and .mmn files needed by Wannier90.
"""
function dftk2wan_wannierization_files(prefix::String,pw_basis::PlaneWaveBasis,scf_res;
                              centers = [],
                              write_amn = true,
                              write_mmn = true,
                              write_eig = true,
                              SCDM = false)

    if SCDM
        error("SCDM not yet implemented")
    end
    
    if isempty(centers) & !SCDM
        error("You have to specify centers for gaussian guesses with 'centers = ' or use 'SCDM = true' (not yet implemented)")
    end
    
    ψ = scf_res.ψ
    #read the .nnkp file
    nn_kpts,nn_num,projs = read_nnkp_file(prefix) 
    
    ## Generate_files
    if write_eig
        generate_eig_file("Si",scf_res)
    end
    
    if write_amn
        generate_amn_file("Si",pw_basis,ψ; centers = centers)
    end
    
    if write_mmn
        generate_mmn_file("Si",pw_basis,ψ,nn_kpts,nn_num)
    end
    
    return nothing

end
