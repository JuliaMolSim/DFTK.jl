using Dates
using wannier90_jll

"""
    Create a .win file for Wannier90, compatible with the system studied with DFTK.
    Parameters to Wannier90 can be added as kwargs : e.g. num_iter = 500.
"""
function write_win(prefix::String, basis, ψ,
                           num_wann::Integer;
                           bands_plot=false,
                           kwargs...
                           )
    open("$prefix.win", "w") do f

        write(f, "! $prefix.win file generated by DFTK at $(now())\n\n")

        # Required parameters
        num_bands = size(ψ[1],2)
        write(f,@sprintf("%-20s %s %5s \n","num_bands","=",num_bands))
        write(f,@sprintf("%-20s %s %5s \n","num_wann","=",num_wann))
        write(f,"\n")

        # Optional parameters in kwargs
        for (key,value) in kwargs
            write(f,@sprintf("%-20s %s   %-30s\n",key,"=",value))
        end
        write(f,"\n\n")

        # System
        write(f,"!"^20*" System \n\n")

        # Unit cell block
        write(f,"begin unit_cell_cart\n"*"bohr\n")
        unit_cell = transpose(basis.model.lattice) # cell vectors are in rows in Wannier90
        for vec in eachrow(unit_cell)
            write(f,@sprintf("%10.6f %10.6f %10.6f \n",vec[1],vec[2],vec[3]))
        end
        write(f,"end unit_cell_cart \n\n")

        # Atoms block
        atoms = basis.model.atoms
        write(f,"begin atoms_frac\n")
        for i in 1:length(atoms)                  # loop over different elements 
            for coord in atoms[i][2]              # loop over atoms of the same element
                write(f,@sprintf("%-2s %10.6f %10.6f %10.6f \n",
                                 atoms[i][1].symbol,coord[1],coord[2],coord[3]) )
            end
        end
        write(f,"end atoms_frac\n\n")

        # k points
        write(f,"!"^20*" k_points\n\n")

        # Bands_plot block
        if bands_plot
            trash, labels, path = high_symmetry_kpath(basis.model)
            write(f,"begin kpoint_path\n")
            for i in 1:size(path[1],1)-1
                # write path section A -> B
                A,B = path[1][i],path[1][i+1]
                (A == "\\Gamma") && (A = "Γ")
                (B == "\\Gamma") && (B = "Γ")

                # kpoints are writen in the file with normalized coordinates
                k_A = round.(DFTK.normalize_kpoint_coordinate.(get!(labels,A,1)),
                             digits = 5)
                k_B = round.(DFTK.normalize_kpoint_coordinate.(get!(labels,B,1)),
                             digits = 5)
                write(f,@sprintf("%s %10.6f %10.6f %10.6f  %s %10.6f %10.6f %10.6f \n",
                                 A,k_A[1],k_A[2],k_A[3],B,k_B[1],k_B[2],k_B[3]) )
            end
            write(f,"end kpoint_path\n")
            write(f,"bands_plot = T\n\n")
        end

        # Mp grid
        write(f,"mp_grid : $(basis.kgrid[1]) $(basis.kgrid[2]) $(basis.kgrid[3])\n\n")

        # kpoints block
        write(f,"begin kpoints"*"\n")
        for i in 1:size(ψ,1)
            coord =  basis.kpoints[i].coordinate
            write(f,@sprintf("%10.6f %10.6f %10.6f \n",coord[1],coord[2],coord[3]))
        end
        write(f,"end kpoints\n\n")
    end
    return nothing
end

write_win(prefix::String, scfres, num_wann::Integer; bands_plot=false, kwargs...) =
    write_win(prefix, scfres.basis, scfres.ψ, num_wann; bands_plot=bands_plot, kwargs...)


"""
Read the .nnkp file provided by the preprocessing routine of Wannier90
(i.e. "wannier90.x -pp prefix")
Returns:
1) the array 'nn_kpts' of k points, their respective nearest neighbors and associated
   shifing vectors (non zero if the neighbor of is located in another cell).
2) the number 'nn_num' of neighbors per k point.
2) the array 'projs' of projections. Each line contains the informations for one 
   projection :  [center],[quantum numbers: l,mr,r],[ref. z_axis],[ref. x_axis],α = Z/a
"""
function read_nnkp(prefix::String)

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

    # Do we keep wannier90's guesses ?
    # Extract projections block : needed for win guesses only
    i_projs = findall(x-> endswith(x,"projections"),ln) # 1st and last line of the block
    @assert size(i_projs,1) == 2
    raw_projs = split.(ln[i_projs[1]+1:i_projs[2]-1],' ',keepempty = false)

    n_projs = parse(Int64,only(popfirst!(raw_projs)))     # number of projections
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
function write_eig(prefix::String, scfres)
    open("$prefix.eig", "w") do f
        for (k,e_k) in enumerate(scfres.eigenvalues)
            for (n,e_kn) in enumerate(e_k)
                # write(f,@sprintf("%3i  %3i   %25.18f \n",n,k,(1/DFTK.units.eV)*e_kn))
                write(f,@sprintf("%3i  %3i   %25.18f \n",n,k,auconvert(Unitful.eV,e_kn).val))
            end
        end
    end
    return nothing
end

@doc raw"""
 Computes the matrix ``[M^{k,b}]_{m,n} = \langle u_{m,k} | u_{n,k+b} \rangle``
 for given k, kpb = k+b.

 G_shift is the "shifting" vector, correction due to the periodicity conditions
 imposed on k -> ψ_k.
 It is non zero if kpb is taken in another unit cell of the reciprocal lattice.
 We use here that : ``u_{n(k + G_shift)}(r) = e^{-i*\langle G_shift,r \rangle} u_{nk}``
"""
function overlap_Mmn_k_kpb(basis::PlaneWaveBasis, ψ, ik::Integer, ikpb::Integer,
                           n_bands::Integer;
                           G_shift=[0,0,0])

    Mkb = zeros(ComplexF64,(n_bands,n_bands))
    # Search for common Fourier modes and their resp. indices in bloch states k and kpb
    k = basis.kpoints[ik]; kpb = basis.kpoints[ikpb] #renaming for clarity
    map_fourier_modes = [ (iGk,DFTK.index_G_vectors(basis, kpb, Gk+G_shift))
                          for (iGk,Gk) in enumerate(G_vectors(k))
                          if !isnothing(DFTK.index_G_vectors(basis,kpb,Gk+G_shift)) ]
    iGk = [i[1] for i in map_fourier_modes]   # TODO search for better structure than
    iGkpb = [i[2] for i in map_fourier_modes] # map_fourier_modes

    # Compute overlaps
    for n in 1:n_bands
        for m in 1:n_bands
            # Select the coefficient in right order
            Gk_coeffs = @view ψ[ik][iGk,m]
            Gkpb_coeffs = @view ψ[ikpb][iGkpb,n]
            Mkb[m,n] = dot(Gk_coeffs,Gkpb_coeffs)
        end
    end
    Mkb
end

"""
Iteratively use the preceding function on each k and kpb to generate the whole .mmn file.
"""
function write_mmn(prefix::String, basis::PlaneWaveBasis, ψ,
                           nn_kpts, nn_num::Integer)
    # general parameters
    n_bands = size(ψ[1],2)
    k_size = length(ψ)

    # Small function for the sake of clarity
    read_nn_kpts(n) = nn_kpts[n][1],nn_kpts[n][2],nn_kpts[n][3:end]

    # Write file
    open("$prefix.mmn", "w") do f
        write(f,"Generated by DFTK at $(now())\n")
        write(f,"$n_bands  $k_size  $nn_num\n")
        # Loop over all (k_points, nearest_neighbor, shif_vector)
        for i_nnkp in 1:size(nn_kpts,1)
            # Write the label of the matrix
            k,kpb,shift = read_nn_kpts(i_nnkp)
            write(f,@sprintf("%i  %i  %i  %i  %i \n",
                             k,kpb,shift[1],shift[2],shift[3]) )
            # The matrix itself
            Mkb = overlap_Mmn_k_kpb(basis, ψ, k, kpb, n_bands,; G_shift=shift)
            for ovlp in Mkb
                write(f, @sprintf("%22.18f %22.18f \n",real(ovlp),imag(ovlp)) )
            end
        end
    end
    return nothing
end

write_mmn(prefix::String, scfres, nn_kpts, nn_num) =
    write_mmn(prefix, scfres.basis, scfres.ψ, nn_kpts, nn_num)


@doc raw"""
Compute the matrix ``[A_k]_{m,n} = \langle ψ_m^k | g^{\text{per}}_n \rangle``

``g^{per}_n`` are periodized gaussians whose respective centers are given as an
 (n_bands,1) array [ [center 1], ... ].

Centers are to be given in lattice coordinates and G_vectors in reduced coordinates.
The dot product is computed in the Fourier space.

Given an orbital ``g_n``, the periodized orbital is defined by :
 ``g^{per}_n =  \sum\limits_{R \in {\rm lattice}} g_n( \cdot - R)``.
The  Fourier coefficient of ``g^{per}_n`` at any G
is given by the value of the Fourier transform of ``g_n`` in G.
"""
function Ak_matrix_gaussian_guess(basis::PlaneWaveBasis, ψ_k, k_pt,
                                     n_bands::Integer, num_wann::Integer;
                                     centers=[], projs=[])

    # associate a center with the fourier transform of the corresponding gaussian
    fourier_gn(center,qs) = [exp(-im*dot(q,center) - dot(q,q)/4) for q in qs]
    # All q = k+G in reduced coordinates (2π .*)
    q_vec = (2π) .*[G + k_pt.coordinate_cart for (iG,G) in enumerate(G_vectors(basis))]

    # Indices of the Fourier modes of the Bloch states in the general FFT_grid for given k
    index = [DFTK.index_G_vectors(basis,G) for G in G_vectors(k_pt)]
    Ak = zeros(eltype(ψ_k),(n_bands,num_wann))

    # Compute Ak
    for n in 1:num_wann
        center = basis.model.lattice*transpose(centers[n])
        # Functions are l^2 normalized in Fourier, in DFTK conventions.
        norm_gn_per = norm(fourier_gn(center,q_vec),2)
        # Fourier coeffs of gn_per for q_vectors in common with ψ_k
        coeffs_gn_per = fourier_gn(center,q_vec[index]) ./ norm_gn_per
        # Compute overlap
        for m in 1:n_bands
            coeffs_ψm = ψ_k[:,m]
            Ak[m,n] = dot(coeffs_ψm, coeffs_gn_per)
        end
    end
    Ak
end

@doc raw"""
    The quantum numbers given by Wannier90 are not in common use.
    In turn the order in which the orbitals s,p,d and f are given in the Tables 3
    (see [Wannier90's user guide][http://www.wannier.org/support/] p.54)
    is not matching the classic order given in spherical_harmonics.jl.

    The purpose of this function is to retrieve the proper quantum number m
    from the one given in the nnkp file.

    Corresponding m Wannier <-> DFTK:  p  [1,2,3]         <->  [0,1,-1]
                                       d  [1,2,3,4,5]     <->  [0,1,-1,2,-2]
                                       f  [1,2,3,4,5,6,7] <->  [0,1,-1,2,-2,3,-3]
"""
function retrieve_proper_m(l::Integer,mr::Integer)
    @assert 0 ≤ l
    m_p = (0,1,-1); m_d = (0,1,-1,2,-2); m_f = (0,1,-1,2,-2,3,-3)
    (l == 0) && return 0
    (l == 1) && return m_p[mr]  # p
    (l == 2) && return m_d[mr]  # d
    (l == 3) && return m_f[mr]  # f
    error("Quantum numbers are not matching any implemented
                orbital (s,p,d,f)")
end

@doc raw"""
    Gives the analytic expression of the integral
    ``I_l(q) = \int_{\mathbb{R}^+} r^(l+2) exp(-r^2/2) j_l(|q|r)dr``
    as given in arXiv:1908.07374v2, equation (2.5).

    ``j_l`` is the spherical Bessel function of order l.
    q is expected in cartesian coordinates
"""
function intR_l(l::Integer, norm_q_cart)
    √(π/2) * (norm_q_cart^l) * exp(-(norm_q_cart^2)/2)
end

@doc raw"""
Given quantum numbers and center (in cartesian coordinates), evaluate the fourier
transform of the corresponding orbital at given reciprocal vector ``q = k + G`` in cartesian
coordinates, using wannier90 conventions for `l` and `m`.

For the orbital ``g(r) = Rl(r)Y_l^m(r/|r|)`` the fourier transform is given by:

``\hat(g)(q) = 4\pi Y_l^m(-q/|q|)i^l * \int_{\mathbb{R}^+}r^2 R(r)j_l(|q|r)dr``
             = y_lm * intR_l

Only ``Rl(r) = r^l e^{-r^2/2}`` have been implemented.
"""
function eval_fourier_orbital(center, l::Integer, mr::Integer, q_cart)
    # TODO : Optimise to compute the whole list of q_vectors at once
    # instead of treating the case |q| = zero separatly
    if iszero(q_cart)
        (l == 0) && return (√(2)π)/2  # explicit value of y_0 * intR_0
        (l != 0) && return zero(eltype(q_cart)) # since j_l(0) = 0 for l≥1.
    end

    # |G| ≠ 0
    q_norm = norm(q_cart)
    arg_ylm = -q_cart ./ q_norm

    # Computes the phase prefactor due to center ≠ [0,0,0]
    phase_prefac = exp(-im*dot(q_cart,center))

    if l ≥ 0  # s,p,d or f
        m = retrieve_proper_m(l,mr)
        return (phase_prefac *
                (4π*im^l)*DFTK.ylm_real(l,m,arg_ylm) * intR_l(l,q_norm) )
    else      # hybrid orbitals
        if l == -3  # sp3
            s  = √(2π)/2 * q_norm * exp(-q_norm^2/2)
            px = (4π*im) * DFTK.ylm_real(1,1,arg_ylm)  * intR_l(1,q_norm)
            py = (4π*im) * DFTK.ylm_real(1,-1,arg_ylm) * intR_l(1,q_norm)
            pz = (4π*im) * DFTK.ylm_real(1,0,arg_ylm)  * intR_l(1,q_norm)

            (mr==1) && (return phase_prefac * (1/√2)*(s + px + py + pz))
            (mr==2) && (return phase_prefac * (1/√2)*(s + px - py - pz))
            (mr==3) && (return phase_prefac * (1/√2)*(s - px + py - pz))
            (mr==4) && (return phase_prefac * (1/√2)*(s - px - py + pz))
        end
    end
    error("No implemented orbital (s, p, d, f, sp3)
            match with the given quantum number")
end

"""
    Uses the above function to generate one Amn matrix given the projection table and
    usual informations on the system (basis etc...)
"""
function Ak_matrix_win_guess(basis::PlaneWaveBasis, ψ_k,
                                k_pt, n_bands, num_wann;
                                projs=[], centers=[], coords="")
    n_projs = size(projs,1)
    @assert n_projs == num_wann

    Ak = zeros(eltype(ψ_k), n_bands, n_projs)

    for n in 1:n_projs
        center, (l,mr,r_qn) = projs[n] #Extract data from projs[n]
        center = basis.model.lattice * center # lattice coords to cartesian coords

        # Obtain fourier coeff of projection g_n.
        q_cart = [k_pt.coordinate_cart + G_cart for G_cart in G_vectors_cart(k_pt)]
        coeffs_gn_per = [eval_fourier_orbital(center, l, mr, q) for q in q_cart]

        # Compute overlaps
        for m in 1:n_bands
            Ak[m,n] = @views dot(ψ_k[:,m], coeffs_gn_per)
        end
    end
    Ak
end

"""
Use the preceding functions on every k to generate the .amn file
"""
function write_amn(prefix::String, basis::PlaneWaveBasis, ψ, num_wann::Integer;
                           projs=[], centers=[], guess="")
    # Select guess
    if guess == "win"
        compute_Ak = Ak_matrix_win_guess
        # Check if the right number of projection is given...
        @assert num_wann == size(projs,1)
    else
        compute_Ak = Ak_matrix_gaussian_guess
        @assert num_wann == size(centers, 1)   # ... same for the number of centers.
    end

    # general parameters
    n_bands = size(ψ[1],2)
    k_size = size(ψ,1)

    # write file
    open("$prefix.amn", "w") do f
        # Header
        write(f,"Generated by DFTK at $(now())"*"\n")
        write(f,"$n_bands   $k_size   $num_wann \n")

        # Matrices
        for k in 1:k_size
            ψ_k = ψ[k]; k_pt = basis.kpoints[k];
            Ak = compute_Ak(basis, ψ_k, k_pt, n_bands, num_wann;
                              centers=centers, projs=projs)
            for n in 1:size(Ak,2)
                for m in 1:size(Ak,1)
                    write(f, @sprintf("%3i %3i %3i  %22.18f %22.18f \n",
                                     m, n, k, real(Ak[m,n]), imag(Ak[m,n])) )
                end
            end
        end
    end
    return nothing
end

write_amn(prefix, scfres, num_wann; projs=[], centers=[], guess="") =
    write_amn(prefix, scfres.basis, scfres.ψ, num_wann, projs=projs, centers=centers,
              guess=guess)

"""
   Random gaussian guess for MLWF used as default. Centers are generated in
   reduced coordinates (see notations and conventions).
"""
random_gaussian_guess(num_wann) = [rand(1,3) for i in 1:num_wann]

"""
    Wraps every step in a single routine by using the package wannier90_jll
"""
function run_wannier90(prefix::String, scfres, num_wann::Integer;
                       bands_plot=false,
                       guess="gaussian",
                       centers=random_gaussian_guess(num_wann),
                       kwargs...)

    @assert guess ∈ ("gaussian", "win")
    @info "Guess = $guess"

    # Unfold scfres to retrieve full k-point list
    scfres_unfold = unfold_bz(scfres)

    # Write wannier90 input file
    write_win(prefix, scfres_unfold, num_wann; bands_plot=bands_plot, kwargs...)

    # Wannier90 preprocessing task
    wannier90() do exe
        run(`$exe -pp $prefix`)
    end

    nn_kpts, nn_num, projs = read_nnkp(prefix)

    # Generate eig, amn and mmn files
    write_eig(prefix, scfres_unfold)
    write_amn(prefix, scfres_unfold, num_wann; centers=centers, projs=projs, guess=guess)
    write_mmn(prefix, scfres_unfold, nn_kpts, nn_num)

    # Wannierization
    wannier90() do exe
        @info "Wannier90 post-processing"
        run(`$exe $prefix`)
    end

    return nothing
end
