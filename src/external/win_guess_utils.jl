# TODO :
# For now does not converge

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
    m = zero(Int64)
    m_p = [0,1,-1]; m_d = [0,1,-1,2,-2]; m_f = [0,1,-1,2,-2,3,-3]
    (l == 0) && return 0
    (l == 1) && return m_p[mr]  # p
    (l == 2) && return m_d[mr]  # d
    (l == 3) && return m_f[mr]  # f
    error("Quantum numbers are not matching any implemented
                orbital (s,p,d,f)")
end


"""
    Evaluate the guess at the real vector `rvec`,
    given the (ish-)quantum numbers `l`,`mr`,`r` and `α` from Wannier90

    Uses the spherical harmonics implemented in src/common/spherical_harmonics.jl
    and adds linear combinations to provide hybrid orbitals  sp, sp2, sp3, sp3d and sp3d2.
    !!! FOR NOW JUST SP AND SP3 TO TEST CONVERGENCE.
"""
function quantum_number_to_guess(l::Integer, mr::Integer, r_qn::Integer, α,
                   rvec::AbstractVector{T}) where T

    Θ = zero(T); R = zero(T)
    
    # Angular parts
    if l ≥ 0  # s,p,d or f
        m = retrieve_proper_m(l,mr)
        Θ = DFTK.ylm_real(l,m,rvec)
    else      # hybrid
        s = DFTK.ylm_real(0,0,rvec)
        px = DFTK.ylm_real(1,1,rvec); py = DFTK.ylm_real(1,-1,rvec);
        pz = DFTK.ylm_real(1,0,rvec);
        if l==-1     # sp
            (mr==1) && (Θ = (1/√2)*(s + px))
            (mr==2) && (Θ = (1/√2)*(s - px))
        elseif l==-3 # sp3
            (mr==1) && (Θ = (1/√2)*(s + px + py + pz))
            (mr==2) && (Θ = (1/√2)*(s + px - py - pz))
            (mr==3) && (Θ = (1/√2)*(s - px + py - pz))
            (mr==4) && (Θ = (1/√2)*(s - px - py + pz))
        end
    end
    
    # Radial parts
    r = norm(rvec)
    if r <= 10 * eps(eltype(rvec))
        R = zero(T)
    else
        (r_qn==1) && (R = 2*α^(2/3)*exp(-α*r))
        (r_qn==2) && (R = (1/√8)*α^(3/2)*(2-α*r)*exp(-α*r/2))
        (r_qn==3) && (R = (4/√27)*α^(3/2)*(1-2*α*r/3 + 2*(α^2)*(r^2)/27)*exp(-α*r/3))
    end

    R*Θ
end


""" 
    Takes one line of the projections table given by read_nnkp_file and
    produce the fourier coefficients of the periodized guess gn_per on the G_vectors
    corresponding to the frequency k.
    Remember that : proj = [ [center], [quantum numbers], [z_axis], [x_axis], α ]
"""
function fourier_gn_per(basis::PlaneWaveBasis,r_cart,k::Integer,proj)
    center = proj[1]
    l,mr,r = proj[2]
    α = proj[5]

    # α = DFTK.atom_decay_length(Si)
    
    # TODO take into account non-canonical basis
    @assert(proj[3] == [0,0,1]) #For now we limit ourselves to the canonical basis
    @assert(proj[4] == [1,0,0])

    # Choose radial and angular parts
    gn(rvec) = quantum_number_to_guess(l,mr,r,α,rvec-center)
    real_gn = complex.([gn(rvec) for rvec in r_cart])

    # DEBUG : Shouldn't we normalize somewhere ?
    
    # Methode 1 : whithout normalization 
    # coeffs_gn_per = r_to_G(basis,basis.kpoints[k],real_gn)
    # coeffs_gn_per
    
    # Methode 2 : with normalization
    coeffs_gn_per = r_to_G(basis,real_gn)
    coeffs_gn_per ./= norm(coeffs_gn_per) #Whitout this line, both methods match
    index = [DFTK.index_G_vectors(basis,G) for G in G_vectors(basis.kpoints[k])]
    coeffs_gn_per[index]
    
end

"""
    Uses the above function to generate one Amn matrix given the projection table and
    usual informations on the system (basis etc...)
"""
function A_k_matrix_win_guesses(basis::PlaneWaveBasis, ψ,
                                k::Integer, n_bands::Integer, n_wann::Integer;
                                projs = [],centers = [], coords = "")
    n_projs = size(projs,1)
    @assert n_projs == n_wann
    
    r_cart = [r for (ir,r) in enumerate(r_vectors_cart(basis)) ]    
    A_k = zeros(Complex,(n_bands,n_projs))

    for n in 1:n_projs
        # Obtain fourier coeff of projection g_n
        coeffs_gn_per = fourier_gn_per(basis,r_cart,k,projs[n])
        # Compute overlaps
        for m in 1:n_bands
            coeffs_ψm = ψ[k][:,m]
            A_k[m,n] = dot(coeffs_ψm,coeffs_gn_per)
        end
    end

    A_k
        
end


# For testing on one projection
# test_proj =  [ [-0.12500,0.12500,-0.1250],[0,1,1],[0.0000,0.0000,1.0000],[1.00000,0.00000,0.00000],1.00 ]

