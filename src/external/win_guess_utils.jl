# TODO :
# Tested and certified for s orbitals
# For now the gn_per coeffs are not normalized but it has
# no effect on the convergence a priori

using SpecialFunctions

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
    Series expansion at order n of int_Rl for l=1.
    Use Gamma function to avoid overflow errors
"""
function approx_radial_1(Gnorm,n)
    sum = zero(Float64)
    for i in 1:n  # 9 is maximum iteration without overflow
        sum += (-1)^(i+1) * 2^(i+1) *  Gnorm^(2i-1) * i * (gamma(i+1)/gamma(2i+2))
    end
    sum
end



@doc raw"""
    Gives the analytic expression of the
    integral ``I_l(G) = \int_{\mathbb{R}^+} r^2 exp(-r^2/2) j_l(|G|r)dr``
    where ``j_l`` is the Bessel function of order l.
    Only ``0 ≤ l ≤ 3`` are needed so any other value raises an error 
    G is expected in cartesian coordinates
"""
function gaussian_intR_l(l::Integer, Gcart)
    if iszero(Gcart) # |G| = 0
        (l==0) && return complex(√(π/2))
        (l!=0) && return zero(ComplexF64)
    else # |G| > 0
        Gnorm = norm(Gcart)
        (l==0) && return Gnorm*√(π/2)*exp(-(Gnorm^2)/2) #tested !
        (l==1) && return approx_radial_1(Gnorm,14)
        (l==2) && return 0 # Not yet implemented
        (l==3) && return 0 # Not yet implemented
    end
    error("l has to be bewteen 0 and 3")
end


@doc raw"""
Given quantum numbers and center (in cartesian coordinates), evaluate the fourier
transform of the corresponding orbital at given reciprocal vector ``G`` in cartesian
coordinates.

For the orbital ``g(r) = R(r)Y_l^m(r/|r|)`` the fourier transform is given by:

``\hat(g)(G) = 4\pi Y_l^m(-G/|G|)i^l * \int_{\mathbb{R}^+}r^2 R(r)j_l(|G|r)dr``
             = y_lm * intR_l

Only gaussian ``R(r) = e^{-r^2/2}`` have been implemented.
The `y_lm` and `intR_l` terms  are respectively given by the routines 
`ylm_real` and `gaussian_intR_l`.

TODO : y_lm and intR_l not needed
"""
function eval_fourier_orbital(center,l,mr, Gcart)
    # TODO : Optimise to compute the whole list of G_vectors at once
    y_lm = zero(ComplexF64); intR_l = zero(ComplexF64)

    # case |G| treated separatly
    if iszero(Gcart)
        (l == 0) && return (√(2)π)/2
        (l != 0) && return zero(ComplexF64) # since j_l(0) = 0 for l≥1.
    end

    # |G| ≠ 0
    Gnorm = norm(Gcart)
    arg_ylm = -Gcart ./ Gnorm

    # Computes the phase prefactor due to center ≠ [0,0,0]
    phase_prefac = exp(-im*dot(Gcart,center))

    # y_lm and intR_l
    if l ≥ 0  # s,p,d or f
        m = retrieve_proper_m(l,mr)
        y_lm = (4π*im^l)*DFTK.ylm_real(l,m,arg_ylm)
        intR_l = gaussian_intR_l(l,Gcart)
        return phase_prefac*y_lm*intR_l
    else      # hybrid orbitals
        s  = √(2π)/2 * Gnorm * exp(-Gnorm^2/2)
        px = (4π*im) * DFTK.ylm_real(1,1,arg_ylm)  * gaussian_intR_l(1,Gcart)
        py = (4π*im) * DFTK.ylm_real(1,-1,arg_ylm) * gaussian_intR_l(1,Gcart)
        pz = (4π*im) * DFTK.ylm_real(1,0,arg_ylm)  * gaussian_intR_l(1,Gcart)
        if  l == -1     # sp
            (mr==1) && (y_lm = (1/√2) * (s + px))
            (mr==2) && (y_lm = (1/√2) * (s - px))
        elseif l == -3  # sp3
            (mr==1) && (return phase_prefac * (1/√2)*(s + px + py + pz))
            (mr==2) && (return phase_prefac * (1/√2)*(s + px - py - pz))
            (mr==3) && (return phase_prefac * (1/√2)*(s - px + py - pz))
            (mr==4) && (return phase_prefac * (1/√2)*(s - px - py + pz))
        end
    end
  
    error("No orbital match with the given quantum number")
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
    
    A_k = zeros(Complex,(n_bands,n_projs))

    # All G vectors in cartesian coordinates.
    Gs_cart_k = [basis.model.recip_lattice*G for G in basis.kpoints[k].G_vectors]
    Gs_cart = [G for (i,G) in enumerate(G_vectors_cart(basis))]
    
    for n in 1:n_projs
        center, (l,mr,r_qn) = projs[n]
        center =  basis.model.lattice*center # lattice coords to cartesian coords
        # Obtain fourier coeff of projection g_n
        coeffs_gn_per = [eval_fourier_orbital(center,l,mr,Gcart) for Gcart in Gs_cart_k]
        coeffs_gn_per ./= norm(coeffs_gn_per)
        
        # Compute overlaps
        for m in 1:n_bands
            coeffs_ψm = ψ[k][:,m]
            A_k[m,n] = dot(coeffs_ψm, coeffs_gn_per)
        end
    end

    A_k
        
end



# For testing on one projection
# test_proj =  [ [-0.12500,0.12500,-0.1250],[0,1,1],[0.0000,0.0000,1.0000],[1.00000,0.00000,0.00000],1.00 ]

