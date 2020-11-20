# TODO :
# This code is obsolete. It should be fixed by using the spherical harmonics
# already implemented in DFTK. See /src/common/spherical_harmonics.jl

# This is the part dedicated to "win" guesses
"""
Generate the dictionary associating quantum numbers r,l and mr with corrsponding radial and angular parts of hydrogen AOs.
(See Tables 3.1 and 3.2 of Wannier90 user_guide.)
Return two dictionaries, radial_parts and angular_parts. Needed to generate guesses from the projection block in the .nnkp file.
"""



# function dict_AOs(α)

#     # Radial parts of hydrogen AOs in a dictionary : quantum number r -> AO radial part
#     radial_parts = Dict()
#     push!(radial_parts, 1 => r -> 2*α^(3/2)*exp(-α*r) ) 
#     push!(radial_parts, 2 => r -> (1/(2*√2)) * α^(3/2) * (2-α*r) * exp(-α*r/2) )
#     push!(radial_parts, 3 => r -> √(4/27) * α^(3/2) * (1-2*α*r/3+2*(α^2)*(r^2)/27) * exp(-α*r/3) )
    
#     # Basic blocks of hydrogen AOs' angular parts
#     s(θ,φ) = 1/√(4*pi)
#     pz(θ,φ) = √( 3/(4*π) )*cos(θ)
#     px(θ,φ) = √( 3/(4*π) )*sin(θ)*cos(φ)
#     py(θ,φ) = √( 3/(4*π) )*sin(θ)*sin(φ)

#     dz2(θ,φ) = √(5/(16π))*(3*cos(θ)^2 -1)
#     dx2(θ,φ) = √(15/(4π))*sin(θ)*cos(θ)*cos(φ)
#     dx2_y2(θ,φ) = √(15/(16π))*(sin(θ)^2)*cos(2*φ)

#     # All angular parts in a dictionary : quantum numbers [l,mr] => AO angular part
#     angular_parts = Dict()
#     push!(angular_parts, [0,1] => s)
#     push!(angular_parts, [1,1] => pz)
#     push!(angular_parts, [1,2] => px)
#     push!(angular_parts, [1,3] => py)

#     push!(angular_parts, [2,1] => dz2 )                                        #dz2
#     push!(angular_parts, [2,2] => dx2 )                                        #dxz
#     push!(angular_parts, [2,3] => (θ,φ) -> √(15/(4π))*sin(θ)*cos(θ)*sin(φ) )   #dyz
#     push!(angular_parts, [2,4] => dx2_y2 )                                     #dx2-y2
#     push!(angular_parts, [2,5] => (θ,φ) -> √(15/(16π))*(sin(θ)^2)*sin(2*φ) )   #dxy

#     push!(angular_parts, [3,1] => (θ,φ) -> (√7/(4*√(2π)))  * (5*cos(θ)^3 - 3*cos(θ)) )                   #fz3  
#     push!(angular_parts, [3,2] => (θ,φ) -> (√21/(4*√(2π))) * (5*cos(θ)^2 - 1)*sin(θ)*cos(φ) )            #fxz2
#     push!(angular_parts, [3,3] => (θ,φ) -> (√21/(4*√(2π))) * (5*cos(θ)^2 - 1)*sin(θ)*sin(φ) )            #fyz2
#     push!(angular_parts, [3,4] => (θ,φ) -> (√105/(4*√π))   * (sin(θ)^2)*cos(θ)*cos(2*φ) )                #fz(x2-y2)
#     push!(angular_parts, [3,5] => (θ,φ) -> (√105/(4*√π))   * (sin(θ)^2)*cos(θ)*sin(2*φ) )                #fxyz
#     push!(angular_parts, [3,6] => (θ,φ) -> (√35/(4*√(2π))) * (sin(θ)^3)*(cos(φ)^2 - 3*sin(φ)^2)*cos(φ) ) #fx(x2-3y2)
#     push!(angular_parts, [3,7] => (θ,φ) -> (√35/(4*√(2π))) * (sin(θ)^3)*(3*cos(φ)^2 - sin(φ)^2)*sin(φ) ) #fy(3x2-y2)
        
#     push!(angular_parts, [-1,1] => (θ,φ) -> (1/√2)*s + (1/√2)*px ) #sp-1 
#     push!(angular_parts, [-1,2] => (θ,φ) -> (1/√2)*s - (1/√2)*px ) #sp-2
    
#     push!(angular_parts, [-2,1] => (θ,φ) -> (1/√3)*s - (1/√6)*px + (1/√2)*py) #sp2-1
#     push!(angular_parts, [-2,2] => (θ,φ) -> (1/√3)*s - (1/√6)*px - (1/√2)*py) #sp2-2
#     push!(angular_parts, [-2,3] => (θ,φ) -> (1/√3)*s + (2/√6)*px )            #sp2-3
    
#     push!(angular_parts, [-3,1] => (θ,φ) -> 0.5*(s(θ,φ)+px(θ,φ)+py(θ,φ)+pz(θ,φ)) )  #sp3-1
#     push!(angular_parts, [-3,2] => (θ,φ) -> 0.5*(s(θ,φ)+px(θ,φ)-py(θ,φ)-pz(θ,φ)) )  #sp3-2
#     push!(angular_parts, [-3,3] => (θ,φ) -> 0.5*(s(θ,φ)-px(θ,φ)+py(θ,φ)-pz(θ,φ)) )  #sp3-3
#     push!(angular_parts, [-3,4] => (θ,φ) -> 0.5*(s(θ,φ)-px(θ,φ)-py(θ,φ)+pz(θ,φ)) )  #sp3-4

#     push!(angular_parts, [-4,1] => (θ,φ) ->  (1/√3)*s - (1/√6)*px + (1/√2)*py)  #sp3d-1
#     push!(angular_parts, [-4,2] => (θ,φ) ->  (1/√3)*s - (1/√6)*px - (1/√2)*py)  #sp3d-2
#     push!(angular_parts, [-4,3] => (θ,φ) ->  (1/√3)*s + (2/√6)*px)              #sp3d-3
#     push!(angular_parts, [-4,4] => (θ,φ) ->  (1/√2)*pz + (1/√2)*dz2)            #sp3d-4
#     push!(angular_parts, [-4,5] => (θ,φ) -> -(1/√2)*pz + (1/√2)*dz2)            #sp3d-5


#     push!(angular_parts, [-5,1] => (θ,φ) -> (1/√6)*s - (1/√2)*px - (1/√12)*dz2 +  (1/2)*dx2_y2) #sp3d2-1
#     push!(angular_parts, [-5,2] => (θ,φ) -> (1/√6)*s + (1/√2)*px - (1/√12)*dz2 +  (1/2)*dx2_y2) #sp3d2-2
#     push!(angular_parts, [-5,3] => (θ,φ) -> (1/√6)*s - (1/√2)*py - (1/√12)*dz2 -  (1/2)*dx2_y2) #sp3d2-3
#     push!(angular_parts, [-5,4] => (θ,φ) -> (1/√6)*s + (1/√2)*py - (1/√12)*dz2 -  (1/2)*dx2_y2) #sp3d2-4
#     push!(angular_parts, [-5,5] => (θ,φ) -> (1/√6)*s - (1/√2)*pz + (1/√3)*dz2)                  #sp3d2-5
#     push!(angular_parts, [-5,6] => (θ,φ) -> (1/√6)*s + (1/√2)*pz + (1/√3)*dz2)                  #sp3d2-6
    
 

#     radial_parts,angular_parts
# end



""" 
    Takes one line of the projections table given by read_nnkp_file and  dictionaries linking quantum numbers
    and radial or angular parts of hydrogen AOs, and produce the analytic expression of the guess.
    Remember that : proj = [ [center], [quantum numbers], [z_axis], [x_axis], α ]
"""
function guess_win(proj)
    center = proj[1]
    l,mr,r = proj[2]
    α = proj[5]

    radial_parts,angular_parts = dict_AOs(α)
    # TODO take into account non-canonical basis
    @assert(proj[3] == [0,0,1]) #For now we limit ourselves to the canonical basis
    @assert(proj[4] == [1,0,0])

    # Choose radial and angular parts
    R = get!(radial_parts,r,1)
    Θ = get!(angular_parts,[l,mr],1)
    
    g_n(r,θ,φ) = R(r)*Θ(θ,φ)

    g_n
    
end

"""
Convert cartesian to spherical coordinates.
"""
function cart_to_spherical(cart_coord)
    x,y,z = cart_coord
    r = norm(cart_coord,2)
    if r != 0.0
        φ = atan(y,x)                 #atan(x,y) gives atan(x/y) and selects the right quadrant
        θ = atan(norm([x,y],2),z)
    else
        φ = 0.0
        θ = 0.0
    end
    [r,φ,θ]
end



function A_k_matrix_win_guesses(pw_basis::PlaneWaveBasis,ψ,k; projs = [],centers = [])
    n_bands = size(ψ[1][1,:],1)
    n_projs = size(projs,1)
    
    r_cart = [pw_basis.model.lattice*r for (i,r) in enumerate(r_vectors(pw_basis)) ]
    r_sph  = cart_to_spherical.(r_cart)
    
    A_k = zeros(Complex,(n_bands,n_projs))

    for n in 1:n_projs
        # Obtain fourier coeff of projection g_n
        gn = guess_win(projs[n])
        real_gn = complex.([gn(r,θ,φ) for (r,θ,φ) in r_sph])
        coeffs_g_per_n = r_to_G(pw_basis,pw_basis.kpoints[k],real_gn)

        # Compute overlaps
        for m in 1:n_bands
            coeffs_ψm = ψ[k][:,m]
            A_k[m,n] = dot(coeffs_ψm,coeffs_g_per_n)
        end
    end

    A_k
        
end

# End of "win" guess part
