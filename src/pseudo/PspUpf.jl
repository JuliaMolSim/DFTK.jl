using Interpolations: interpolate, Gridded, Cubic, Linear
using SpecialFunctions: besselj
using QuadGK: quadgk
using UPF: load_upf

struct PspUpf{T} <: NormConservingPsp
    Zion::Int             # Ionic charge (Z - valence electrons)
    lmax::Int # Maximal angular momentum in the non-local part
    rgrid::Vector{T} # radial grid start from 0.0, mesh point
    lpot::Vector{T} # local potential in the radial grid
    projs::Vector{Vector{Vector{T}}} # BKB projector per AM per projector
    h::Vector{Matrix{T}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
    identifier::String    # String identifying the PSP
end
charge_ionic(psp::PspUpf) = psp.Zion

"""
    PspUpf
    (norm conversiong only at the moment)
"""
function PspUpf(Zion, lmax, rgrid, lpot::Vector{T}, projs, h; identifier="") where {T}
    # TODO: validate the dim of arguments
    lmax = length(h) - 1
    PspUpf{T}(Zion, lmax, rgrid, lpot, projs, h, identifier)
end

function parse_upf_file(path; identifier=path)
    pseudo = load_upf(path)

    lmax = pseudo["header"]["l_max"] 
    tot_num_projs = pseudo["header"]["number_of_proj"]
    beta_projectors = pseudo["beta_projectors"]
    Zion = Int(pseudo["header"]["z_valence"])

    # get number of projects for each AM
    n_projs = zeros(Int, lmax+1)
    for i in 1:tot_num_projs
        beta = beta_projectors[i]

        l = beta["angular_momentum"]
        n_projs[l+1] += 1
    end

    # rgrid
    rgrid = pseudo["radial_grid"]

    # lpot
    lpot = pseudo["local_potential"]

    # projectors
    projs = []
    idx = 1
    for l in 0:lmax
        l_projs = []
        for _ in 1:n_projs[l+1]
            beta = beta_projectors[idx]
            rdata = beta["radial_function"]
            push!(l_projs, rdata)
            idx += 1
        end
        push!(projs, l_projs)
    end
    
    # dij -> h
    dij = pseudo["D_ion"]
    n = 1
    h = []
    for l in 0:lmax
        nprojs = n_projs[l+1]
        push!(h, dij[n:n+nprojs-1, n:n+nprojs-1])
        n += nprojs
    end

    PspUpf(Zion, lmax, rgrid, lpot, projs, h; identifier)
end

"""
Read and interpolate value of projector in real space at r.
i-th projector of AM l. stored in file. Read from UPF file and interpolate.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    interp = interpolate((psp.rgrid,), psp.projs[l+1][i], Gridded(Linear()))
    interp(r)
end

"""
Creating the projector function in kspace
More information on how ABINIT parses the file, see the [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90) page Lines: 800 - 825
"""
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T) where {T <: Real}
    x(r) = 2π * r * q 
    rdep(r) = r^2 * eval_psp_projector_real(psp, i, l, r)

    # Here, must distinguish l==0 from others
    if l == 0
        bess(r) = -besselj(1, x(r))
    else
        bess(r) = besselj(l-1, x(r)) - (l+1) * besselj(l,x(r))/x(r)
    end

    quadgk(r -> 2π * rdep(r) * bess(r), psp.rgrid[1], psp.rgrid[end]; rtol = 1e-5)[1]
end

function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    interp = interpolate((psp.rgrid,), psp.lpot, Gridded(Linear()))
    interp(r)
end

#This was how ABINIT calculated their local fourier potential. 
# See https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90 Lines: 434,435
@doc raw"""
Local potential in inverse space. 
Calculated with the Hankel transformation: 
4π∫(\frac{sin(2π q r)}{2π q r})(r^2 v(r)+r Zv)dr.
"""
function eval_psp_local_fourier(psp::PspUpf, q::T) where {T <: Real}
    j0(r) = sin(2π * q * r)/(2π * q)
    # local Bessel
    f(r) = j0(r) * (r * eval_psp_local_real(psp, r) + psp.Zion)

    # return the integral
    4π * quadgk(f, psp.rgrid[1], psp.rgrid[end]; order = 17, rtol = 1e-8)[1]
end