using Interpolations: cubic_spline_interpolation
using SpecialFunctions: besselj
using QuadGK: quadgk

struct PspUpf{T} <: NormConservingPsp
    Zion::Int             # Ionic charge (Z - valence electrons)
    lmax::Int # Maximal angular momentum in the non-local part
    rgrid::Vector{T} # radial grid start from 0.0, mesh point
    lpot::Vector{T} # local potential in the radial grid
    projs::Vector{Vector{Vector{T}}} # BKB projector per AM per projector
    h::Vector{Matrix{T}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
    identifier::String    # String identifying the PSP
    description::String   # Descriptive string
end
charge_ionic(psp::PspHgh) = psp.Zion

function parse_upf_file(path; identifier=path)
    # use UPF.jl by austin
    # h <- dij
end

"""
Read and interpolate value of projector in real space at r.
i-th projector of AM l. stored in file. Read from UPF file and interpolate.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    interp = cubic_spline_interpolation(psp.rgrid, psp.projs[l][i])
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

    return @timeit to "integ psp local projector" quadgk(r -> 2π * rdep(r) * bess(r), psp.rgrid[1], psp.rgrid[end]; rtol = 1e-5)[1]
end

function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    interp = cubic_spline_interpolation(psp.rgrid, psp.lpot)
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
    @timeit to "integ psp local fourier" 4π * quadgk(f, psp.rgrid[1], psp.rgrid[end]; order = 17, rtol = 1e-8)[1]
end