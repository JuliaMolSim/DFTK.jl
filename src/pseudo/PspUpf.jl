using Interpolations: interpolate, Gridded, Cubic, Linear
using SpecialFunctions: besselj
using QuadGK: quadgk
using UPF: load_upf

struct PspUpf{T} <: NormConservingPsp
    Zion::Int                         # Ionic charge (Z - valence electrons)
    lmax::Int                         # Maximal angular momentum in the non-local part
    rgrid::Vector{T}                  # Radial grid
    dr::T                             # Grid spacing
    lpot::Vector{T}                   # Local potential on the radial grid
    projs::Vector{Vector{Vector{T}}}  # BKB projectors: projs[l][i]
    h::Vector{Matrix{T}}              # Projector coupling coefficients per AM channel: h[l][i1,i2]
    identifier::String                # String identifying the PSP
end
charge_ionic(psp::PspUpf) = psp.Zion

"""
    PspUpf(Zion::Number, lmax::Number, rgrid::Vector, lpot::Vector,
           projs::Vector{Vector{Vector}}, h::Vector{Matrix}; identifier="")

Construct a Unified Pseudopotential Format pseudopotential. Currently only
norm-conserving potentials are supported.
"""
function PspUpf(Zion, lmax, rgrid, dr, lpot::Vector{T}, projs, h; identifier="") where {T}
    length(projs) == length(h) || error("Length of projs and h do not agree.")
    rgrid[2] - rgrid[1] == dr || error("dr doesn't match first difference in rgrid")

    PspUpf{T}(Zion, lmax, rgrid, dr, lpot, projs, h, identifier)
end

function parse_upf_file(path; identifier=path)
    pseudo = load_upf(path)

    lmax = pseudo["header"]["l_max"] 
    nproj = pseudo["header"]["number_of_proj"]
    beta_projectors = pseudo["beta_projectors"]
    Zion = Int(pseudo["header"]["z_valence"])

    # get number of projects for each AM
    n_projs = zeros(Int, lmax+1)
    for i in 1:nproj
        l = beta_projectors[i]["angular_momentum"]
        n_projs[l+1] += 1
    end

    # rgrid
    rgrid = pseudo["radial_grid"]
    dr = pseudo["radial_grid_derivative"][1]

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

    PspUpf(Zion, lmax, rgrid, dr, lpot, projs, h; identifier)
end

"""
    eval_psp_projector_real(psp::PspUpf, i::Number, l::Number, r::Number)

Evaluate the ith BKB projector with angular momentum l at real-space distance r
via linear interpolation on the real-space mesh.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    interp = interpolate((psp.rgrid,), psp.projs[l+1][i], Gridded(Linear()))
    interp(r)
end

"""
    eval_psp_projector_fourier(psp::PspUPF, i::Number, l::Number, q::Number)

Evaluate the ith BKB projector with angular momentum l in at k-space distance q.
More information on how ABINIT parses the file, see 
[m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90)
Lines: 800 - 825.
"""
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T) where {T <: Real}
    x(r::T)::T = 2π * r * q 
    rdep(r::T)::T = r^2 * eval_psp_projector_real(psp, i, l, r)

    # Here, must distinguish l==0 from others
    if l == 0
        bess(r::T)::T = -besselj(1, x(r))
    else
        bess(r::T)::T = besselj(l-1, x(r)) - (l+1) * besselj(l, x(r)) / x(r)
    end

    quadgk(r -> 2π * rdep(r) * bess(r), psp.rgrid[1], psp.rgrid[end]; rtol = 1e-5)[1]
end

"""
    eval_psp_local_real(psp::PspUpf, r::Number)

Evaluate the BKB local potential at real-space distance r via linear interpolation on the
real-space mesh.
"""
function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    interp = interpolate((psp.rgrid,), psp.lpot, Gridded(Linear()))
    interp(r)
end

"""
4π ∫[sin(2π*q*r) / (2π*q*r) * (V(r) + Z/r) * r^2 dr]
"""
function eval_psp_local_fourier_naive(psp::PspUpf, q::T)::T where {T <: Real}
    sin_term = sin.(2T(π) * q .* psp.rgrid) ./ (2T(π) * q)
    integrand = sin_term .* psp.rgrid .* (psp.rgrid .* psp.lpot .+ psp.Zion)
    return 4T(π) * ctrap(integrand, psp.dr)
end

"""
if q == 0
    4π ∫[(V(r) + Z/r) * r^2 dr]
else
    4π ( ∫[sin(r*√q) / q * (r*V(r) + Z*erf(r)) dr] - Z*exp(-q/4) / q)
"""
function eval_psp_local_fourier_qe(psp::PspUpf, q::T)::T where {T <: Real}
    if iszero(q)
        # aux(ir) = r(ir) * (r(ir) * vloc_at(ir) + zp * e2)
        integrand = psp.rgrid .* (psp.rgrid .* psp.lpot .+ psp.Zion)
        # call simpson(msh, aux, rab, vlcp)
        # vloc(1) = vlcp
        v_loc = ctrap(integrand, psp.dr)
    else
        # do ir = 1,msh
        #     aux1(ir) = r(ir) * vloc_at(ir) + zp * e2 * erf(r(ir))
        # end do
        pot_term = (psp.rgrid .* psp.lpot .+ psp.Zion .* erf.(psp.rgrid))
        # gx = sqrt(gl(igl) * tpiba2)  # sqrt(q * 2π/alat)
        # do ir = 1,msh
        #     aux(ir) = aux1(ir) * sin(gx * r(ir)) / gx
        # end do
        sin_term = (sin.(sqrt(q) .* psp.rgrid) ./ sqrt(q))
        integrand = sin_term .* pot_term
        # call simpson(msh, aux, rab, vlcp)
        v_loc = ctrap(integrand, psp.dr)
        # fac = zp * e2 / tpiba2  # zp * elec_chg^2 / (2π/alat)
        # vlcp = vlcp - fac * exp(-gl(igl) * tpiba2 * 0.25) / gl(igl)
        v_loc -= psp.Zion * exp(-q / 4) / q
    end
    # vloc = vloc * fpi / omega
    return 4T(π) * v_loc
end

"""
if q == 0
    -Z / π
else
    (2q * ∫[sin(2π*q*r) * (rV(r) + Z) dr] - Z / π) / q^2
"""
function eval_psp_local_fourier_abinit(psp::PspUpf, q::T)::T where {T <: Real}
    if iszero(q)
        # q2vq(1) = -zion / pi
        v_loc = -psp.Zion / T(π)
    else
        # do ir=1,mmax
        #     rvlpz(ir) = rad(ir) * vloc(ir) + zion
        # end do
        # arg = 2.d0 * pi * qgrid(iq)
        # do ir = 1,mmax
        #     work(ir) = sin(arg * rad(ir)) * rvlpz(ir)
        # end do
        integrand = sin.(2T(π)q .* psp.rgrid) .* (psp.rgrid .* psp.lpot .+ psp.Zion)
        # call ctrap(mmax, work, amesh, result)
        result = ctrap(integrand, psp.dr)
        # q2vq(iq) = q2vq(1) + 2.d0 * qgrid(iq) * result
        qsq_v_loc = -psp.Zion / T(π) + 2q * result
        v_loc = qsq_v_loc / q^2
    end
    return v_loc
end

"""
    ctrap(f::Vector, dx::Number)

Corrected trapezoidal method.
"""
function ctrap(f::Vector{T}, dx::T)::T where {T <: Real}
    a = dx * sum(i -> f[i], 1:length(f)-1)
    b = dx/2 * (f[begin] + f[end])
    return a + b
end

"""
    simpson_qe(f::Vector, dr::Number)

Simpson's method as implemented in QuantumESPRESSO.
"""
function simpson_qe(f::Vector{T}, dr::T)::T where {T <: Real}
    nr = length(f)
    res = sum(i -> 2 * abs(i % 2 - 2) * f[i] * dr, 1:nr)
    if nr % 2 == 1
        res += (f[1] * dr + f[nr] * dr) / 3.
    else
        res += (f[1] * dr + f[nr-1] * dr) / 3.
    end
    return res
end

"""
    ctrap_abinit(f::Vector, dx::Number)

Corrected trapezoidal method as implemented in Abinit.
"""
function ctrap_abinit(f::Vector{T}, dx::T)::T where {T <: Real}
    nf = length(f)
    if nf >= 10
        endpt = (
            T(23.75) * (f[1] + f[nf    ]) + 
            T(95.10) * (f[2] + f[nf - 1]) +
            T(55.20) * (f[3] + f[nf - 2]) +
            T(79.30) * (f[4] + f[nf - 3]) +
            T(70.65) * (f[5] + f[nf - 4])
        ) / 72
        
        if nf > 10
            ans = (sum(f[6:nf-5]) + endpt) * dx
        else
            ans = endpt * dx
        end
    elseif nf >= 8
        endpt = (
            17(f[1] + f[nf    ]) +
            59(f[2] + f[nf - 1]) +
            43(f[3] + f[nf - 2]) +
            49(f[4] + f[nf - 3])
        ) / 48
        if nf == 9
            ans = (f[5] + endpt) * dx
        else
            ans = endpt * dx
        end
    elseif nf == 7
        ans = (17(f[1] + f[7]) + 59(f[2] + f[6]) + 43(f[3] + f[5]) + 50(f[4])) / 48 * dx
    elseif nf == 6
        ans = (17(f[1] + f[6]) + 59(f[2] + f[5]) + 44(f[3] + f[4])           ) / 48 * dx
    elseif nf == 5
        ans = (  (f[1] + f[5]) +  4(f[2] + f[4]) +  2(f[3]       )           ) /  3 * dx
    elseif nf == 4
        ans = ( 3(f[1] + f[4]) +  9(f[2] + f[3])                             ) /  8 * dx
    elseif nf == 3
        ans = (  (f[1] + f[3]) +  4(f[2]       )                             ) /  3 * dx
    elseif nf == 2
        ans = (  (f[1] + f[2])                                               ) /  2 * dx
    elseif nf == 1
        ans = (  (f[1]       )                                               )      * dx
    end
    return ans
end