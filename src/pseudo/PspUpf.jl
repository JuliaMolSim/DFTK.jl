using Interpolations: linear_interpolation
using SpecialFunctions: sphericalbesselj
using UPF: load_upf

struct PspUpf{T} <: NormConservingPsp
    Zion::Int                         # Ionic charge (Z - valence electrons)
    lmax::Int                         # Maximal angular momentum in the non-local part
    rgrid::Vector{T}                  # Radial grid
    rab::Vector{T}
    vloc::Vector{T}                   # Local potential on the radial grid
    projs::Vector{Vector{Vector{T}}}  # Kleinman-Bylander β projectors: projs[l][i]
    h::Vector{Matrix{T}}              # Projector coupling coefficients per AM channel: h[l][i1,i2]
    identifier::String                # String identifying the PSP
end
charge_ionic(psp::PspUpf) = psp.Zion

"""
    PspUpf(Zion::Number, lmax::Number, rgrid::Vector, vloc::Vector,
           projs::Vector{Vector{Vector}}, h::Vector{Matrix}; identifier="")

Construct a Unified Pseudopotential Format pseudopotential. Currently only
norm-conserving potentials are supported.
"""
function PspUpf(Zion, lmax, rgrid, rab,
                vloc::Vector{T}, projs, h; identifier="") where {T}
    length(projs) == length(h) || error("Length of projs and h do not agree.")

    PspUpf{T}(Zion, lmax, rgrid, rab, vloc, projs, h, identifier)
end

function parse_upf_file(path; identifier=path)
    pseudo = load_upf(path)

    # Maximum angular momentum channel
    lmax = pseudo["header"]["l_max"] 
    # Number of Kleinman-Bylander β projectors
    nproj = pseudo["header"]["number_of_proj"]
    # Kleinman-Bylander β projectors
    beta_projectors = pseudo["beta_projectors"]
    # Pseudo-ion charge
    Zion = Int(pseudo["header"]["z_valence"])
    # Number of projectors for each angular momentum channel
    n_projs = zeros(Int, lmax+1)
    for i in 1:nproj
        l = beta_projectors[i]["angular_momentum"]
        n_projs[l+1] += 1
    end
    # Radial grid
    rgrid = pseudo["radial_grid"]
    rab = pseudo["radial_grid_derivative"]
    # Local potential
    vloc = pseudo["local_potential"]
    # Kleinman-Bylander β projectors
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
    # Dij -> h
    dij = pseudo["D_ion"]
    n = 1
    h = []
    for l in 0:lmax
        nprojs = n_projs[l+1]
        push!(h, dij[n:n+nprojs-1, n:n+nprojs-1])
        n += nprojs
    end

    PspUpf(Zion, lmax, rgrid, rab, vloc, projs, h; identifier)
end

abstract type LocalCorrection end
function correction_real(C::LocalCorrection, psp::PspUpf{T})::T where {T <: Real}
    return zero(T)
end
function correction_fourier(C::LocalCorrection, psp::PspUpf{T}, q::T)::T where {T <: Real}
    return zero(T)
end
struct InvrCorrection <: LocalCorrection end
function correction_real(C::InvrCorrection, psp::PspUpf{T}, r::T)::T where {T <: Real}
    return -(2*psp.Zion)
end
function correction_fourier(C::InvrCorrection, psp::PspUpf{T}, q::T)::T where {T <: Real}
    return -(2*psp.Zion) / q^2
end
struct ErfrInvrCorrection <: LocalCorrection end
function correction_real(C::ErfrInvrCorrection, psp::PspUpf{T}, r::T)::T where {T <: Real}
    return -(2*psp.Zion) * erf(r)
end
function correction_fourier(C::ErfrInvrCorrection, psp::PspUpf{T}, q::T)::T where {T <: Real}
    return -(2*psp.Zion) * exp(-q^2) / q^2
end

import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(C::LocalCorrection) = Ref(C)

"""
    eval_psp_projector_real(psp::PspUpf, i::Number, l::Number, r::Number)

Evaluate the ith Kleinman-Bylander β projector with angular momentum l at real-space distance r
via cubic spline interpolation on the real-space mesh.

!!! Note
UPFs store `rgrid[i] * β[l,n](rgrid[i])`, so we must divide by `r`.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    ir_cut = length(psp.projs[l+1][i])
    linear_interpolation((psp.rgrid[1:ir_cut],), psp.projs[l+1][i])(r) / r
end

"""
    eval_psp_projector_fourier(psp::PspUPF, i::Number, l::Number, q::Number)
Evaluate the ith Kleinman-Bylander β projector with angular momentum l in at k-space distance q.

!!! Note
UPFs store `rgrid[i] * β[l,n](rgrid[i])`, so the integrand has `r` instead of `r^2`
"""
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T)::T where {T <: Real}
    ir_cut = length(psp.projs[l+1][i])
    rgrid = view(psp.rgrid, 1:ir_cut)
    proj = view(psp.projs[l+1][i], 1:ir_cut)
    x = q .* rgrid
    if iszero(q)
        jl = sphericalbesselj.(1, x)
    else
        jl = sphericalbesselj.(l, x)
    end
    integrand = jl .* proj .* rgrid  # jl(qr) * rβ * r
    proj_q = 4T(π) * ctrap(integrand, psp.rab)
    return proj_q
end

"""
    eval_psp_local_real(psp::PspUpf, r::Number)

Evaluate the local potential at real-space distance r via cubic spline interpolation on the
real-space mesh.
"""
function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    linear_interpolation((psp.rgrid,), psp.vloc)(r) / 2  # Ry -> Ha
end

function eval_psp_local_fourier(psp::PspUpf, q::T)::T where {T <: Real}
    vloc = psp.vloc  # in Rydberg
    Z = psp.Zion * 2 # for some strange reason
    # -Z/r correction (canceling `r` in the code):
    #   F[V(r) - -Z/r] / 4π = ∫ sin(qr) / (qr) * (V(r) - -Z/r) r^2
    sin_term = sin.(q .* psp.rgrid) ./ q 
    pot_term = psp.rgrid .* vloc
    corr_real = -Z
    vloc_corr_ignd = sin_term .* (pot_term .- corr_real)
    vloc_corr_itgl = ctrap(vloc_corr_ignd, psp.rab)
    # F[-Z/r] / 4π = -Z / q^2
    corr_fourier = -Z / q^2
    vloc = 4T(π) * (vloc_corr_itgl + corr_fourier)
    return vloc / 2  # in Ha
end

function eval_psp_local_fourier(
    c::LocalCorrection,
    integrator::Function,
    psp::PspUpf,
    q::T
)::T where {T <: Real}
    vloc = psp.vloc  # in Rydberg
    # F[V(r) - correction(r)] / 4π = ∫ sin(qr) / (qr) * (V(r) - correction(r)) r^2
    sin_term = sin.(q .* psp.rgrid) ./ q  # 1/r canceled with r^2
    # correction contains 1/r, and this is canceled with the remaining r
    pot_term = (psp.rgrid .* vloc)
    corr_real =correction_real.(c, psp, psp.rgrid)
    vloc_corr_ignd = sin_term .* (pot_term .- corr_real)
    vloc_corr_itgl = integrator(vloc_corr_ignd, psp.rab)
    corr_fourier = correction_fourier(c, psp, q)
    vloc = 4T(π) * (vloc_corr_itgl + corr_fourier)  # still in Rydberg
    return vloc / 2  # in Hartree
end

function eval_psp_energy_correction(T, psp::PspUpf, n_electrons)
    Z = psp.Zion * T(2)  # for some strange reason
    pot_term = psp.rgrid .* (psp.rgrid .* psp.vloc .- -Z)
    ignd = n_electrons .* pot_term
    return 4T(π) * ctrap(ignd, psp.rab) / T(2)
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

function ctrap(f::Vector{T}, dx::Vector{T})::T where {T <: Real}
    a = sum(i -> f[i] * dx[i], 1:length(f)-1)
    b = 1/T(2) * (f[begin]*dx[begin] + f[end]*dx[end])
    return a + b
end

"""
    simpson_qe(f::Vector, dr::Number)

Simpson's method as implemented in QuantumESPRESSO.
"""
function simpson_qe(f::Vector{T}, dx::Vector{T})::T where {T <: Real}
    nr = length(f)
    res = sum(i -> 2 * abs(i % 2 - 2) * f[i] * dx[i], 1:nr)
    if nr % 2 == 1
        res += (f[1] * dx[1] + f[nr] * dx[nr]) / 3.
    else
        res += (f[1] * dx[1] + f[nr-1] * dx[nr]) / 3.
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