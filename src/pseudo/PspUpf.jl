using LinearAlgebra
using Interpolations: linear_interpolation
using PseudoPotentialIO: load_upf

struct PspUpf{T,IT} <: NormConservingPsp
    Zion::Int                             # Ionic charge (Z - valence electrons)
    lmax::Int                             # Maximal angular momentum in the non-local part

    rgrid::Vector{T}                      # Radial grid
    drgrid::Vector{T}                     # Radial grid derivative / integration factor

    vloc::Vector{T}                       # Local potential on the radial grid
    vloc_interp::IT                       # Local potential interpolator

    projs::Vector{Vector{Vector{T}}}      # Kleinman-Bylander β projectors: projs[l+1][i]
    projs_interp::Vector{Vector{IT}}      # Projector interpolator
    h::Vector{Matrix{T}}                  # Projector coupling coefficients per AM channel: h[l+1][i1,i2]

    pswfcs::Vector{Vector{Vector{T}}}     # Pseudo-atomic wavefunctions
    pswfc_occs::Vector{Vector{T}}         # Occupations of the pseudo-atomic wavefunctions

    rhoatom::Vector{T}                    # Pseudo-atomic charge density

    rvlocpzdr::Vector{T}                  # (r_i V_{loc}(r_i) + Z) dr_i
    r2projsdr::Vector{Vector{Vector{T}}}  # r_j^2 β_{li}(r_j) dr_i

    identifier::String                    # String identifying the PSP
end

"""
    PspUpf(Zion::Number, lmax::Number, rgrid::Vector, drgrid::Vector, vloc::Vector,
           vloc_interp, projs::Vector{Vector{Vector}}, projs_interp, h::Vector{Matrix},
           pswfcs::Vector{Vector}, pswfc_occs::Vector, rhoatom::Vector, rvlocpzdr::Vector,
           r2projsdr::Vector{Vector{Vector}}; identifier="")

Construct a Unified Pseudopotential Format pseudopotential. Currently only
norm-conserving potentials are supported.
"""
function PspUpf(Zion, lmax, rgrid::Vector{T}, drgrid, vloc, vloc_interp::IT, projs,
                projs_interp, h, pswfcs, pswfc_occs, rhoatom, rvlocpzdr, r2projsdr;
                identifier="") where {T, IT}
    PspUpf{T,IT}(Zion, lmax, rgrid, drgrid, vloc, vloc_interp, projs, projs_interp, h,
                 pswfcs, pswfc_occs, rhoatom, rvlocpzdr, r2projsdr, identifier)
end

function parse_upf_file(path; identifier=path)
    pseudo = load_upf(path)

    unsupported = []
    pseudo["header"]["core_correction"]      && push!(unsupported, "non-lin. core correction")
    pseudo["header"]["has_so"]               && push!(unsupported, "spin-orbit coupling")
    pseudo["header"]["pseudo_type"] == "SL"  && push!(unsupported, "semilocal potential")
    pseudo["header"]["pseudo_type"] == "US"  && push!(unsupported, "ultrasoft")
    pseudo["header"]["pseudo_type"] == "PAW" && push!(unsupported, "plane-augmented wave")
    pseudo["header"]["has_gipaw"]            && push!(unsupported, "gipaw data")
    pseudo["header"]["pseudo_type"] == "1/r" && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported" *
                                     " features/quantities: $unsupported")

    Zion        = Int(pseudo["header"]["z_valence"])     # Pseudo-ion charge
    rgrid       = pseudo["radial_grid"]                  # Radial grid
    drgrid      = pseudo["radial_grid_derivative"]       # Integration coeff.s
    lmax        = pseudo["header"]["l_max"]              # Maximum angular momentum channel
    vloc        = pseudo["local_potential"] ./ 2         # Local potential (Ry -> Ha)
    vloc_interp = linear_interpolation((rgrid, ), vloc)  # Interp.or for the local potential

    # Kleinman-Bylander β projectors projs[l+1][i][ir]
    # NB: UPFs store rβ, not just β
    projs = map(0:lmax) do l
        betas_l = filter(beta -> beta["angular_momentum"] == l, pseudo["beta_projectors"])
        map(beta -> beta["radial_function"] ./ 2, betas_l)  # Ry -> Ha
    end
    projs_interp = map(projs) do projs_l
        map(projs_l) do proj
            ir_cut = lastindex(proj)
            linear_interpolation((rgrid[1:ir_cut], ), proj)
        end
    end

    # β-projector coupling coefficients h[l+1][i,j] (also called e_{KB}, D_{ij})
    h = Matrix[]
    n = 1
    for l = 0:lmax
        nproj_l = length(projs[l+1])
        Dij_l = pseudo["D_ion"][n:n+nproj_l-1, n:n+nproj_l-1] .* 2  # 1/Ry -> 1/Ha
        push!(h, Dij_l)
        n += nproj_l
    end

    # Pseudo-atomic wavefunctions
    # Currently not used; can be used for initializing the starting wavefunction and as
    # projectors for projected densities of states, projected wavefunctions, and DFT+U(+V).
    pswfcs = map(0:lmax - 1) do l
        pswfcs_l = filter(pseudo["atomic_wave_functions"]) do pswfc
            pswfc["angular_momentum"] == l
        end
        map(pswfc -> pswfc["radial_function"], pswfcs_l)
    end
    pswfc_occs = map(0:lmax - 1) do l
        pswfcs_l = filter(pseudo["atomic_wave_functions"]) do pswfc
            pswfc["angular_momentum"] == l
        end
        map(pswfc -> pswfc["occupation"], pswfcs_l)
    end

    # Currently not used; can be used for initializing the starting guess density.
    rhoatom = pseudo["total_charge_density"]

    # Useful precomputed quantities (q-independent parts of Fourier transform integrands)
    rvlocpzdr = (rgrid .* vloc .+ Zion) .* drgrid
    r2projsdr = map(projs) do projs_l
        map(projs_l) do proj
            ir_cut = lastindex(proj)
            rgrid[1:ir_cut] .* proj .* drgrid[1:ir_cut]
        end
    end

    PspUpf(Zion, lmax, rgrid, drgrid, vloc, vloc_interp, projs, projs_interp, h, pswfcs,
           pswfc_occs, rhoatom, rvlocpzdr, r2projsdr; identifier)
end

charge_ionic(psp::PspUpf) = psp.Zion

"""
    eval_psp_projector_real(psp::PspUpf, i::Number, l::Number, r::Number)

Evaluate the ith Kleinman-Bylander β projector with angular momentum l at real-space
distance r via linear interpolation on the real-space mesh.

Note: UPFs store `r[i] p_{il}(r[i])`, so r=0 is undefined and will error.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    psp.projs_interp[l+1][i](r) / r
end

"""
    eval_psp_projector_fourier(psp::PspUPF, i::Number, l::Number, q::Number)


Evaluate the radial part of the i-th projector for angular momentum l at the
reciprocal vector with modulus q:
p(q) = ∫R^3 p{il}(r) e^{-iqr} dr
     = 4π ∫{R+} r^2 p_{il}(r) j_l(q r) dr
     = 4π Σ{i} r[i]^2 p_{il}(r[i]) j_l(q r[i]) dr[i]

Note: UPFs store `r[i] p_{il}(r[i])`
"""
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T)::T where {T <: Real}
    s = zero(T)
    @inbounds for ir = eachindex(psp.r2projsdr[l+1][i])
        s += sphericalbesselj_fast(l, q * psp.rgrid[ir]) * psp.r2projsdr[l+1][i][ir]
    end
    4T(π) * s
end

"""
     eval_psp_local_real(psp::PspUpf, r::Number)

Evaluate the local potential at real-space distance `r` via linear
interpolation on the real-space mesh.
 """
eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real} = psp.vloc_interp(r)

"""
    eval_psp_local_fourier(psp::PspUpf, q::Number)

Evaluate the local part of the pseudopotential in reciprocal space
using a Coulomb correction term -Z/r:
V(q) = ∫{R^3} (Vloc(r) + Z/r) e^{-iqr} dr
     = 4π ∫{R+} (Vloc(r) + Z/r) sin(qr)/qr r^2 dr
     = 4π/q Σ{i} sin(q r[i]) (r[i] V(r[i]) + Z) dr[i]
"""
function eval_psp_local_fourier(psp::PspUpf, q::T)::T where {T <: Real}
    s = zero(T)
    @inbounds for ir = eachindex(psp.rvlocpzdr)
        s += sin(q * psp.rgrid[ir]) * psp.rvlocpzdr[ir]
    end
    4T(π) * (s - psp.Zion / q) / q
end

"""
    eval_psp_energy_correction(T::Type, psp::PspUpf, n_electrons::Number)

Evaluate the energy correction to the Ewald electrostatic interaction energy of one unit
cell, which is required compared the Ewald expression for point-like nuclei.
n_electrons is the number of electrons per unit cell.
This defines the uniform compensating background charge, which is assumed here.

Notice: The returned result is the energy per unit cell and not the energy per volume.
To obtain the latter, the caller needs to divide by the unit cell volume.

The energy correction is defined as the limit of the Fourier-transform of the local
potential as q -> 0:
lim{q->0} 4π Nelec ∫{R+} (V(r) + Z/r) sin(qr)/qr r^2 dr
= 4π Nelec ∫{R+} (V(r) + Z/r) r^2 dr
= 4π Nelec Σ{i} r[i] (r[i] V(r[i]) + Z) dr[i]
"""
function eval_psp_energy_correction(T, psp::PspUpf, n_electrons)
    4T(π) * n_electrons * dot(psp.rgrid, psp.rvlocpzdr)
end