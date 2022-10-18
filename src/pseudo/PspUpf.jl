using LinearAlgebra
using Interpolations: linear_interpolation
using PseudoPotentialIO: load_upf

"""
Read from file:
- `Zion`: Pseudo-atomic (valence) charge. UPF: `z_valence`
- `lmax`: Maximal angular momentum in the non-local part. UPF: `l_max`
- `rgrid`: Radial grid, can be linear or logarithmic. UPF: `PP_MESH/PP_R`
- `drgrid`: Radial grid derivative / integration factors. UPF: `PP_MESH/PP_RAB`
- `vloc`: Local part of the potential on the radial grid. UPF: `PP_LOCAL`
- `projs`: r * β where β are Kleinman-Bylander non-local projectors on the radial grid.
UPF: `PP_NONLOCAL/PP_BETA.i`
- `h`: Kleinman-Bylander energies. Stored per AM channel `h[l+1][i,j]`. UPF: `PP_DIJ`
- `pswfcs`: (UNUSED) Pseudo-wavefunctions on the radial grid. Can be used for wavefunction
initialization and as projectors for PDOS and DFT+U(+V). UPF: `PP_PSWFC/PP_CHI.i`
- `pswfc_occs`: (UNUSED) Occupations of the pseudo-atomic wavefunctions.
UPF: `PP_PSWFC/PP_CHI.i['occupation']`
- `rhoatom`: (UNUSED) Pseudo-atomic (valence) charge density on the radial grid.
Can be used for charge density initialization. UPF: `PP_RHOATOM`

Pre-computed for performance:
- `vloc_interp`: (USED IN TESTS) Local potential interpolator, stored for performance.
- `projs_interp`: (USED IN TESTS) Projector interpolators, stored for performance.
- `r_vloc_corr_dr`: r_i V_{corr}(r_i) dr_i where V_{corr} = (V_{local}(r_i) + Z / r_i)
- `r2_projs_dr`: r_i^2 β_{ln}(r_i) dr_i

Extras:
- `identifier`: String identifying the pseudopotential.
"""
struct PspUpf{T,IT} <: NormConservingPsp
    Zion::Int
    lmax::Int
    rgrid::Vector{T}
    drgrid::Vector{T}
    vloc::Vector{T}
    projs::Vector{Vector{Vector{T}}}
    h::Vector{Matrix{T}}
    pswfcs::Vector{Vector{Vector{T}}}
    pswfc_occs::Vector{Vector{T}}
    rhoatom::Vector{T}

    vloc_interp::IT    
    projs_interp::Vector{Vector{IT}}

    r_vloc_corr_dr::Vector{T}
    r2_projs_dr::Vector{Vector{Vector{T}}}

    identifier::String
end

"""
    PspUpf(Zion::Number, lmax::Number, rgrid::Vector, drgrid::Vector, vloc::Vector,
           projs::Vector{Vector{Vector}}, h::Vector{Matrix}, pswfcs::Vector{Vector{Vector}},
           pswfc_occs::Vector{Vector}, rhoatom::Vector, vloc_interp, projs_interp,
           r_vloc_corr_dr::Vector, r2_projs_dr::Vector{Vector{Vector}}; identifier="")

Construct a Unified Pseudopotential Format pseudopotential.
Currently only scalar-relativistic norm-conserving potentials are supported.
Non-linear core corrections are not supported.
"""
function PspUpf(Zion, lmax, rgrid::Vector{T}, drgrid, vloc, projs, h, pswfcs, pswfc_occs,
                rhoatom, vloc_interp::IT, projs_interp, r_vloc_corr_dr, r2_projs_dr;
                identifier="") where {T, IT}
    PspUpf{T,IT}(Zion, lmax, rgrid, drgrid, vloc, projs, h, pswfcs, pswfc_occs, rhoatom,
                 vloc_interp, projs_interp, r_vloc_corr_dr, r2_projs_dr, identifier)
end

function parse_upf_file(path; identifier=path)
    pseudo = load_upf(path)

    unsupported = []
    pseudo["header"]["core_correction"]      && push!(unsupported, "non-lin. core correction")
    pseudo["header"]["has_so"]               && push!(unsupported, "spin-orbit coupling")
    pseudo["header"]["pseudo_type"] == "SL"  && push!(unsupported, "semilocal potential")
    pseudo["header"]["pseudo_type"] == "US"  && push!(unsupported, "ultrasoft")
    pseudo["header"]["pseudo_type"] == "PAW" && push!(unsupported, "projector-augmented wave")
    pseudo["header"]["has_gipaw"]            && push!(unsupported, "gipaw data")
    pseudo["header"]["pseudo_type"] == "1/r" && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported" *
                                     " features/quantities: $(join(unsupported, ","))")
    ###
    ### Quantities from file
    ###
    Zion   = Int(pseudo["header"]["z_valence"])
    rgrid  = pseudo["radial_grid"]
    drgrid = pseudo["radial_grid_derivative"]
    lmax   = pseudo["header"]["l_max"]
    vloc   = pseudo["local_potential"] ./ 2  # (Ry -> Ha)

    # There are two possible units schemes for the projectors and coupling coefficients:
    # β [Ry Bohr^{-1/2}]  h [Ry^{-1}]
    # β [Bohr^{-1/2}]     h [Ry]
    # The quantity that's used in calculations is β h β, so the units don't practically
    # matter. However, HGH pseudos in UPF format use the first units, so we assume them
    # to facilitate comparison of the intermediate quantities with analytical HGH.

    projs = map(0:lmax) do l
        betas_l = filter(beta -> beta["angular_momentum"] == l, pseudo["beta_projectors"])
        map(beta -> beta["radial_function"] ./ 2, betas_l)  # Ry -> Ha
    end
    
    h = Matrix[]
    count = 1
    for l = 0:lmax
        nproj_l = length(projs[l+1])
        Dij_l = pseudo["D_ion"][count:count+nproj_l-1, count:count+nproj_l-1] .* 2  # 1/Ry -> 1/Ha
        push!(h, Dij_l)
        count += nproj_l
    end

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
    
    rhoatom = pseudo["total_charge_density"]

    ###
    ### Calculated quantities
    ###
    vloc_interp = linear_interpolation((rgrid, ), vloc)
    
    projs_interp = map(projs) do projs_l
        @views map(projs_l) do proj
            ir_cut = lastindex(proj)
            linear_interpolation((rgrid[1:ir_cut], ), proj)
        end
    end

    r_vloc_corr_dr = (rgrid .* vloc .+ Zion) .* drgrid
    
    r2_projs_dr = map(projs) do projs_l
        @views map(projs_l) do proj
            ir_cut = lastindex(proj)
            rgrid[1:ir_cut] .* proj .* drgrid[1:ir_cut]
        end
    end

    PspUpf(Zion, lmax, rgrid, drgrid, vloc, projs, h, pswfcs, pswfc_occs, rhoatom,
           vloc_interp, projs_interp, r_vloc_corr_dr, r2_projs_dr; identifier)
end

charge_ionic(psp::PspUpf) = psp.Zion

"""
    eval_psp_projector_real(psp::PspUpf, i::Number, l::Number, r<:Real)

Evaluate the ith Kleinman-Bylander β projector with angular momentum l at real-space
distance r via linear interpolation on the real-space mesh.

Note: UPFs store `r[j] β_{li}(r[j])`, so r=0 is undefined and will error.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    psp.projs_interp[l+1][i](r) / r
end

"""
    eval_psp_projector_fourier(psp::PspUPF, i::Number, l::Number, q<:Real)

For UPFs, the integral is transformed to the following sum:
4π Σ{k} j_l(q r[k]) (r[k]^2 p_{il}(r[k]) dr[k])
"""
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T) where {T <: Real}
    s = zero(T)
    @inbounds for ir = eachindex(psp.r2_projs_dr[l+1][i])
        s += sphericalbesselj_fast(l, q * psp.rgrid[ir]) * psp.r2_projs_dr[l+1][i][ir]
    end
    4T(π) * s
end

"""
     eval_psp_local_real(psp::PspUpf, r<:Real)

Evaluate the local potential at real-space distance `r` via linear
interpolation on the real-space mesh.
 """
eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real} = psp.vloc_interp(r)

"""
    eval_psp_local_fourier(psp::PspUpf, q<:Real)

for UPFs, the integral is transformed to the following sum:
4π/q (Σ{i} sin(q r[i]) (r[i] V(r[i]) + Z) dr[i] - Z/q)
"""
function eval_psp_local_fourier(psp::PspUpf, q::T)::T where {T <: Real}
    s = zero(T)
    @inbounds for ir = eachindex(psp.r_vloc_corr_dr)
        s += sin(q * psp.rgrid[ir]) * psp.r_vloc_corr_dr[ir]
    end
    4T(π) * (s - psp.Zion / q) / q
end

"""
    eval_psp_energy_correction(T::Type, psp::PspUpf, n_electrons::Number)

For UPFs, the integral is transformed to the following sum:
4π Nelec Σ{i} r[i] (r[i] V(r[i]) + Z) dr[i]
"""
function eval_psp_energy_correction(T, psp::PspUpf, n_electrons)
    4T(π) * n_electrons * dot(psp.rgrid, psp.r_vloc_corr_dr)
end
