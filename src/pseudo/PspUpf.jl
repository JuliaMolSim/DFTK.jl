using LinearAlgebra
using Interpolations: linear_interpolation
using PseudoPotentialIO: load_upf
using NumericalIntegration

struct PspUpf{T,I} <: NormConservingPsp
    ## From file
    Zion::Int          # Pseudo-atomic (valence) charge. UPF: `z_valence`
    lmax::Int          # Maximal angular momentum in the non-local part. UPF: `l_max`
    rgrid::Vector{T}   # Radial grid, can be linear or logarithmic. UPF: `PP_MESH/PP_R`
    drgrid::Vector{T}  # Radial grid derivative / integration factors. UPF: `PP_MESH/PP_RAB`
    vloc::Vector{T}    # Local part of the potential on the radial grid. UPF: `PP_LOCAL`
    # r * β where β are Kleinman-Bylander non-local projectors on the radial grid.
    # UPF: `PP_NONLOCAL/PP_BETA.i`
    r_projs::Vector{Vector{Vector{T}}}
    # Kleinman-Bylander energies. Stored per AM channel `h[l+1][i,j]`. UPF: `PP_DIJ`
    h::Vector{Matrix{T}}
    # (UNUSED) Pseudo-wavefunctions on the radial grid. Can be used for wavefunction
    # initialization and as projectors for PDOS and DFT+U(+V). UPF: `PP_PSWFC/PP_CHI.i`
    pswfcs::Vector{Vector{Vector{T}}}
    # (UNUSED) Occupations of the pseudo-atomic wavefunctions.
    # UPF: `PP_PSWFC/PP_CHI.i['occupation']`
    pswfc_occs::Vector{Vector{T}}
    # Pseudo-atomic (valence) charge density on the radial grid multiplied by
    # 4πr^2. Can be used for charge density initialization. UPF: `PP_RHOATOM`
    r2_4π_ρion::Vector{T}
    # Atomic core charge density on the radial grid, used for non-linear core correction.
    # Unlike the pseudo-atomic valence charge density, this is a true charge density with
    # no prefactor. UPF: `PP_NLCC`
    ρcore::Vector{T}

    ## Precomputed for performance
    # (USED IN TESTS) Local potential interpolator, stored for performance.
    vloc_interp::I
    # (USED IN TESTS) Projector interpolators, stored for performance.
    r_projs_interp::Vector{Vector{I}}
    # (USED IN TESTS) Valence charge density interpolator, stored for performance.
    r2_4π_ρion_interp::I
    # (USED IN TESTS) Core charge density interpolator, stored for performance.
    ρcore_interp::I

    ## Extras
    identifier::String   # String identifying the pseudopotential.
    description::String  # Descriptive string. UPF: `comment`
end

"""
    PspUpf(path[, identifier])

Construct a Unified Pseudopotential Format pseudopotential from file.

Does not support:
- Non-linear core correction
- Fully-realtivistic / spin-orbit pseudos
- Bare Coulomb / all-electron potentials
- Semilocal potentials
- Ultrasoft potentials
- Projector-augmented wave potentials
- GIPAW reconstruction data
"""
function PspUpf(path; identifier=path, rcut=nothing)
    pseudo = load_upf(path)

    unsupported = []
    pseudo["header"]["has_so"] && push!(unsupported, "spin-orbit coupling")
    pseudo["header"]["pseudo_type"] == "SL" && push!(unsupported, "semilocal potential")
    pseudo["header"]["pseudo_type"] == "US" && push!(unsupported, "ultrasoft")
    pseudo["header"]["pseudo_type"] == "PAW" && push!(unsupported, "projector-augmented wave")
    pseudo["header"]["has_gipaw"] && push!(unsupported, "gipaw data")
    pseudo["header"]["pseudo_type"] == "1/r" && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported" *
                                     " features/quantities: $(join(unsupported, ","))")

    Zion = Int(pseudo["header"]["z_valence"])
    rgrid = pseudo["radial_grid"]
    drgrid = pseudo["radial_grid_derivative"]
    lmax = pseudo["header"]["l_max"]
    vloc = pseudo["local_potential"] ./ 2  # (Ry -> Ha)
    description = get(pseudo["header"], "comment", "")

    # There are two possible units schemes for the projectors and coupling coefficients:
    # β [Ry Bohr^{-1/2}]  h [Ry^{-1}]
    # β [Bohr^{-1/2}]     h [Ry]
    # The quantity that's used in calculations is β h β, so the units don't practically
    # matter. However, HGH pseudos in UPF format use the first units, so we assume them
    # to facilitate comparison of the intermediate quantities with analytical HGH.

    r_projs = map(0:lmax) do l
        betas_l = filter(beta -> beta["angular_momentum"] == l, pseudo["beta_projectors"])
        map(beta -> beta["radial_function"] ./ 2, betas_l)  # Ry -> Ha
    end

    h = Matrix[]
    count = 1
    for l = 0:lmax
        nproj_l = length(r_projs[l+1])
        Dij_l = pseudo["D_ion"][count:count+nproj_l-1, count:count+nproj_l-1] .* 2  # 1/Ry -> 1/Ha
        push!(h, Dij_l)
        count += nproj_l
    end

    pswfcs = map(0:lmax-1) do l
        pswfcs_l = filter(pseudo["atomic_wave_functions"]) do pswfc
            pswfc["angular_momentum"] == l
        end
        map(pswfc -> pswfc["radial_function"], pswfcs_l)
    end

    pswfc_occs = map(0:lmax-1) do l
        pswfcs_l = filter(pseudo["atomic_wave_functions"]) do pswfc
            pswfc["angular_momentum"] == l
        end
        map(pswfc -> pswfc["occupation"], pswfcs_l)
    end

    r2_4π_ρion = pseudo["total_charge_density"]

    if pseudo["header"]["core_correction"]
        ρcore = pseudo["core_charge_density"]
    else
        ρcore = zeros(Float64, length(rgrid))
    end

    if !isnothing(rcut)
        mask = rgrid .<= rcut
        rgrid = rgrid[mask]
        drgrid = drgrid[mask]
        vloc = vloc[mask]
        r_projs = map(r_projs) do betas_l
            map(betas_l) do beta
                beta[mask[1:length(beta)]]
            end
        end
        pswfcs = map(pswfcs) do pswfcs_l
            map(pswfcs_l) do pswfc
                pswfc[mask[1:length(pswfc)]]
            end
        end
        r2_4π_ρion = r2_4π_ρion[mask]
        ρcore = ρcore[mask]
    end

    vloc_interp = linear_interpolation((rgrid,), vloc)
    r_projs_interp = map(r_projs) do r_projs_l
        map(r_projs_l) do r_proj  # Can't use views here; have to match `vloc_interp`'s type
            ir_cut = lastindex(r_proj)
            linear_interpolation((rgrid[1:ir_cut],), r_proj)
        end
    end
    r2_4π_ρion_interp = linear_interpolation((rgrid,), r2_4π_ρion)
    ρcore_interp = linear_interpolation((rgrid,), ρcore)

    PspUpf{eltype(rgrid),typeof(vloc_interp)}(Zion, lmax, rgrid, drgrid, vloc,
        r_projs, h, pswfcs, pswfc_occs, r2_4π_ρion, ρcore, vloc_interp,
        r_projs_interp, r2_4π_ρion_interp, ρcore_interp, identifier, description
    )
end

charge_ionic(psp::PspUpf) = psp.Zion
has_valence_density(psp::PspUpf) = !all(iszero, psp.r2_4π_ρion)
has_core_density(psp::PspUpf) = !all(iszero, psp.ρcore)

# Note: UPFs store `r[j] β_{li}(r[j])`, so r=0 is undefined and will error.
function eval_psp_projector_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r_projs_interp[l+1][i](r) / r
end

# For UPFs, the integral is transformed to the following sum:
# 4π Σ{k} j_l(q r[k]) (r[k]^2 p_{il}(r[k]) dr[k])
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T)::T where {T<:Real}
    ir_cut = length(psp.r_projs[l+1][i])
    f = (
        psp.rgrid[1:ir_cut] .* psp.r_projs[l+1][i] .*
        sphericalbesselj_fast.(l, q .* psp.rgrid[1:ir_cut])
    )
    4T(π) * trapezoidal(psp.rgrid[1:ir_cut], f)
end

eval_psp_local_real(psp::PspUpf, r::T) where {T<:Real} = psp.vloc_interp(r)

# For UPFs, the integral is transformed to the following sum:
# 4π/q (Σ{i} sin(q r[i]) (r[i] V(r[i]) + Zerf(r)) dr[i] - Z/q exp(-q^2 / 4))
function eval_psp_local_fourier(psp::PspUpf, q::T)::T where {T<:Real}
    # QE style -Zerf(r)/r correction; ABINIT uses -Z/r
    ir_cut = length(psp.rgrid)
    f = (
        psp.rgrid .* (psp.rgrid .* psp.vloc .- -psp.Zion * erf.(psp.rgrid)) .*
        sphericalbesselj_fast.(0, q .* psp.rgrid)
    )
    I = trapezoidal(psp.rgrid, f)
    4T(π) * (I + -psp.Zion / q^2 * exp(-q^2 / T(4)))
end

function eval_psp_density_valence_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_4π_ρion_interp(r) / (r^2 * 4T(π))
end

# For UPFs, the integral is transformed into the following sum:
# Σ{i} j_0(q r[i]) r^2 4π ρval[i] dr[i]
function eval_psp_density_valence_fourier(psp::PspUpf, q::T) where {T<:Real}
    ir_cut = length(psp.rgrid)
    f = sphericalbesselj_fast.(0, q .* psp.rgrid) .* psp.r2_4π_ρion
    trapezoidal(psp.rgrid, f)
end

function eval_psp_density_core_real(psp::PspUpf, r::T) where {T<:Real}
    psp.ρcore_interp(r)
end

# For UPFs, the integral is transformed into the following sum:
# 4π Σ{i} j_0(q r[i]) r^2 ρcore[i] dr[i]
function eval_psp_density_core_fourier(psp::PspUpf, q::T) where {T<:Real}
    ir_cut = length(psp.rgrid)
    f = psp.rgrid .^ 2 .* sphericalbesselj_fast.(0, q .* psp.rgrid) .* psp.ρcore
    4T(π) * trapezoidal(psp.rgrid, f)
end

# For UPFs, the integral is transformed to the following sum:
# 4π Nelec Σ{i} r[i] (r[i] V(r[i]) + Z) dr[i]
function eval_psp_energy_correction(T, psp::PspUpf, n_electrons)
    ir_cut = length(psp.rgrid)
    f = psp.rgrid .* (psp.rgrid .* psp.vloc .- -psp.Zion)
    4T(π) * n_electrons * trapezoidal(psp.rgrid, f)
end
