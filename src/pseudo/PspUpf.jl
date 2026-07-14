using LinearAlgebra
using Interpolations: linear_interpolation
import PseudoPotentialIO: load_psp_file, UpfFile, Psp8File

struct PspUpf{T,I} <: NormConservingPsp
    ## From file
    Zion::Int          # Pseudo-atomic (valence) charge. UPF: `z_valence`
    lmax::Int          # Maximal angular momentum in the non-local part. UPF: `l_max`
    rgrid::Vector{T}   # Radial grid, can be linear or logarithmic. UPF: `PP_MESH/PP_R`
    drgrid::Vector{T}  # Radial grid derivative / integration factors. UPF: `PP_MESH/PP_RAB`
    vloc::Vector{T}    # Local part of the potential on the radial grid. UPF: `PP_LOCAL`
    # r^2 * β where β are Kleinman-Bylander non-local projectors on the radial grid.
    # UPF: `PP_NONLOCAL/PP_BETA.i`
    r2_projs::Vector{Vector{Vector{T}}}
    # Kleinman-Bylander energies. Stored per AM channel `h[l+1][i,j]`.
    # UPF: `PP_DIJ`
    h::Vector{Matrix{T}}
    # Pseudo-wavefunctions on the radial grid. Used as projectors for PDOS
    # and DFT+U(+V), could be used for wavefunction initialization as well.
    # r^2 * χ where χ are pseudo-atomic wavefunctions on the radial grid.
    # UPF: `PP_PSWFC/PP_CHI.i`
    r2_pswfcs::Vector{Vector{Vector{T}}}
    # (UNUSED) Occupations of the pseudo-atomic wavefunctions.
    # UPF: `PP_PSWFC/PP_CHI.i['occupation']`
    pswfc_occs::Vector{Vector{T}}
    # (UNUSED) Energies of the pseudo-atomic wavefunctions.
    # UPF: `PP_PSWFC/PP_CHI.i['pseudo_energy']`
    pswfc_energies::Vector{Vector{T}}
    # Labels of the pseudo-atomic wavefunctions.
    # Used for projector selection in PDOS and DFT+U(+V).
    # UPF: `PP_PSWFC/PP_CHI.i['label']`
    pswfc_labels::Vector{Vector{String}}
    # 4πr^2 ρion where ρion is the pseudo-atomic (valence) charge density on the
    # radial grid. Can be used for charge density initialization.
    # UPF: `PP_RHOATOM`
    r2_ρion::Vector{T}
    # r^2 ρcore where ρcore is the atomic core charge density on the radial grid,
    # used for non-linear core correction.
    # UPF: `PP_NLCC`
    r2_ρcore::Vector{T}
    # same as `r2_ρcore` but for the kinetic energy density τ
    # UPF: `PP_TAUMOD`
    r2_τcore::Vector{T}

    ## Precomputed for performance
    # (USED IN TESTS) Local potential interpolator, stored for performance.
    vloc_interp::I
    # (USED IN TESTS) Projector interpolators, stored for performance.
    r2_projs_interp::Vector{Vector{I}}
    # (USED IN TESTS) Valence charge density interpolator, stored for performance.
    r2_ρion_interp::I
    # (USED IN TESTS) Core charge density interpolator, stored for performance.
    r2_ρcore_interp::I
    # (USED IN TESTS) Core kinetic energy density interpolator, stored for performance.
    r2_τcore_interp::I

    ## Tabulated modified Hankel transforms, see psp_fourier_table.jl. These make
    ## `eval_psp_*_fourier` an O(1) interpolation rather than a quadrature per p value.
    # `vloc_table` is that of the erf tail-corrected part of vloc (l = 0), see PspUpf below.
    vloc_table::HankelTable{HANKEL_TABLE_ORDER,T,Vector{T}}
    r2_projs_tables::Vector{Vector{HankelTable{HANKEL_TABLE_ORDER,T,Vector{T}}}}
    r2_pswfcs_tables::Vector{Vector{HankelTable{HANKEL_TABLE_ORDER,T,Vector{T}}}}
    r2_ρion_table::HankelTable{HANKEL_TABLE_ORDER,T,Vector{T}}
    r2_ρcore_table::HankelTable{HANKEL_TABLE_ORDER,T,Vector{T}}
    r2_τcore_table::HankelTable{HANKEL_TABLE_ORDER,T,Vector{T}}

    ## Extras
    rcut::T              # Radial cutoff for all quantities except pswfc.
                         # Used to avoid some numerical issues encountered when
                         # integrating over the full radial mesh.
    ircut::Int           # Index of the radial cutoff.
    identifier::String   # String identifying the pseudopotential.
    description::String  # Descriptive string. UPF: `comment`
end

"""
    PspUpf(path[; identifier])

Construct a Unified Pseudopotential Format pseudopotential by reading a file.

Does not support:
- Fully-realtivistic / spin-orbit pseudos
- Bare Coulomb / all-electron potentials
- Semilocal potentials
- Ultrasoft potentials
- Projector-augmented wave potentials
- GIPAW reconstruction data
"""
function PspUpf(path::AbstractString; identifier=path, rcut=nothing)
    PspUpf(load_psp_file(path); identifier=identifier, rcut=rcut)
end

"""
    PspUpf(pseudo::Psp8File; identifier)

Construct a Unified Pseudopotential Format pseudopotential from a parsed psp8 file.
Internally, the pseudo is first converted to a `UpfFile` using `PseudoPotentialIO`.
"""
function PspUpf(pseudo::Psp8File; identifier, rcut=nothing)
    PspUpf(UpfFile(pseudo); identifier=identifier, rcut=rcut)
end

"""
    PspUpf(pseudo::UpfFile; identifier)

Construct a Unified Pseudopotential Format pseudopotential from a parsed upf file.
"""
function PspUpf(pseudo::UpfFile; identifier, rcut=nothing)
    unsupported = []
    pseudo.header.has_so                && push!(unsupported, "spin-orbit coupling")
    pseudo.header.pseudo_type == "SL"   && push!(unsupported, "semilocal potential")
    pseudo.header.pseudo_type == "US"   && push!(unsupported, "ultrasoft")
    pseudo.header.pseudo_type == "USPP" && push!(unsupported, "ultrasoft")
    pseudo.header.pseudo_type == "PAW"  && push!(unsupported, "projector-augmented wave")
    pseudo.header.has_gipaw             && push!(unsupported, "gipaw data")
    pseudo.header.pseudo_type == "1/r"  && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported" *
                                     " features/quantities: $(join(unsupported, ","))")

    Zion        = Int(pseudo.header.z_valence)
    rgrid       = pseudo.mesh.r
    drgrid      = pseudo.mesh.rab
    lmax        = pseudo.header.l_max
    vloc        = pseudo.local_ ./ 2  # (Ry -> Ha)
    description = something(pseudo.header.comment, "")

    # Ensure rcut is at most the end of the rgrid.
    rcut = isnothing(rcut) ? last(rgrid) : min(rcut, last(rgrid))
    ircut = findfirst(>=(rcut), rgrid)

    # There are two possible units schemes for the projectors and coupling coefficients:
    # rβ [Ry Bohr^{-1/2}]  h [Ry^{-1}]
    # rβ [Bohr^{-1/2}]     h [Ry]
    # The quantity that's used in calculations is β h β, so the units don't practically
    # matter. However, GTH pseudos in UPF format use the first units, so we assume them
    # to facilitate comparison of the intermediate quantities with analytical GTH.

    r2_projs = map(0:lmax) do l
        betas_l = filter(beta -> beta.angular_momentum == l, pseudo.nonlocal.betas)
        map(betas_l) do beta_li
            r_beta_ha = beta_li.beta[1:beta_li.cutoff_radius_index] ./ 2  # Ry -> Ha
            rgrid[1:length(r_beta_ha)] .* r_beta_ha  # rβ -> r²β
        end
    end
    h = map(0:lmax) do l
        mask_l = findall(beta -> beta.angular_momentum == l, pseudo.nonlocal.betas)
        pseudo.nonlocal.dij[mask_l, mask_l] .* 2  # 1/Ry -> 1/Ha
    end

    r2_pswfcs = [Vector{Float64}[] for _ = 0:lmax]
    pswfc_occs     = [Float64[]    for _ = 0:lmax]
    pswfc_energies = [Float64[]    for _ = 0:lmax]
    pswfc_labels   = [String[]     for _ = 0:lmax]
    for l = 0:lmax
        pswfcs_l = filter(χ -> χ.l == l, pseudo.pswfc)
        for pswfc_li in pswfcs_l
            push!(r2_pswfcs[l+1], rgrid .* pswfc_li.chi)  # rχ -> r²χ
            push!(pswfc_occs[l+1], pswfc_li.occupation)
            # TODO: energies and labels can be nothing,
            #       we'll see if this is a problem in practice
            push!(pswfc_energies[l+1], pswfc_li.pseudo_energy)
            push!(pswfc_labels[l+1], pswfc_li.label)
        end
    end

    r2_ρion = pseudo.rhoatom ./ (4π)
    r2_ρcore = rgrid .^ 2 .* (@something pseudo.nlcc   zeros(length(rgrid)))
    r2_τcore = rgrid .^ 2 .* (@something pseudo.taumod zeros(length(rgrid)))

    vloc_interp = linear_interpolation((rgrid,), vloc)
    r2_projs_interp = map(r2_projs) do r2_projs_l
        map(proj -> linear_interpolation((rgrid[1:length(proj)],), proj), r2_projs_l)
    end
    r2_ρion_interp = linear_interpolation((rgrid,), r2_ρion)
    r2_ρcore_interp = linear_interpolation((rgrid,), r2_ρcore)
    r2_τcore_interp = linear_interpolation((rgrid,), r2_τcore)

    # Tabulate the Fourier transforms. Everything but the pseudo-atomic wavefunctions is
    # cut off at `rcut`, so those two families need one transform plan each (they coincide
    # in the common case rcut == last(rgrid)). Projectors ending before their family's rcut
    # are zero-padded, which is exact: they are strictly zero past their cutoff radius.
    plan_cut = hankel_table_plan(rgrid[ircut], lmax)
    plan_full = ircut == length(rgrid) ? plan_cut : hankel_table_plan(last(rgrid), lmax)

    # The local potential is transformed with its Coulomb tail taken out (see
    # `eval_psp_local_fourier`), which is also what makes it decay by the end of the mesh.
    rgrid_cut = view(rgrid, 1:ircut)
    r2_vloc_corrected = rgrid .^ 2 .* vloc .+ Zion .* rgrid .* erf.(rgrid)
    vloc_table = build_hankel_table(plan_cut, rgrid_cut,
                                    view(r2_vloc_corrected, 1:ircut), 0)
    r2_projs_tables = map(0:lmax) do l
        map(r2_projs[l+1]) do proj
            ircut_proj = min(ircut, length(proj))
            build_hankel_table(plan_cut, view(rgrid, 1:ircut_proj),
                               view(proj, 1:ircut_proj), l)
        end
    end
    r2_pswfcs_tables = map(0:lmax) do l
        map(pswfc -> build_hankel_table(plan_full, rgrid, pswfc, l), r2_pswfcs[l+1])
    end
    r2_ρion_table  = build_hankel_table(plan_cut, rgrid_cut, view(r2_ρion, 1:ircut), 0)
    r2_ρcore_table = build_hankel_table(plan_cut, rgrid_cut, view(r2_ρcore, 1:ircut), 0)
    r2_τcore_table = build_hankel_table(plan_cut, rgrid_cut, view(r2_τcore, 1:ircut), 0)

    PspUpf{eltype(rgrid),typeof(vloc_interp)}(
        Zion, lmax, rgrid, drgrid,
        vloc, r2_projs, h, r2_pswfcs, pswfc_occs, pswfc_energies, pswfc_labels,
        r2_ρion, r2_ρcore, r2_τcore,
        vloc_interp, r2_projs_interp, r2_ρion_interp, r2_ρcore_interp, r2_τcore_interp,
        vloc_table, r2_projs_tables, r2_pswfcs_tables,
        r2_ρion_table, r2_ρcore_table, r2_τcore_table,
        rcut, ircut, identifier, description
    )
end

charge_ionic(psp::PspUpf) = psp.Zion
max_momentum_fourier(psp::PspUpf) = psp.vloc_table.pmax
has_valence_density(psp::PspUpf) = !all(iszero, psp.r2_ρion)
has_core_density(psp::PspUpf) = !all(iszero, psp.r2_ρcore)
has_core_kinetic_energy_density(psp::PspUpf) = !all(iszero, psp.r2_τcore)

function eval_psp_projector_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r2_projs_interp[l+1][i](r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_projector_fourier(psp::PspUpf, i, l, p::T)::T where {T<:Real}
    psp.r2_projs_tables[l+1][i](p)
end

# Vectorized version of the above, GPU compatible
function eval_psp_projector_fourier(psp::PspUpf, i, l, ps::AbstractVector{T}) where {T<:Real}
    eval_hankel_table(psp.r2_projs_tables[l+1][i], ps)
end

count_n_pswfc_radial(psp::PspUpf, l) = length(psp.r2_pswfcs[l+1])

pswfc_label(psp::PspUpf, i, l) = psp.pswfc_labels[l+1][i]

function eval_psp_pswfc_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r2_pswfcs_interp[l+1][i](r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_pswfc_fourier(psp::PspUpf, i, l, p::T)::T where {T<:Real}
    # Pseudo-atomic wavefunctions are _not_ currently cut off like the other
    # quantities. They are the reason that PseudoDojo UPF files have a much
    # larger radial grid than their psp8 counterparts.
    # If issues arise, try cutting them off too.
    psp.r2_pswfcs_tables[l+1][i](p)
end

eval_psp_local_real(psp::PspUpf, r::T) where {T<:Real} = psp.vloc_interp(r)

# Add back the Hankel transform of the Coulomb tail that was taken out of the tabulated part
# before transforming it (see the table construction in `PspUpf`). The local potential decays
# like -Z/r, too slowly for its Hankel transform to be computed accurately, so we subtract a
# QE-style C(r) = -Z erf(r)/r -- which has the same tail, leaving a short-ranged remainder --
# transform that, and add H[C] back analytically here:
#     H[vloc] = H[vloc - C] + H[C],   H[-Z erf(r)/r] = -4π Z/p² exp(-p²/4).
# (ABINIT instead uses the pure Coulomb C(r) = -Z/r, with H[-Z/r] = -4π Z/p².)
function _add_local_coulomb_tail(Zion, p::T)::T where {T<:Real}
    p == 0 && return zero(T)  # Compensating charge background
    4T(π) * -Zion / p^2 * exp(-p^2 / T(4))
end

function eval_psp_local_fourier(psp::PspUpf, p::T)::T where {T<:Real}
    # The p → 0 limit of the tail-corrected part is finite and nonzero; the zero at p = 0
    # (compensating charge background) is an API contract applied on top of it.
    p == 0 && return zero(T)
    psp.vloc_table(p) + _add_local_coulomb_tail(psp.Zion, p)
end

# Vectorized version of the above, GPU optimized
function eval_psp_local_fourier(psp::PspUpf, ps::AbstractVector{T}) where {T<:Real}
    Zion = psp.Zion
    tabulated = eval_hankel_table(psp.vloc_table, ps)
    map(ps, tabulated) do p, tail_corrected
        p == 0 && return zero(T)  # Compensating charge background
        tail_corrected + _add_local_coulomb_tail(Zion, p)
    end
end

function eval_psp_valence_density_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_ρion_interp(r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_valence_density_fourier(psp::PspUpf, p::T) where {T<:Real}
    psp.r2_ρion_table(p)
end

function eval_psp_core_density_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_ρcore_interp(r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_core_density_fourier(psp::PspUpf, p::T) where {T<:Real}
    psp.r2_ρcore_table(p)
end

# Vectorized version of the above, GPU optimized
function eval_psp_core_density_fourier(psp::PspUpf, ps::AbstractVector{T}) where {T<:Real}
    eval_hankel_table(psp.r2_ρcore_table, ps)
end

function eval_psp_core_kinetic_energy_density_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_τcore_interp(r) / r^2  # TODO if r is below a threshold, return zero
end

function eval_psp_core_kinetic_energy_density_fourier(psp::PspUpf, p::T) where {T<:Real}
    psp.r2_τcore_table(p)
end

function eval_psp_energy_correction(T, psp::PspUpf)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc  = @view psp.vloc[1:psp.ircut]
    res = 4T(π) * simpson(rgrid) do i, r
        r * (r * vloc[i] - -psp.Zion)
    end
    convert(T, res)
end
