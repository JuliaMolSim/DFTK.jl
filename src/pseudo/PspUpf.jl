using LinearAlgebra
using SpecialFunctions: erf
import PseudoPotentialIO: load_psp_file, UpfFile, Psp8File

struct PspUpf{T} <: NormConservingPsp
    ## From file, kept for the real-space local potential and the energy correction
    Zion::Int          # Pseudo-atomic (valence) charge. UPF: `z_valence`
    lmax::Int          # Maximal angular momentum in the non-local part. UPF: `l_max`
    rgrid::Vector{T}   # Radial grid, can be linear or logarithmic. UPF: `PP_MESH/PP_R`
    drgrid::Vector{T}  # Radial grid derivative / integration factors. UPF: `PP_MESH/PP_RAB`
    vloc::Vector{T}    # Local part of the potential on the radial grid. UPF: `PP_LOCAL`
    # Kleinman-Bylander energies. Stored per AM channel `h[l+1][i,j]`. UPF: `PP_DIJ`
    h::Vector{Matrix{T}}
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

    ## Radial quantities: raw data (real-space eval) + tabulated modified Hankel transform
    ## (Fourier-space eval), see radial_part.jl.
    # Local potential. The `RadialPart` tabulates the transform of the erf tail-corrected part
    # r²V + Z r erf(r) (l = 0); `eval_psp_local_fourier` adds the analytic -Z/p² tail back.
    # UPF: `PP_LOCAL`.
    vloc_fourier::RadialPart{T}
    # r² β where β are Kleinman-Bylander non-local projectors. `projs[l+1][i]`.
    # UPF: `PP_NONLOCAL/PP_BETA.i`.
    projs::Vector{Vector{RadialPart{T}}}
    # r² χ where χ are pseudo-atomic wavefunctions, used as projectors for PDOS and DFT+U(+V).
    # `pswfcs[l+1][i]`. UPF: `PP_PSWFC/PP_CHI.i`.
    pswfcs::Vector{Vector{RadialPart{T}}}
    # r² ρion, the pseudo-atomic (valence) charge density. UPF: `PP_RHOATOM`.
    ρion::RadialPart{T}
    # r² ρcore, the atomic core charge density for the non-linear core correction (or all-zero
    # if absent). UPF: `PP_NLCC`.
    ρcore::RadialPart{T}
    # r² τcore, likewise for the core kinetic energy density. UPF: `PP_TAUMOD`.
    τcore::RadialPart{T}

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

    # Tabulate the modified Hankel transforms. Projectors and pseudo-wavefunctions live on the
    # orbital cutoff sphere; the local potential and densities on the denser density grid,
    # hence the two `pmax`. All are integrated only up to the radial cutoff `ircut` (except the
    # projectors, already cut in the file, and the pseudo-wavefunctions, kept on the full mesh).
    #
    # The local potential's transform is that of its erf tail-corrected part r²V + Z r erf(r);
    # see `eval_psp_local_fourier`, which adds the -Z/p² Coulomb tail back analytically.
    r_cut = rgrid[1:ircut]
    r2_vloc_corr = r_cut .^ 2 .* vloc[1:ircut] .+ Zion .* r_cut .* erf.(r_cut)

    # Collect every radial quantity as an (l, rgrid, r²f, pmax) build job -- local potential
    # and densities first, then projectors, then pseudo-wavefunctions -- and tabulate them in
    # parallel: the quantities are independent and the transform dominates load_psp.
    jobs = [(0, r_cut, r2_vloc_corr,      RADIAL_TABLE_PMAX_LOCAL),
            (0, r_cut, r2_ρion[1:ircut],  RADIAL_TABLE_PMAX_LOCAL),
            (0, r_cut, r2_ρcore[1:ircut], RADIAL_TABLE_PMAX_LOCAL),
            (0, r_cut, r2_τcore[1:ircut], RADIAL_TABLE_PMAX_LOCAL)]
    for l = 0:lmax, r2_proj in r2_projs[l+1]
        ncut = min(ircut, length(r2_proj))
        push!(jobs, (l, rgrid[1:ncut], r2_proj[1:ncut], RADIAL_TABLE_PMAX_PROJ))
    end
    for l = 0:lmax, r2_pswfc in r2_pswfcs[l+1]
        push!(jobs, (l, rgrid, r2_pswfc, RADIAL_TABLE_PMAX_PROJ))
    end
    # The few wide l = 0 quantities dwarf the rest; `parallel_loop_over_range` strides across
    # workers, so those heavy jobs spread out rather than piling onto one thread.
    tables = Vector{RadialPart{eltype(rgrid)}}(undef, length(jobs))
    parallel_loop_over_range(eachindex(jobs)) do i
        tables[i] = RadialPart(jobs[i][1], jobs[i][2], jobs[i][3]; pmax=jobs[i][4])
    end

    # Place the results back into their structure, in the order the jobs were pushed.
    vloc_fourier, ρion, ρcore, τcore = tables[1], tables[2], tables[3], tables[4]
    projs  = Vector{Vector{RadialPart{eltype(rgrid)}}}(undef, lmax + 1)
    pswfcs = Vector{Vector{RadialPart{eltype(rgrid)}}}(undef, lmax + 1)
    i = 4
    for l = 0:lmax
        n = length(r2_projs[l+1]);   projs[l+1]  = tables[i+1:i+n];  i += n
    end
    for l = 0:lmax
        n = length(r2_pswfcs[l+1]);  pswfcs[l+1] = tables[i+1:i+n];  i += n
    end

    PspUpf{eltype(rgrid)}(
        Zion, lmax, rgrid, drgrid, vloc, h, pswfc_occs, pswfc_energies, pswfc_labels,
        vloc_fourier, projs, pswfcs, ρion, ρcore, τcore,
        rcut, ircut, identifier, description
    )
end

charge_ionic(psp::PspUpf) = psp.Zion
has_valence_density(psp::PspUpf) = !all(iszero, psp.ρion.r2_f)
has_core_density(psp::PspUpf) = !all(iszero, psp.ρcore.r2_f)
has_core_kinetic_energy_density(psp::PspUpf) = !all(iszero, psp.τcore.r2_f)

eval_psp_projector_real(psp::PspUpf, i, l, r::Real) = eval_real(psp.projs[l+1][i], r)
eval_psp_projector_fourier(psp::PspUpf, i, l, p::Real) = eval_fourier(psp.projs[l+1][i], p)
eval_psp_projector_fourier(psp::PspUpf, i, l, ps::AbstractVector{<:Real}) =
    eval_fourier(psp.projs[l+1][i], ps)

count_n_pswfc_radial(psp::PspUpf, l) = length(psp.pswfcs[l+1])

pswfc_label(psp::PspUpf, i, l) = psp.pswfc_labels[l+1][i]

eval_psp_pswfc_real(psp::PspUpf, i, l, r::Real) = eval_real(psp.pswfcs[l+1][i], r)
eval_psp_pswfc_fourier(psp::PspUpf, i, l, p::Real) = eval_fourier(psp.pswfcs[l+1][i], p)
eval_psp_pswfc_fourier(psp::PspUpf, i, l, ps::AbstractVector{<:Real}) =
    eval_fourier(psp.pswfcs[l+1][i], ps)

# The local potential in real space, by linear interpolation of the raw mesh (clamped: unlike
# the compactly-supported quantities, vloc has a Coulomb tail and is only queried in range).
function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    (; rgrid, vloc) = psp
    r ≤ rgrid[1]   && return T(vloc[1])
    r ≥ rgrid[end] && return T(vloc[end])
    j = searchsortedlast(rgrid, r)
    t = (r - rgrid[j]) / (rgrid[j+1] - rgrid[j])
    (1 - t) * vloc[j] + t * vloc[j+1]
end

# `vloc_fourier` tabulates the transform of the erf tail-corrected local potential; the
# long-range -Z/p² exp(-p²/4) Coulomb tail (H of the QE-style C(r) = -Z erf(r)/r) is added
# back here. See `eval_psp_local_fourier` in NormConservingPsp.jl for the definitions.
function _add_local_coulomb_tail(corrected::T, Zion, p) where {T}
    iszero(p) && return zero(T)  # G = 0: the divergence is a compensating charge background
    corrected + 4T(π) * (-Zion / p^2 * exp(-p^2 / T(4)))
end
eval_psp_local_fourier(psp::PspUpf, p::Real) =
    _add_local_coulomb_tail(eval_fourier(psp.vloc_fourier, p), psp.Zion, p)
function eval_psp_local_fourier(psp::PspUpf, ps::AbstractVector{<:Real})
    corrected = eval_fourier(psp.vloc_fourier, ps)
    Zion = psp.Zion
    map((c, p) -> _add_local_coulomb_tail(c, Zion, p), corrected, ps)
end

eval_psp_valence_density_real(psp::PspUpf, r::Real) = eval_real(psp.ρion, r)
eval_psp_valence_density_fourier(psp::PspUpf, p::Real) = eval_fourier(psp.ρion, p)
eval_psp_valence_density_fourier(psp::PspUpf, ps::AbstractVector{<:Real}) =
    eval_fourier(psp.ρion, ps)

eval_psp_core_density_real(psp::PspUpf, r::Real) = eval_real(psp.ρcore, r)
eval_psp_core_density_fourier(psp::PspUpf, p::Real) = eval_fourier(psp.ρcore, p)
eval_psp_core_density_fourier(psp::PspUpf, ps::AbstractVector{<:Real}) =
    eval_fourier(psp.ρcore, ps)

eval_psp_core_kinetic_energy_density_real(psp::PspUpf, r::Real) = eval_real(psp.τcore, r)
eval_psp_core_kinetic_energy_density_fourier(psp::PspUpf, p::Real) = eval_fourier(psp.τcore, p)
eval_psp_core_kinetic_energy_density_fourier(psp::PspUpf, ps::AbstractVector{<:Real}) =
    eval_fourier(psp.τcore, ps)

function eval_psp_energy_correction(T, psp::PspUpf)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc  = @view psp.vloc[1:psp.ircut]
    res = 4T(π) * simpson(rgrid) do i, r
        r * (r * vloc[i] - -psp.Zion)
    end
    convert(T, res)
end
