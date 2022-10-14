using LinearAlgebra
using Interpolations: linear_interpolation
using PseudoPotentialIO: load_upf

struct PspUpf{T} <: NormConservingPsp
    Zion::Int                             # Ionic charge (Z - valence electrons)
    lmax::Int                             # Maximal angular momentum in the non-local part
    
    rgrid::Vector{T}                      # Radial grid
    drgrid::Vector{T}                     # Radial grid derivative / integration factor
    
    vloc::Vector{T}                       # Local potential on the radial grid
    vloc_interp                           # Local potential interpolator
    
    projs::Vector{Vector{Vector{T}}}      # Kleinman-Bylander β projectors: projs[l+1][i]
    projs_interp                          # Projector interpolator
    h::Vector{Matrix{T}}                  # Projector coupling coefficients per AM channel: h[l+1][i1,i2]
    
    pswfcs::Vector{Vector{T}}             # Pseudo-atomic wavefunctions
    pswfc_ang_moms::Vector{Int}           # Angular momenta of the pseudo-atomic wavefunctions
    pswfc_occs::Vector{T}                 # Occupations of the pseudo-atomic wavefunctions

    rhoatom::Vector{T}                    # Pseudo-atomic charge density

    rvlocpzdr::Vector{T}                  # (r_i V_{loc}(r_i) + Z) dr_i
    r2projsdr::Vector{Vector{Vector{T}}}  # r_j^2 β_{li}(r_j) dr_i
    
    identifier::String                    # String identifying the PSP
end

"""
    PspUpf(Zion::Number, lmax::Number, rgrid::Vector, drgrid::Vector, vloc::Vector,
           vloc_interp, projs::Vector{Vector{Vector}}, projs_interp, h::Vector{Matrix},
           pswfcs::Vector{Vector}, pswfc_ang_moms::Vector, pswfc_occs::Vector, rhoatom::Vector,
           rvlocpzdr::Vector, r2projsdr::Vector{Vector{Vector}}; identifier="")

Construct a Unified Pseudopotential Format pseudopotential. Currently only
norm-conserving potentials are supported.
"""
function PspUpf(Zion, lmax, rgrid::Vector{T}, drgrid, vloc, vloc_interp, projs, projs_interp, h,
                pswfcs, pswfc_ang_moms, pswfc_occs, rhoatom, rvlocpzdr, r2projsdr;
                identifier="") where {T}
    length(projs) == length(h) || error("Length of projs and h do not agree.")
    length(r2projsdr) == length(h) || error("Length of rgrid * projs * dr and h do not agree.")
    length(pswfcs) == length(pswfc_ang_moms) || error("Length of pseudo wfcs and pseudo wfc" *
                                                      "angular momenta do not agree.")
    length(pswfcs) == length(pswfc_occs) || error("Length of pseudo wfcs and pseudo wfc" *
                                                      "occupations do not agree.")

    return PspUpf{T}(Zion, lmax, rgrid, drgrid, vloc, vloc_interp, projs, projs_interp, h,
                     pswfcs, pswfc_ang_moms, pswfc_occs, rhoatom, rvlocpzdr, r2projsdr, identifier)
end

function parse_upf_file(path; identifier=path)
    pseudo = load_upf(path)

    unsupported = []
    pseudo["header"]["core_correction"] && push!(unsupported, "nonlinear core correction")
    pseudo["header"]["has_so"] && push!(unsupported, "spin-orbit coupling")
    pseudo["header"]["pseudo_type"] == "SL" && push!(unsupported, "semilocal potential")
    pseudo["header"]["pseudo_type"] == "US" && push!(unsupported, "ultrasoft")
    pseudo["header"]["pseudo_type"] == "PAW" && push!(unsupported, "plane-augmented wave")
    pseudo["header"]["has_gipaw"] && push!(unsupported, "gipaw reconstruction data")
    pseudo["header"]["pseudo_type"] == "1/r" && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported "
                                        * "features/quantities: $unsupported")

    Zion = Int(pseudo["header"]["z_valence"])           # Pseudo-ion charge
    rgrid = pseudo["radial_grid"]                       # Radial grid
    drgrid = pseudo["radial_grid_derivative"]           # Grid derivative / integration coeff.s
    lmax = pseudo["header"]["l_max"]                    # Maximum angular momentum channel
    nproj = pseudo["header"]["number_of_proj"]          # Number of Kleinman-Bylander β projectors
    nwfc = pseudo["header"]["number_of_wfc"]            # Number of pseudo-atomic wavefunctions
    vloc = pseudo["local_potential"] ./ 2               # Local potential (Ry -> Ha)
    vloc_interp = linear_interpolation((rgrid,), vloc)  # Interpolator for the local potential

    # Kleinman-Bylander β projectors
    # From the UPF, we get a flat vector of the projectors, which need to be put into a nested
    # vector projs[l+1][i][ir] where l is the angular momentum, i is the index over projectors
    # for one angular momentum channel, and ir is the grid index.
    # We also initialize linear interpolators for each projector, used in `eval_psp_projector_real`.
    # NB: UPFs store rβ, not just β
    projs = [Vector[] for _ = 0:lmax]   # Projectors on the grid
    projs_interp = [[] for _ = 0:lmax]  # Interpolators for the non-local projectors
    for i = 1:nproj
        proj_data = pseudo["beta_projectors"][i]
        l = proj_data["angular_momentum"]
        proj = proj_data["radial_function"] ./ 2  # Ry -> Ha
        ir_start = iszero(rgrid[1]) ? 2 : 1  # Some grids start at 0., so (rβ)[1]/r[1] is undefined
        ir_cut = length(proj)  # Projectors are cut off by the UPF parser @ ir_cut given by the UPF
        proj_interp = linear_interpolation((rgrid[ir_start:ir_cut],), proj[ir_start:ir_cut])
        push!(projs[l+1], proj)
        push!(projs_interp[l+1], proj_interp)
    end

    # β-projector coupling coefficients
    # From the UPF, we get a matrix `Dij` of the coupling coefficients between each projector
    # where i,j are indices corresponding to the order in which the projectors are listed in the
    # UPF. Similar to how we lay out the projectors above, we make a vector of matrices, one for
    # each angular momentum channel, containing the corresponding block from the full `Dij` matrix.
    h = Matrix[]
    n = 1
    for l = 0:lmax
        nproj_l = length(projs[l+1])
        Dij_l = pseudo["D_ion"][n:n+nproj_l-1, n:n+nproj_l-1] .* 2  # 1/Ry -> 1/Ha
        push!(h, Dij_l)
        n += nproj_l
    end

    # Pseudo-atomic wavefunctions
    # The pseudo-atomic wavefunctions (χ) are not currently used in DFTK, but they are can be
    # used for initializing the starting wavefunction and as projectors for projected densities
    # of states, projected wavefunctions, and DFT+U(+V)
    pswfcs = Vector{Vector}(undef, nwfc)
    pswfc_ang_moms = Vector(undef, nwfc)
    pswfc_occs = Vector(undef, nwfc)
    for i = 1:nwfc
        pswfc_data = pseudo["atomic_wave_functions"][i]
        pswfcs[i] = pswfc_data["radial_function"]
        pswfc_ang_moms[i] = pswfc_data["angular_momentum"]
        pswfc_occs[i] = pswfc_data["occupation"]
    end

    # Pseudo-atomic charge density
    # The pseudo-atomic charge density (ρ_{atom}) is not currenctly used in DFTK, but it can be
    # used for initializing the starting guess density.
    rhoatom = pseudo["total_charge_density"]

    # Useful precomputed quantities
    rvlocpzdr = (rgrid .* vloc .+ Zion) .* drgrid  # q-independent part of the integrand for the Fourier-transform of Vloc
    r2projsdr = [Vector[] for _ = 0:lmax]  # q-independent part of the integrand for the Fourier-transform of the projectors
    for l = 0:lmax
        for proj in projs[l+1]
            push!(r2projsdr[l+1],
                  proj .* view(rgrid, 1:length(proj)) .* view(drgrid, 1:length(proj)))
        end
    end

    PspUpf(Zion, lmax, rgrid, drgrid, vloc, vloc_interp, projs, projs_interp, h,
           pswfcs, pswfc_ang_moms, pswfc_occs, rhoatom, rvlocpzdr, r2projsdr; identifier)
end

charge_ionic(psp::PspUpf) = psp.Zion
nprojs(psp::PspUpf) = sum(l -> nprojs(psp, l), 0:psp.lmax)
nprojs(psp::PspUpf, l) = length(psp.projs[l+1])

"""
    eval_psp_projector_real(psp::PspUpf, i::Number, l::Number, r::Number)

Evaluate the ith Kleinman-Bylander β projector with angular momentum l at real-space distance r
via linear interpolation on the real-space mesh.

Note: UPFs store `r[i] p_{il}(r[i])`
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    # NB: the definition of the interpolator in `parse_upf_file` will restrict the domain of the
    # interpolation to exclude r=0 so that this division is well-defined.
    psp.projs_interp[l+1][i](r) / r
end

"""
    eval_psp_projector_fourier(psp::PspUPF, i::Number, l::Number, q::Number)


Evaluate the radial part of the i-th projector for angular momentum l at the reciprocal vector
with modulus q:
p(q) 
= ∫R^3 p{il}(r) e^{-iqr} dr
= 4π ∫{R+} r^2 p_{il}(r) j_l(q r) dr
= 4π Σ{i} r[i]^2 p_{il}(r[i]) j_l(q r[i]) dr[i]

Note: UPFs store `r[i] p_{il}(r[i])`
"""
function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T)::T where {T <: Real}
    eval_psp_projector_fourier(psp, i, Val(l), q)
end

function eval_psp_projector_fourier(psp::PspUpf, i, val_l::Val{l}, q::T)::T where {T <: Real, l}
    if iszero(q)
        # When q=0: 4π Σ[ j_{l}(0) * r[ir]^2 * β_{li}[ir] * dr[ir] ]
        #           where j_{l=0}(0) = 1 -> 4π Σ[ r[ir]^2 * β_{li}[ir] * dr[ir] ]
        #                 j_{l>0}(0) = 0 -> 0
        return iszero(l) ? 4T(π) * sum(psp.r2projsdr[l+1][i]) : zero(T)
    end

    if iszero(psp.rgrid[begin])
        # When the r-grid starts at 0, treat the first index specially to avoid
        # divide-by-zero NaNs from sphericalbesselj.
        s = iszero(l) ? one(T) * psp.r2projsdr[l+1][i][begin] : zero(T)
        ir_start = firstindex(psp.rgrid) + 1
    else
        s = zero(T)
        ir_start = firstindex(psp.rgrid)
    end
    
    ir_cut = lastindex(psp.r2projsdr[l+1][i])
    r2projdr = psp.r2projsdr[l+1][i]
    @inbounds for ir = ir_start:ir_cut
        s += sphericalbesselj(val_l, q * psp.rgrid[ir]) * r2projdr[ir]
    end
    
    return 4T(π) * s
end

"""
     eval_psp_local_real(psp::PspUpf, r::Number)

 Evaluate the local potential at real-space distance `r` via linear interpolation on the
 real-space mesh.
 """
function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    psp.vloc_interp(r)
end

"""
    eval_psp_local_fourier(psp::PspUpf, q::Number)

Evaluate the local part of the pseudopotential in reciprocal space using a Coulomb correction
term -Z/r:
V(q)
= ∫{R^3} (Vloc(r) + Z/r) e^{-iqr} dr
= 4π ∫{R+} (Vloc(r) + Z/r) sin(qr)/qr r^2 dr
= 4π/q Σ{i} sin(q r[i]) (r[i] V(r[i]) + Z) dr[i]
"""
function eval_psp_local_fourier(psp::PspUpf, q::T)::T where {T <: Real}
    s = zero(T)
    @inbounds @fastmath for ir = eachindex(psp.rvlocpzdr)
        s += sin(q * psp.rgrid[ir]) * psp.rvlocpzdr[ir]
    end
    4T(π) * (s - psp.Zion/q) / q
end

"""
    eval_psp_energy_correction(T::Type, psp::PspUpf, n_electrons::Number)

Evaluate the energy correction to the Ewald electrostatic interaction energy of one unit cell,
which is required compared the Ewald expression for point-like nuclei.
n_electrons is the number of electrons per unit cell.
This defines the uniform compensating background charge, which is assumed here.

Notice: The returned result is the energy per unit cell and not the energy per volume.
To obtain the latter, the caller needs to divide by the unit cell volume.

The energy correction is defined as the limit of the Fourier-transform of the local potential as
q -> 0:
lim{q->0} 4π Nelec ∫{R+} (V(r) + Z/r) sin(qr)/qr r^2 dr
= 4π Nelec ∫{R+} (V(r) + Z/r) r^2 dr
= 4π Nelec Σ{i} r[i] (r[i] V(r[i]) + Z) dr[i]
"""
function eval_psp_energy_correction(T, psp::PspUpf, n_electrons)
    4T(π) * n_electrons * dot(psp.rgrid, psp.rvlocpzdr)
end