# Truncated Coulomb kernels for non-periodic electrostatics, following
#
#     C. A. Rozzi, D. Varsano, A. Marini, E. K. U. Gross, A. Rubio,
#     "Exact Coulomb cutoff technique for supercell calculations",
#     Phys. Rev. B 73, 205119 (2006).
#     https://arxiv.org/abs/cond-mat/0601031
#
# `truncated_coulomb_fourier(G, ...)` returns the Fourier transform v_c(G) of a
# Coulomb interaction that has been truncated in the non-periodic directions of
# the simulation cell, and replaces the usual 4π/|G|² kernel in the Hartree and
# ionic-local electrostatics. It has the same call signature as 4π/|G|² but is
# dispatched according to the number of periodic electrostatic directions in
# `model.periodicity`:
#
#   * 3 — fully periodic: v_c(G) = 4π/|G|² (standard Coulomb with neutralising
#     background, i.e. v_c(0) = 0). In this case the truncated kernel is
#     *identically* the periodic one, and the code below degenerates to the
#     standard formulas — no branching needed at the call site.
#
#   * 2 — 2D slab: one isolated direction, orthogonal to the periodic plane.
#     v_c(G) = 4π/|G|² (1 - exp(-|G_∥| R) cos(G_z R)), with R half the length
#     of the isolated lattice vector.  G=0 set to 0 (compensating background
#     in the periodic plane).
#
#   * 1 — 1D wire: NOT YET IMPLEMENTED. Will require Bessel-function formulas
#     with special handling of the G_∥=0 limit (Rozzi et al., eqs. 21–22).
#
#   * 0 — 0D fully isolated molecule: spherical cutoff at radius R equal to
#     half the minimum lattice edge length. v_c(G) = 4π/|G|² (1 - cos(|G| R))
#     for G ≠ 0 and v_c(0) = 2π R² (the finite limit).
#
# The per-G kernel `truncated_coulomb_fourier(G_cart, n_per, R, aiso_unit)`
# takes only isbits parameters so that it can be safely used inside GPU `map`
# closures. A convenience wrapper `truncated_coulomb_fourier(G_cart, model)`
# extracts the parameters from the model for one-off calls (e.g. in tests).

"""
Truncation radius `R` used by the Rozzi truncated Coulomb method for a given
`Model`. Chosen automatically as half the minimum edge length over the
non-electrostatically-periodic lattice directions. Returns `zero(T)` for fully
periodic models.
"""
function truncated_coulomb_radius(model::Model{T}) where {T}
    n_periodic_electrostatics(model) == 3 && return zero(T)
    lengths = T[norm(model.lattice[:, i])
                for i = 1:3 if !is_electrostatics_periodic(model, i)]
    return minimum(lengths) / 2
end

"""
Precomputed Rozzi truncated Coulomb parameters for `model`: the number of
periodic electrostatic directions, the truncation radius `R`, and the unit
vector along the isolated direction in the 2D slab case (zero otherwise).
Grouping these as a `NamedTuple` lets us pass only isbits quantities to the
GPU-friendly `truncated_coulomb_fourier` method below.
"""
function truncated_coulomb_params(model::Model{T}) where {T}
    n_per = n_periodic_electrostatics(model)
    R = T(truncated_coulomb_radius(model))
    aiso_unit = if n_per == 2
        iiso = findfirst(i -> !is_electrostatics_periodic(model, i), 1:3)::Int
        a = model.lattice[:, iiso]
        Vec3{T}(a / norm(a))
    else
        zero(Vec3{T})
    end
    (; n_per, R, aiso_unit)
end

"""
Fourier-space value of the truncated Coulomb kernel at the Cartesian vector
`G_cart`, using precomputed parameters `(n_per, R, aiso_unit)` (see
[`truncated_coulomb_params`](@ref)). Only isbits arguments, safe to call
inside GPU kernels.
"""
function truncated_coulomb_fourier(G_cart, n_per::Int, R::T, aiso_unit::Vec3{T}) where {T}
    Gsq = sum(abs2, G_cart)
    if n_per == 3
        # Fully periodic: standard Coulomb with neutralising background.
        return iszero(Gsq) ? zero(T) : 4T(π) / Gsq
    elseif n_per == 0
        # Fully isolated 0D: spherical cutoff at R.
        iszero(Gsq) && return 2T(π) * R^2
        Gnorm = sqrt(Gsq)
        return 4T(π) * (1 - cos(Gnorm * R)) / Gsq
    elseif n_per == 2
        # 2D slab: G_z is the component along the isolated lattice direction.
        iszero(Gsq) && return zero(T)
        Gz = dot(G_cart, aiso_unit)
        Gpar = sqrt(max(Gsq - Gz^2, zero(T)))
        return 4T(π) * (1 - exp(-Gpar * R) * cos(Gz * R)) / Gsq
    else  # n_per == 1
        error("Truncated Coulomb for 1D-periodic wire geometries is not yet " *
              "implemented. TODO: add the Bessel-function formulas from " *
              "Rozzi et al. (2006), with special treatment of the G_∥=0 limit.")
    end
end

"""
Convenience wrapper extracting the truncation parameters from `model` on the
fly. Intended for one-off calls (tests, setup code); in tight loops use
[`truncated_coulomb_params`](@ref) and pass the tuple directly so the kernel
remains GPU-friendly.
"""
function truncated_coulomb_fourier(G_cart, model::Model)
    (; n_per, R, aiso_unit) = truncated_coulomb_params(model)
    truncated_coulomb_fourier(G_cart, n_per, R, aiso_unit)
end
