# Truncated Coulomb kernels for non-periodic electrostatics, following
#
#     C. A. Rozzi, D. Varsano, A. Marini, E. K. U. Gross, A. Rubio,
#     "Exact Coulomb cutoff technique for supercell calculations",
#     Phys. Rev. B 73, 205119 (2006).
#     https://arxiv.org/abs/cond-mat/0601031
#
# These functions return the Fourier transform V_c(G) of a Coulomb interaction
# that has been truncated in the directions of the simulation cell that are
# marked as non-periodic in `model.periodicity`. V_c(G) replaces the usual
# 4π/|G|² kernel in the Hartree and ionic-local electrostatics.
#
# Four cases arise depending on how many lattice directions carry periodic
# electrostatics (as returned by `n_periodic_electrostatics(model)`):
#
#   * 3 — fully periodic: V_c(G) = 4π/|G|² (standard Ewald convention with
#     compensating background, i.e. V_c(0) = 0).
#
#   * 2 — 2D slab: one isolated direction, orthogonal to the periodic plane.
#     V_c(G) = 4π/|G|² (1 - exp(-|G_∥| R) cos(G_z R)), with R half the length
#     of the isolated lattice vector.  G=0 set to 0 (compensating background
#     in the periodic plane).
#
#   * 1 — 1D wire: NOT YET IMPLEMENTED. Will require Bessel-function formulas
#     with special handling of the G_∥=0 limit (Rozzi et al., eqs. 21–22).
#
#   * 0 — 0D fully isolated molecule: spherical cutoff at radius R equal to
#     half the minimum lattice edge length. V_c(G) = 4π/|G|² (1 - cos(|G| R))
#     for G ≠ 0 and V_c(0) = 2π R² (the finite limit).
#
# For case 0, the cell is required to be orthogonal (this is enforced by the
# `Model` constructor when all directions are non-periodic).

"""
Compute the truncation radius `R` used by the Rozzi truncated Coulomb method
for a given `Model`. Chosen automatically as half the minimum edge length
over the non-electrostatically-periodic lattice directions. Returns `zero(T)`
for fully periodic models.
"""
function truncated_coulomb_radius(model::Model{T}) where {T}
    n_periodic_electrostatics(model) == 3 && return zero(T)
    lengths = T[norm(model.lattice[:, i])
                for i = 1:3 if !is_electrostatics_periodic(model, i)]
    return minimum(lengths) / 2
end

"""
Fourier-space value of the truncated Coulomb kernel V_c(G_cart) at the given
Cartesian G vector for the given model. Returns zero for G=0 in the 3D/2D
(compensating-background) cases and a finite value for the 0D case.

This implements the Rozzi et al. formulas for 0D (spherical cutoff) and 2D
(slab cutoff). The 3D case falls back to the standard 4π/|G|² kernel. The 1D
wire case is not yet implemented and will raise an error.
"""
function truncated_coulomb_fourier(G_cart::AbstractVector{T}, model::Model) where {T}
    n_per = n_periodic_electrostatics(model)
    Gsq = sum(abs2, G_cart)

    if n_per == 3
        # Fully periodic: standard Coulomb with neutralising background (G=0 set to 0).
        iszero(Gsq) && return zero(T)
        return 4T(π) / Gsq

    elseif n_per == 0
        # Fully isolated 0D: spherical cutoff at R.
        R = T(truncated_coulomb_radius(model))
        if iszero(Gsq)
            return 2T(π) * R^2
        else
            Gnorm = sqrt(Gsq)
            return 4T(π) * (1 - cos(Gnorm * R)) / Gsq
        end

    elseif n_per == 2
        # 2D slab: find the isolated direction (orthogonal to the periodic plane).
        iiso = findfirst(i -> !is_electrostatics_periodic(model, i), 1:3)
        # Unit vector along the isolated lattice direction (assumed orthogonal
        # to the periodic ones by the Model constructor checks).
        aiso_cart = model.lattice[:, iiso]
        Liso = norm(aiso_cart)
        R = T(Liso / 2)
        ẑ = aiso_cart / Liso
        Gz = dot(G_cart, ẑ)
        Gpar = sqrt(max(Gsq - Gz^2, zero(T)))
        if iszero(Gsq)
            # Compensating background in the periodic plane: drop DC.
            return zero(T)
        else
            return 4T(π) * (1 - exp(-Gpar * R) * cos(Gz * R)) / Gsq
        end

    else  # n_per == 1
        error("Truncated Coulomb for 1D-periodic wire geometries is not yet " *
              "implemented. TODO: add the Bessel-function formulas from " *
              "Rozzi et al. (2006), with special treatment of the G_∥=0 limit.")
    end
end

"""
Build a 3D array of the Rozzi truncated Coulomb Green's function coefficients
for every `G` vector in `basis`, optionally shifted by `q`. The layout matches
`G_vectors(basis)`.
"""
function truncated_coulomb_green_coeffs(basis::PlaneWaveBasis{T};
                                        q=zero(Vec3{T})) where {T}
    model = basis.model
    recip_lattice = model.recip_lattice
    coeffs = map(G -> truncated_coulomb_fourier(recip_lattice * (G + q), model),
                 G_vectors(basis))
    coeffs
end
