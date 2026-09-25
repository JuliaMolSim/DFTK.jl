import SpecialFunctions: erfc

"""
Ewald term: electrostatic energy per unit cell of the array of point
charges defined by `model.atoms` in a uniform background of
compensating charge yielding net neutrality.
"""
Base.@kwdef struct Ewald
    η = nothing  # Parameter used for the splitting 1/r ≡ erf(η·r)/r + erfc(η·r)/r
                 # (or nothing if autoselected)
end
(ewald::Ewald)(basis) = TermEwald(basis; η=something(ewald.η, default_η(basis.model.lattice)))

struct TermEwald{T} <: TermLinear
    energy::T                # precomputed energy
    forces::Vector{Vec3{T}}  # and forces
    η::T                     # Parameter used for the splitting
    #                          1/r ≡ erf(η·r)/r + erfc(η·r)/r
end
@timing "precomp: Ewald" function TermEwald(basis::PlaneWaveBasis{T};
                                            η=default_η(basis.model.lattice)) where {T}
    model = basis.model
    charges = charge_ionic.(model.atoms)
    (; energy, forces) = energy_forces_ewald(model.lattice, charges, model.positions; η)
    TermEwald(energy, forces, η)
end

function ene_ops(term::TermEwald, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (; E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end
compute_forces(term::TermEwald, ::PlaneWaveBasis, ψ, occupation; kwargs...) = term.forces

# To compute the electrostatics of the system, we use the Ewald splitting method due to the
# slow convergence of the energy in ``1/r``.
# It uses the the identity ``1/r ≡ erf(η·r)/r + erfc(η·r)/r``, where the first (smooth) part
# of the energy term is computed in the reciprocal space and the second (singular) one in
# the real-space.
# `η` is an arbitrary parameter that enables to balance the computation of those to parts.
# By default, we choose it to have a slight bias towards the reciprocal summation.
function default_η(lattice::AbstractArray{T}) where {T}
    any(iszero.(eachcol(lattice))) && return  # We won't compute anything
    recip_lattice = compute_recip_lattice(lattice)
    sqrt(sqrt(T(1.69) * norm(recip_lattice ./ 2T(π)) / norm(lattice))) / 2
end

@doc raw"""
Compute the electrostatic energy and forces. The energy is the electrostatic interaction
energy per unit cell between point charges in a uniform background of compensating charge to
yield net neutrality. The forces is the opposite of the derivative of the energy with
respect to `positions`.

`lattice` should contain the lattice vectors as columns. `charges` and `positions` are the
point charges and their positions (as an array of arrays) in fractional coordinates.

For now this function returns zero energy and force on non-3D systems. Use a pairwise
potential term if you want to customise this treatment.

For phonon (`q` ≠ 0), this computes the local energy and forces on the atoms of the
reference unit cell 0, for an infinite array of atoms at positions
``r_{iR} = {\rm positions}_i + R + {\rm ph_disp}_i e^{-iq·R}``.
`q` is the phonon `q`-point, and `ph_disp` a list of displacements to compute the Fourier
transform of (only the direct part of) the force constant matrix.
"""
function energy_forces_ewald(S, lattice::AbstractArray{T}, charges, positions, q, ph_disp;
                             η=default_η(lattice)) where {T}
    # This could be merged with Pairwise, but its use of `symbols` would slow down this
    # computationally intensive Ewald sums. So we leave it as it for now.
    @assert length(charges) == length(positions)
    if isempty(charges)
        return (; energy=zero(T), forces=zero(positions))
    end

    isnothing(ph_disp) && @assert iszero(q)
    if !isnothing(ph_disp)
        @assert size(ph_disp) == size(positions)
    end

    # TODO should something more clever be done here? For now
    # we assume that we are not interested in the Ewald
    # energy of non-3D systems
    any(iszero.(eachcol(lattice))) && return (; energy=zero(T), forces=zero(positions))

    # Numerical cutoffs to obtain meaningful contributions. These are very conservative.
    # The largest argument to the exp(-x) function
    max_exp_arg = -log(eps(T)) + 5  # add some wiggle room
    max_erfc_arg = sqrt(max_exp_arg)  # erfc(x) ~= exp(-x^2)/(sqrt(π)x) for large x

    # Precomputing summation bounds from cutoffs.
    # In the reciprocal-space term we have exp(-||B G||^2 / 4η^2),
    # where B is the reciprocal-space lattice, and
    # thus use the bound  ||B G|| / 2η ≤ sqrt(max_exp_arg)
    recip_lattice = compute_recip_lattice(lattice)
    Glims = estimate_integer_lattice_bounds(recip_lattice, sqrt(max_exp_arg) * 2η)

    #
    # Reciprocal space sum
    #
    # Initialize reciprocal sum with correction term for charge neutrality
    sum_recip::S = - (sum(charges)^2 / 4η^2)
    forces_recip = zeros(Vec3{S}, length(positions))

    for G1 in -Glims[1]:Glims[1], G2 in -Glims[2]:Glims[2], G3 in -Glims[3]:Glims[3]
        G = Vec3(G1, G2, G3)
        iszero(G) && continue
        Gsq = norm2(recip_lattice * G)
        Gsq > 4η^2 * max_exp_arg && continue
        factor = exp(-Gsq / 4η^2) / Gsq
        cos_strucfac = sum(Z * cos2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sin_strucfac = sum(Z * sin2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sum_strucfac = cos_strucfac^2 + sin_strucfac^2
        sum_recip += sum_strucfac * factor
        for (ir, r) in enumerate(positions)
            Z = charges[ir]
            dc = -Z*2S(π)*G*sin2pi(dot(r, G))
            ds = +Z*2S(π)*G*cos2pi(dot(r, G))
            dsum = cos_strucfac*dc + sin_strucfac*ds
            forces_recip[ir] -= dsum * factor
        end
    end

    # Amend reciprocal quantities by proper scaling factors:
    sum_recip     *= 4S(π) / compute_unit_cell_volume(lattice)
    forces_recip .*= 4S(π) / compute_unit_cell_volume(lattice)

    #
    # Real-space sum
    #
    # Initialize real-space sum with correction term for uniform background
    sum_real::S = -2η / sqrt(S(π)) * sum(Z -> Z^2, charges)
    forces_real = zeros(Vec3{S}, length(positions))

    radius = max_erfc_arg / η
    inv_lattice_t = compute_inverse_lattice(lattice')
    # ||A(d - R)|| ≤ radius implies |d[k] - R[k]| ≤ ||A⁻ᵀ eₖ|| radius.
    image_scales = Vec3(ntuple(k -> norm(inv_lattice_t[:, k]), 3))
    cart_positions = [Vec3(lattice * r) for r in positions]
    cart_displacements = isnothing(ph_disp) ? nothing : [Vec3(lattice * u) for u in ph_disp]
    a1, a2, a3 = Vec3.(eachcol(lattice))
    derivative_factor = -2η / sqrt(S(π))

    for i in eachindex(positions)
        # Accumulate per atom to limit roundoff in large cells.
        sum_real_i = zero(S)
        force_cart = zero(Vec3{S})
        for j in eachindex(positions)
            d = Vec3(positions[i] - positions[j])
            image_bounds = radius * image_scales
            if !isnothing(ph_disp)
                # Include finite displacements, also allowing for the imaginary part of
                # the complex-analytic distance used in the phonon calculation.
                # The ℓ¹ bound avoids norm's zero-vector scaling for complex duals.
                displacement_bound = sum(abs, cart_displacements[i]) +
                                     sum(abs, cart_displacements[j])
                image_bounds += displacement_bound * image_scales
                image_bounds += abs.(ph_disp[i]) + abs.(ph_disp[j])
            end
            Rmin = ceil.(Int, d - image_bounds)
            Rmax = floor.(Int, d + image_bounds)
            Δr_pair = cart_positions[i] - cart_positions[j]
            charge_product = charges[i] * charges[j]
            for R1 = Rmin[1]:Rmax[1]
                Δr1 = Δr_pair - R1 * a1
                for R2 = Rmin[2]:Rmax[2]
                    Δr12 = Δr1 - R2 * a2
                    for R3 = Rmin[3]:Rmax[3]
                        R = Vec3(R1, R2, R3)
                        iszero(R) && i == j && continue
                        Δr = Δr12 - R3 * a3
                        if !isnothing(ph_disp)
                            Δr += cart_displacements[i]
                            Δr -= cart_displacements[j] * cis2pi(-dot(q, R))
                        end
                        dist_sq = sum(x -> x * x, Δr)
                        real(dist_sq) > radius^2 && continue
                        dist = sqrt(dist_sq)
                        energy_contribution = charge_product * erfc(η * dist) / dist
                        sum_real_i += energy_contribution
                        # Derivative of charge_product * erfc(η * dist) / dist.
                        dE_ddist = charge_product * derivative_factor * exp(-η^2 * dist_sq)
                        dE_ddist = (dE_ddist - energy_contribution) / dist
                        force_cart -= (dE_ddist / dist) * Δr
                    end
                end
            end
        end
        sum_real += sum_real_i
        forces_real[i] = lattice' * force_cart
    end

    (; energy=(sum_recip + sum_real) / 2,  # divide by 2 (because of double counting)
       forces=forces_recip .+ forces_real)
end
# For convenience
function energy_forces_ewald(lattice::AbstractArray{T}, charges::AbstractArray,
                             positions; kwargs...) where {T}
    energy_forces_ewald(T, lattice, charges, positions, zero(Vec3{T}), nothing; kwargs...)
end
function energy_forces_ewald(lattice::AbstractArray{T}, charges, positions, q,
                             ph_disp; kwargs...) where{T}
    S = promote_type(complex(T), eltype(ph_disp[1]))
    energy_forces_ewald(S, lattice, charges, positions, q, ph_disp; kwargs...)
end

# TODO: See if there is a way to express this with AD.
function dynmat_ewald_recip(model::Model{T}, s, t; η, q=zero(Vec3{T})) where {T}
    # Numerical cutoffs to obtain meaningful contributions. These are very conservative.
    # The largest argument to the exp(-x) function
    max_exp_arg = -log(eps(T)) + 5  # add some wiggle room

    lattice       = model.lattice
    recip_lattice = model.recip_lattice
    # Precomputing summation bounds from cutoffs.
    # In the reciprocal-space term we have exp(-||B G||^2 / 4η^2),
    # where B is the reciprocal-space lattice, and
    # thus use the bound  ||B G|| / 2η ≤ sqrt(max_exp_arg)
    Glims = estimate_integer_lattice_bounds(recip_lattice, sqrt(max_exp_arg) * 2η)

    charges   = T.(charge_ionic.(model.atoms))
    positions = model.positions
    @assert length(charges) == length(positions)
    ps = positions[s]
    pt = positions[t]

    dynmat_recip = zeros(complex(T), length(q), length(q))
    for G1 in -Glims[1]:Glims[1], G2 in -Glims[2]:Glims[2], G3 in -Glims[3]:Glims[3]
        G = Vec3(G1, G2, G3)
        if !iszero(G + q)
            Gsqq = sum(abs2, recip_lattice * (G + q))
            term = exp(-Gsqq / 4η^2) / Gsqq * charges[t] * charges[s]
            term *= cis2pi(dot(G + q, pt - ps))
            term *= (2T(π) * (G + q)) * transpose(2T(π) * (G + q))
            dynmat_recip += term
        end

        (iszero(G) || t ≢ s) && continue
        Gsq = sum(abs2, recip_lattice * G)

        strucfac = sum(Z * cos2pi(dot(pt - r, G)) for (r, Z) in zip(positions, charges))
        dsum = charges[t] * strucfac
        dsum *= (2T(π) * G) * transpose(2T(π) * G)
        dynmat_recip -= exp(-Gsq / 4η^2) / Gsq * dsum
    end

    # Amend `dynmat_recip` by proper scaling factors:
    dynmat_recip *= 4T(π) / compute_unit_cell_volume(lattice)
end

# Computes the Fourier transform of the force constant matrix of the Ewald term.
function compute_dynmat(ewald::TermEwald, basis::PlaneWaveBasis{T}, ψ, occupation;
                        q=zero(Vec3{T}), kwargs...) where {T}
    model = basis.model
    n_atoms = length(model.positions)
    n_dim = model.n_dim
    charges = T.(charge_ionic.(model.atoms))

    dynmat = zeros(complex(T), 3, n_atoms, 3, n_atoms)
    # Real part
    for s = 1:n_atoms, α = 1:n_dim
        displacement = zero.(model.positions)
        displacement[s] = setindex(displacement[s], one(T), α)
        real_part = -ForwardDiff.derivative(zero(T)) do ε
            ph_disp = ε .* displacement
            forces = energy_forces_ewald(model.lattice, charges, model.positions, q,
                                         ph_disp; ewald.η).forces
            stack(forces)
        end

        dynmat[:, :, α, s] = real_part
    end
    # Reciprocal part
    for s = 1:n_atoms, t = 1:n_atoms
        dynmat[:, t, :, s] += dynmat_ewald_recip(model, t, s; ewald.η, q)
    end
    dynmat
end
