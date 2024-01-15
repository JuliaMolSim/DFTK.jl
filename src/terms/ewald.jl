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

struct TermEwald{T} <: Term
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

"""
Standard computation of energy and forces.
"""
function energy_forces_ewald(lattice::AbstractArray{T}, charges::AbstractArray,
                             positions; kwargs...) where {T}
    energy_forces_ewald(T, lattice, charges, positions, zero(Vec3{T}), nothing)
end

"""
Computation for phonons; required to build the dynamical matrix.
"""
function energy_forces_ewald(lattice::AbstractArray{T}, charges, positions, q,
                             ph_disp; kwargs...) where{T}
    S = promote_type(complex(T), eltype(ph_disp[1]))
    energy_forces_ewald(S, lattice, charges, positions, q, ph_disp; kwargs...)
end

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

# This could be merged with Pairwise, but its use of `atom_types` would slow down this
# computationally intensive Ewald sums. So we leave it as it for now.
# Phonons:
# Computes the local energy and forces on the atoms of the reference unit cell 0, for an
# infinite array of atoms at positions r_{iR} = positions[i] + R + ph_disp[i]*e^{-iq·R}.
# `q` is the phonon `q`-point (`Vec3`), and `ph_disp` a list of `Vec3` displacements to
# compute the Fourier transform of (only the direct part of) the force constant matrix.
"""
Compute the electrostatic energy and forces. The energy is the electrostatic interaction
energy per unit cell between point charges in a uniform background of compensating charge to
yield net neutrality. The forces is the opposite of the derivative of the energy with
respect to `positions`.

`lattice` should contain the lattice vectors as columns. `charges` and `positions` are the
point charges and their positions (as an array of arrays) in fractional coordinates.

For now this function returns zero energy and force on non-3D systems. Use a pairwise
potential term if you want to customise this treatment.
"""
function energy_forces_ewald(S, lattice::AbstractArray{T}, charges, positions, q, ph_disp;
                             η=default_η(lattice)) where {T}
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

    # In the real-space term we have erfc(η ||A(rj - rk - R)||),
    # where A is the real-space lattice, rj and rk are atomic positions and
    # thus use the bound  ||A(rj - rk - R)|| * η ≤ max_erfc_arg
    poslims = [maximum(rj[i] - rk[i] for rj in positions for rk in positions) for i = 1:3]
    Rlims = estimate_integer_lattice_bounds(lattice, max_erfc_arg / η, poslims)

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
        cos_strucfac = sum(Z * cos2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sin_strucfac = sum(Z * sin2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sum_strucfac = cos_strucfac^2 + sin_strucfac^2
        sum_recip += sum_strucfac * exp(-Gsq / 4η^2) / Gsq
        for (ir, r) in enumerate(positions)
            Z = charges[ir]
            dc = -Z*2S(π)*G*sin2pi(dot(r, G))
            ds = +Z*2S(π)*G*cos2pi(dot(r, G))
            dsum = cos_strucfac*dc + sin_strucfac*ds
            forces_recip[ir] -= dsum * exp(-Gsq / 4η^2)/Gsq
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

    for R1 in -Rlims[1]:Rlims[1], R2 in -Rlims[2]:Rlims[2], R3 in -Rlims[3]:Rlims[3]
        R = Vec3(R1, R2, R3)
        for i = 1:length(positions), j = 1:length(positions)
            # Avoid self-interaction
            iszero(R) && i == j && continue
            Zi = charges[i]
            Zj = charges[j]
            ti = positions[i]
            tj = positions[j] + R
            if !isnothing(ph_disp)
                ti += ph_disp[i]  # * cis2pi(-dot(q, zeros(3))) === 1
                                  #  as we use the forces at the nuclei in the unit cell
                tj += ph_disp[j] * cis2pi(-dot(q, R))
            end
            Δr = lattice * (ti .- tj)
            dist = norm_cplx(Δr)
            energy_contribution = Zi * Zj * erfc(η * dist) / dist
            sum_real += energy_contribution
            # `dE_ddist` is the derivative of `energy_contribution` w.r.t. `dist`
            # dE_ddist = Zi * Zj * η * (-2exp(-(η * dist)^2) / sqrt(S(π)))
            dE_ddist = ForwardDiff.derivative(zero(T)) do ε
                Zi * Zj * erfc(η * (dist + ε))
            end
            dE_ddist -= energy_contribution
            dE_ddist /= dist
            dE_dti = lattice' * ((dE_ddist / dist) * Δr)
            forces_real[i] -= dE_dti
        end
    end

    (; energy=(sum_recip + sum_real) / 2,  # divide by 2 (because of double counting)
       forces=forces_recip .+ forces_real)
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

    dynmat = zeros(complex(T), n_dim, n_atoms, n_dim, n_atoms)
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
