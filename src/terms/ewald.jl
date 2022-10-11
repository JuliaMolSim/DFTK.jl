import SpecialFunctions: erfc

"""
Ewald term: electrostatic energy per unit cell of the array of point
charges defined by `model.atoms` in a uniform background of
compensating charge yielding net neutrality.
"""
struct Ewald end
(::Ewald)(basis) = TermEwald(basis)

struct TermEwald{T} <: Term
    energy::T  # precomputed energy
end
function TermEwald(basis::PlaneWaveBasis{T}) where {T}
    TermEwald(T(energy_ewald(basis.model)))
end

function ene_ops(term::TermEwald, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

@timing "forces: Ewald" function compute_forces(term::TermEwald, basis::PlaneWaveBasis{T},
                                                ψ, occupation; kwargs...) where {T}
    # TODO this could be precomputed
    forces = zero(basis.model.positions)
    energy_ewald(basis.model; forces)
    forces
end

function energy_ewald(model::Model{T}; kwargs...) where {T}
    isempty(model.atoms) && return zero(T)
    charges = T.(charge_ionic.(model.atoms))
    energy_ewald(model.lattice, charges, model.positions; kwargs...)
end

"""
Compute the electrostatic interaction energy per unit cell between point
charges in a uniform background of compensating charge to yield net
neutrality. The `lattice` and `recip_lattice` should contain the
lattice and reciprocal lattice vectors as columns. `charges` and
`positions` are the point charges and their positions (as an array of
arrays) in fractional coordinates. If `forces` is not nothing, minus the derivatives
of the energy with respect to `positions` is computed.
"""
function energy_ewald(lattice, charges, positions; η=nothing, forces=nothing)
    T = eltype(lattice)
    for i=1:3
        if iszero(lattice[:, i])
            # TODO should something more clever be done here? For now
            # we assume that we are not interested in the Ewald
            # energy of non-3D systems
            return zero(T)
        end
    end
    energy_ewald(lattice, compute_recip_lattice(lattice), charges, positions; η, forces)
end

# This could be factorised with Pairwise, but its use of `atom_types` would slow down this
# computationally intensive Ewald sums. So we leave it as it for now.
function energy_ewald(lattice, recip_lattice, charges, positions; η=nothing, forces=nothing)
    T = eltype(lattice)
    @assert T == eltype(recip_lattice)
    @assert length(charges) == length(positions)
    if η === nothing
        # Balance between reciprocal summation and real-space summation
        # with a slight bias towards reciprocal summation
        η = sqrt(sqrt(T(1.69) * norm(recip_lattice ./ 2T(π)) / norm(lattice))) / 2
    end
    if forces !== nothing
        @assert size(forces) == size(positions)
        forces_real = copy(forces)
        forces_recip = copy(forces)
    end

    # Numerical cutoffs to obtain meaningful contributions. These are very conservative.
    # The largest argument to the exp(-x) function
    max_exp_arg = -log(eps(T)) + 5  # add some wiggle room
    max_erfc_arg = sqrt(max_exp_arg)  # erfc(x) ~= exp(-x^2)/(sqrt(π)x) for large x

    # Precomputing summation bounds from cutoffs.
    # In the reciprocal-space term we have exp(-||B G||^2 / 4η^2),
    # where B is the reciprocal-space lattice, and
    # thus use the bound  ||B G|| / 2η ≤ sqrt(max_exp_arg)
    Glims = estimate_integer_lattice_bounds(recip_lattice, sqrt(max_exp_arg) * 2η)

    # In the real-space term we have erfc(η ||A(rj - rk - R)||),
    # where A is the real-space lattice, rj and rk are atomic positions and
    # thus use the bound  ||A(rj - rk - R)|| * η ≤ max_erfc_arg
    poslims = [maximum(rj[i] - rk[i] for rj in positions for rk in positions) for i in 1:3]
    Rlims = estimate_integer_lattice_bounds(lattice, max_erfc_arg / η, poslims)

    #
    # Reciprocal space sum
    #
    # Initialize reciprocal sum with correction term for charge neutrality
    sum_recip::T = - (sum(charges)^2 / 4η^2)

    for G1 in -Glims[1]:Glims[1], G2 in -Glims[2]:Glims[2], G3 in -Glims[3]:Glims[3]
        G = Vec3(G1, G2, G3)
        (G == zero(G)) && continue
        Gsq = sum(abs2, recip_lattice * G)
        cos_strucfac = sum(Z * cos2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sin_strucfac = sum(Z * sin2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sum_strucfac = cos_strucfac^2 + sin_strucfac^2
        sum_recip += sum_strucfac * exp(-Gsq / 4η^2) / Gsq
        if forces !== nothing
            for (ir, r) in enumerate(positions)
                Z = charges[ir]
                dc = -Z*2T(π)*G*sin2pi(dot(r, G))
                ds = +Z*2T(π)*G*cos2pi(dot(r, G))
                dsum = 2cos_strucfac*dc + 2sin_strucfac*ds
                forces_recip[ir] -= dsum * exp(-Gsq / 4η^2)/Gsq
            end
        end
    end

    # Amend sum_recip by proper scaling factors:
    sum_recip *= 4T(π) / compute_unit_cell_volume(lattice)
    if forces !== nothing
        forces_recip .*= 4T(π) / compute_unit_cell_volume(lattice)
    end

    #
    # Real-space sum
    #
    # Initialize real-space sum with correction term for uniform background
    sum_real::T = -2η / sqrt(T(π)) * sum(Z -> Z^2, charges)

    for R1 in -Rlims[1]:Rlims[1], R2 in -Rlims[2]:Rlims[2], R3 in -Rlims[3]:Rlims[3]
        R = Vec3(R1, R2, R3)
        for i = 1:length(positions), j = 1:length(positions)
            # Avoid self-interaction
            R == zero(R) && i == j && continue
            Zi = charges[i]
            Zj = charges[j]
            Δr = lattice * (positions[i] - positions[j] - R)
            dist = norm(Δr)
            energy_contribution = Zi * Zj * erfc(η * dist) / dist
            sum_real += energy_contribution
            if forces !== nothing
                # `dE_ddist` is the derivative of `energy_contribution` w.r.t. `dist`
                dE_ddist = Zi * Zj * η * (-2exp(-(η * dist)^2) / sqrt(T(π)))
                dE_ddist -= energy_contribution
                dE_ddist /= dist
                dE_dti = lattice' * ((dE_ddist / dist) * Δr)
                forces_real[i] -= dE_dti
                forces_real[j] += dE_dti
            end
        end
    end
    energy = (sum_recip + sum_real) / 2  # Divide by 2 (because of double counting)
    if forces !== nothing
        forces .= (forces_recip .+ forces_real) ./ 2
    end
    energy
end
