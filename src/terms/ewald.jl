import SpecialFunctions: erfc

"""
Ewald term: electrostatic energy per unit cell of the array of point
charges defined by `model.atoms` in a uniform background of
compensating charge yielding net neutrality.
"""
struct Ewald end
(::Ewald)(basis) = TermEwald(basis)

struct TermEwald{T} <: Term
    energy::T                # precomputed energy
    forces::Vector{Vec3{T}}  # and forces
end
@timing "precomp: Ewald" function TermEwald(basis::PlaneWaveBasis{T}) where {T}
    energy, forces = energy_forces_ewald(basis.model; compute_forces=true)
    TermEwald(energy, forces)
end

function ene_ops(term::TermEwald, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (; E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end
compute_forces(term::TermEwald, ::PlaneWaveBasis, ψ, occupation; kwargs...) = term.forces

function energy_forces_ewald(model::Model{T}; kwargs...) where {T}
    isempty(model.atoms) && return (; energy=zero(T), forces=zero(model.positions))
    charges = T.(charge_ionic.(model.atoms))
    energy_forces_ewald(model.lattice, charges, model.positions; kwargs...)
end

# This could be merged with Pairwise, but its use of `atom_types` would slow down this
# computationally intensive Ewald sums. So we leave it as it for now.
"""
Compute the electrostatic interaction energy per unit cell between point charges in
a uniform background of compensating charge to yield net neutrality.`lattice` should
contain the lattice vectors as columns. `charges` and `positions` are the point charges and
their positions (as an array of arrays) in fractional coordinates. If `compute_forces` is
true, minus the derivatives of the energy with respect to `positions` is computed.

For now this function returns zero energy and force on non-3D systems. Use a pairwise
potential term if you want to customise this treatment.
"""
function energy_forces_ewald(lattice::AbstractArray{T}, charges, positions;
                             η=nothing, compute_forces=false) where {T}
    # TODO should something more clever be done here? For now
    # we assume that we are not interested in the Ewald
    # energy of non-3D systems
    any(iszero.(eachcol(lattice))) && return zero(T)

    recip_lattice = compute_recip_lattice(lattice)
    @assert length(charges) == length(positions)
    if isnothing(η)
        # Balance between reciprocal summation and real-space summation
        # with a slight bias towards reciprocal summation
        η = sqrt(sqrt(T(1.69) * norm(recip_lattice ./ 2T(π)) / norm(lattice))) / 2
    end
    if compute_forces
        forces_real  = zeros(Vec3{T}, length(positions))
        forces_recip = zeros(Vec3{T}, length(positions))
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
        iszero(G) && continue
        Gsq = norm2(recip_lattice * G)
        cos_strucfac = sum(Z * cos2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sin_strucfac = sum(Z * sin2pi(dot(r, G)) for (r, Z) in zip(positions, charges))
        sum_strucfac = cos_strucfac^2 + sin_strucfac^2
        sum_recip += sum_strucfac * exp(-Gsq / 4η^2) / Gsq
        if compute_forces
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
    if compute_forces
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
            iszero(R) && i == j && continue
            Zi = charges[i]
            Zj = charges[j]
            Δr = lattice * (positions[i] - positions[j] - R)
            dist = norm(Δr)
            energy_contribution = Zi * Zj * erfc(η * dist) / dist
            sum_real += energy_contribution
            if compute_forces
                # `dE_ddist` is the derivative of `energy_contribution` w.r.t. `dist`
                # dE_ddist = Zi * Zj * η * (-2exp(-(η * dist)^2) / sqrt(T(π)))
                dE_ddist = ForwardDiff.derivative(zero(T)) do ε
                    Zi * Zj * erfc(η * (dist + ε))
                end
                dE_ddist -= energy_contribution
                dE_ddist /= dist
                dE_dti = lattice' * ((dE_ddist / dist) * Δr)
                forces_real[i] -= dE_dti
                forces_real[j] += dE_dti
            end
        end
    end
    energy = (sum_recip + sum_real) / 2  # Divide by 2 (because of double counting)
    res = (; energy)

    if compute_forces
        forces = (forces_recip .+ forces_real) ./ 2
        res = merge(res, (; forces))
    end

    res
end
