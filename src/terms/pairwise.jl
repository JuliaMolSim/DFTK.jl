struct PairwisePotential
    V
    params
    max_radius
end

@doc raw"""
Pairwise terms: Pairwise potential between nuclei, e.g., Van der Waals potentials, such as
Lennard—Jones terms.
The potential is dependent on the distance between to atomic positions and the pairwise
atomic types:
For a distance `d` between to atoms `A` and `B`, the potential is `V(d, params[(A, B)])`.
The parameters `max_radius` is of `100` by default, and gives the maximum (reduced) distance
between nuclei for which we consider interactions.
"""
function PairwisePotential(V, params; max_radius=100)
    params = Dict(minmax(key[1], key[2]) => value for (key, value) in params)
    PairwisePotential(V, params, max_radius)
end
function (P::PairwisePotential)(basis::PlaneWaveBasis{T}) where {T}
    E = energy_pairwise(basis.model, P.V, P.params; P.max_radius)
    TermPairwisePotential(P.V, P.params, T(P.max_radius), E)
end

struct TermPairwisePotential{TV, Tparams, T} <:Term
    V::TV
    params::Tparams
    max_radius::T
    energy::T
end

function ene_ops(term::TermPairwisePotential, basis::PlaneWaveBasis, ψ, occ; kwargs...)
    (E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

@timing "forces: Pairwise" function compute_forces(term::TermPairwisePotential,
                                                   basis::PlaneWaveBasis{T}, ψ, occ;
                                                   kwargs...) where {T}
    forces = zero(basis.model.positions)
    energy_pairwise(basis.model, term.V, term.params; term.max_radius, forces)
    forces
end


"""
Compute the pairwise interaction energy per unit cell between atomic sites. If `forces` is
not nothing, minus the derivatives of the energy with respect to `positions` is computed.
The potential is expected to decrease quickly at infinity.
"""
function energy_pairwise(model::Model{T}, V, params; kwargs...) where {T}
    isempty(model.atoms) && return zero(T)
    symbols = Symbol.(atomic_symbol.(model.atoms))
    energy_pairwise(model.lattice, symbols, model.positions, V, params; kwargs...)
end


# This could be factorised with Ewald, but the use of `symbols` would slow down the
# computationally intensive Ewald sums. So we leave it as it for now.
function energy_pairwise(lattice, symbols, positions, V, params;
                         max_radius=100, forces=nothing)
    T = eltype(lattice)
    @assert length(symbols) == length(positions)

    if forces !== nothing
        @assert size(forces) == size(positions)
        forces_pairwise = copy(forces)
    end

    # The potential V(dist) decays very quickly with dist = ||A (rj - rk - R)||,
    # so we cut off at some point. We use the bound  ||A (rj - rk - R)|| ≤ max_radius
    # where A is the real-space lattice, rj and rk are atomic positions.
    poslims = [maximum(rj[i] - rk[i] for rj in positions for rk in positions) for i in 1:3]
    Rlims = estimate_integer_lattice_bounds(lattice, max_radius, poslims)

    # Check if some coordinates are not used.
    is_dim_trivial = [norm(lattice[:,i]) == 0 for i=1:3]
    max_shell(n, trivial) = trivial ? 0 : n
    Rlims = max_shell.(Rlims, is_dim_trivial)

    #
    # Energy loop
    #
    sum_pairwise::T = zero(T)
    # Loop over real-space
    for R1 in -Rlims[1]:Rlims[1], R2 in -Rlims[2]:Rlims[2], R3 in -Rlims[3]:Rlims[3]
        R = Vec3(R1, R2, R3)
        for i = 1:length(positions), j = 1:length(positions)
            # Avoid self-interaction
            R == zero(R) && i == j && continue
            ai, aj = minmax(symbols[i], symbols[j])
            param_ij = params[(ai, aj)]
            Δr = lattice * (positions[i] - positions[j] - R)
            dist = norm(Δr)
            energy_contribution = V(dist, param_ij)
            sum_pairwise += energy_contribution
            if forces !== nothing
                # We use ForwardDiff for the forces
                dE_ddist = ForwardDiff.derivative(d -> V(d, param_ij), dist)
                dE_dti = lattice' * dE_ddist / dist * Δr
                forces_pairwise[i] -= dE_dti
                forces_pairwise[j] += dE_dti
            end
        end # i,j
    end # R
    energy = sum_pairwise / 2  # Divide by 2 (because of double counting)
    if forces !== nothing
        forces .= forces_pairwise ./ 2
    end
    energy
end
