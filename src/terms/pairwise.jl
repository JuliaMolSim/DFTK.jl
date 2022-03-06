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
    E = energy_pairwise(basis.model, V, params; max_radius)
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
    atoms = basis.model.atoms
    forces_pairwise = zeros(Vec3{T}, sum(length(positions) for (elem, positions) in atoms))
    energy_pairwise(basis.model, term.V, term.params; term.max_radius, forces=forces_pairwise)
    # translate to the "folded" representation
    f = [zeros(Vec3{T}, length(positions)) for (type, positions) in atoms]
    count = 0
    for i = 1:length(atoms)
        for j = 1:length(atoms[i][2])
            count += 1
            f[i][j] += forces_pairwise[count]
        end
    end
    @assert count == sum(at -> length(at[2]), atoms)
    f
end


"""
Compute the pairwise interaction energy per unit cell between atomic sites. If `forces` is
not nothing, minus the derivatives of the energy with respect to `positions` is computed.
The potential is expected to decrease quickly at infinity.
"""
function energy_pairwise(model::Model{T}, V, params; kwargs...) where {T}
    isempty(model.atoms) && return zero(T)
    symbols = Symbol.(atomic_symbol.(model.atoms))
    energy_pairwise(model.lattice, symbols, positions, V, params; kwargs...)
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

    # Function to return the indices corresponding
    # to a particular shell.
    # Not performance critical, so we do not type the function
    max_shell(n, trivial) = trivial ? 0 : n
    # Check if some coordinates are not used.
    is_dim_trivial = [norm(lattice[:,i]) == 0 for i=1:3]
    function shell_indices(nsh)
        ish, jsh, ksh = max_shell.(nsh, is_dim_trivial)
        [[i,j,k] for i in -ish:ish for j in -jsh:jsh for k in -ksh:ksh
         if maximum(abs.([i,j,k])) == nsh]
    end

    #
    # Energy loop
    #
    sum_pairwise::T = zero(T)
    # Loop over real-space shells
    rsh = 0 # Include R = 0
    any_term_contributes = true
    while any_term_contributes || rsh <= 1
        any_term_contributes = false

        # Loop over R vectors for this shell patch
        for R in shell_indices(rsh)
            for i = 1:length(positions), j = 1:length(positions)
                # Avoid self-interaction
                rsh == 0 && i == j && continue

                ti = positions[i]
                tj = positions[j]
                ai, aj = minmax(symbols[i], symbols[j])
                param_ij =params[(ai, aj)]

                Δr = lattice * (ti .- tj .- R)
                dist = norm(Δr)

                # the potential decays very quickly, so cut off at some point
                dist > max_radius && continue

                any_term_contributes = true
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
        rsh += 1
    end
    energy = sum_pairwise / 2  # Divide by 2 (because of double counting)
    if forces !== nothing
        forces .= forces_pairwise ./ 2
    end
    energy
end
