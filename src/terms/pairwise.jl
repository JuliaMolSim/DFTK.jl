"""
Pairwise terms: For example Van der Waals energies, such as Lennard—Jones. The potential is
dependent on the pairwise atomic types.
"""
struct PairwisePotential
    V
    params
end
(P::PairwisePotential)(basis::AbstractBasis) = TermPairwisePotential(P.V, P.params, basis)

struct TermPairwisePotential{TV, Tparams} <:Term
    V::TV
    params::Tparams
    energy::Real
end
function TermPairwisePotential(V::TV, params::Tparams, basis::PlaneWaveBasis{T}) where {TV, Tparams, T}
    TermPairwisePotential(V, params, T(energy_pairwise(basis.model, V, params)))
end

function ene_ops(term::TermPairwisePotential, basis::PlaneWaveBasis, ψ, occ; kwargs...)
    (E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

@timing "forces: Pairwise" function compute_forces(term::TermPairwisePotential,
                                                   basis::PlaneWaveBasis{T}, ψ, occ;
                                                   kwargs...) where {T}
    atoms = basis.model.atoms
    forces_pairwise = zeros(Vec3{T}, sum(length(positions) for (elem, positions) in atoms))
    energy_pairwise(basis.model, term.V, term.params; forces=forces_pairwise)
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

function energy_pairwise(model::Model, V, params; kwargs...)
    atom_types = [element for (element, positions) in model.atoms for _ in positions]
    positions = [pos for (_, positions) in model.atoms for pos in positions]

    energy_pairwise(model.lattice, atom_types, positions, V, params; kwargs...)
end

"""
Compute the pairwise interaction energy per unit cell between atomic sites. If `forces` is
not nothing, minus the derivatives of the energy with respect to `positions` is computed.
The potential is expected to decrease quickly at infinity, so the reciprocal energy is not
computed.
"""
function energy_pairwise(lattice, atom_types, positions, V, params; forces=nothing,
                         max_radius=100)
    T = eltype(lattice)
    @assert length(atom_types) == length(positions)

    if forces !== nothing
        @assert size(forces) == size(positions)
        forces_pairwise = copy(forces)
    end

    # Function to return the indices corresponding
    # to a particular shell
    function shell_indices(nsh)
        ish = nsh
        jsh = nsh
        ksh = nsh
        if norm(lattice[:, 1]) == 0
            ish = 0
        end
        if norm(lattice[:, 2]) == 0
            jsh = 0
        end
        if norm(lattice[:, 3]) == 0
            ksh = 0
        end
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
                param_ij = params[(atom_types[i], atom_types[j])]
                ti = positions[i]
                tj = positions[j]

                # Avoid self-interaction
                if rsh == 0 && ti == tj
                    continue
                end

                Δr = lattice * (ti .- tj .- R)
                dist = norm(Δr)

                # the potential decays very quickly, so cut off at some point
                if dist > max_radius
                    continue
                end
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
