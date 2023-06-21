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
The parameters `max_radius` is of `100` by default, and gives the maximum distance (in
Cartesian coordinates) between nuclei for which we consider interactions.
"""
function PairwisePotential(V, params; max_radius=100)
    params = Dict(minmax(key[1], key[2]) => value for (key, value) in params)
    PairwisePotential(V, params, max_radius)
end
@timing "precomp: Pairwise" function (P::PairwisePotential)(basis::PlaneWaveBasis{T}) where {T}
    (; energy, forces) = energy_forces_pairwise(basis.model, P.V, P.params; P.max_radius)
    TermPairwisePotential(P.V, P.params, T(P.max_radius), energy, forces)
end

struct TermPairwisePotential{TV, Tparams, T} <:Term
    V::TV
    params::Tparams
    max_radius::T
    energy::T
    forces::Vector{Vec3{T}}
end

function ene_ops(term::TermPairwisePotential, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end
compute_forces(term::TermPairwisePotential, ::PlaneWaveBasis, ψ, occ; kwargs...) = term.forces

"""
Compute the pairwise energy and forces. The energy is the interaction energy per unit cell
between atomic sites. The forces is the opposite of the derivative of the energy with
respect to `positions`.

The potential is expected to decrease quickly at infinity.
"""
function energy_forces_pairwise(model::Model{T}, V, params; kwargs...) where {T}
    isempty(model.atoms) && return (; energy=zero(T), forces=zero(model.positions))
    symbols = Symbol.(atomic_symbol.(model.atoms))
    energy_forces_pairwise(model.lattice, symbols, model.positions, V, params; kwargs...)
end


# This could be merged with Ewald, but the use of `symbols` would slow down the
# computationally intensive Ewald sums. So we leave it as it for now.
# `q` is the phonon `q`-point (`Vec3`), and `ph_disp` a list of `Vec3` displacements to
# compute the Fourier transform of the force constant matrix.
# Computes the local energy and forces on the atoms of the reference unit cell 0, for an
# infinite array of atoms at positions r_{iR} = positions[i] + R + ph_disp[i]*e^{iq·R}.
function energy_forces_pairwise(lattice, symbols, positions, V, params; max_radius=100,
                                ph_disp=nothing, q=nothing)
    isnothing(ph_disp) && @assert isnothing(q)
    @assert length(symbols) == length(positions)

    T = eltype(positions[1])
    if !isnothing(ph_disp)
        @assert !isnothing(q)
        T = promote_type(complex(T), eltype(ph_disp[1]))
        @assert size(ph_disp) == size(positions)
    end

    # The potential V(dist) decays very quickly with dist = ||A (rj - rk - R)||,
    # so we cut off at some point. We use the bound  ||A (rj - rk - R)|| ≤ max_radius
    # where A is the real-space lattice, rj and rk are atomic positions.
    poslims = [maximum(rj[i] - rk[i] for rj in positions for rk in positions) for i in 1:3]
    Rlims = estimate_integer_lattice_bounds(lattice, max_radius, poslims)

    # Check if some coordinates are not used.
    is_dim_trivial = iszero.(eachcol(lattice))
    max_shell(n, trivial) = trivial ? 0 : n
    Rlims = max_shell.(Rlims, is_dim_trivial)

    #
    # Energy loop
    #
    sum_pairwise::T = zero(T)
    forces = zeros(Vec3{T}, length(positions))
    # Loop over real-space
    for R1 in -Rlims[1]:Rlims[1], R2 in -Rlims[2]:Rlims[2], R3 in -Rlims[3]:Rlims[3]
        R = Vec3(R1, R2, R3)
        for i = 1:length(positions), j = 1:length(positions)
            # Avoid self-interaction
            iszero(R) && i == j && continue
            ai, aj = minmax(symbols[i], symbols[j])
            param_ij = params[(ai, aj)]
            ti = positions[i]
            tj = positions[j] + R
            if !isnothing(ph_disp)
                ti += ph_disp[i]  # * cis2pi(dot(q, zeros(3))) === 1
                                  #  as we use the forces at the nuclei in the unit cell
                tj += ph_disp[j] * cis2pi(dot(q, R))
            end
            Δr = lattice * (ti .- tj)
            dist = norm_cplx(Δr)
            energy_contribution = V(dist, param_ij)
            sum_pairwise += energy_contribution
            dE_ddist = ForwardDiff.derivative(zero(real(eltype(dist)))) do ε
                V(dist + ε, param_ij)
            end
            dE_dti = lattice' * dE_ddist / dist * Δr
            forces[i] -= dE_dti
        end # i,j
    end # R

    (; energy=sum_pairwise / 2,  # divide by 2 (because of double counting)
       forces)
end
