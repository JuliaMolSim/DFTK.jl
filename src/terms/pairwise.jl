# We cannot use `LinearAlgebra.norm` with complex numbers due to the need to use its
# analytic continuation
function norm_cplx(x)
    # TODO: ForwardDiff bug (https://github.com/JuliaDiff/ForwardDiff.jl/issues/324)
    sqrt(sum(x.*x))
end

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
The parameters `max_radius` is of `1000` by default, and gives the maximum (Cartesian)
distance between nuclei for which we consider interactions.
"""
function PairwisePotential(V, params; max_radius=1000)
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
    TT = eltype(lattice)
    forces = zero(TT, basis.model.positions)
    energy_pairwise(basis.model, term.V, term.params; max_radius=term.max_radius,
                    forces=forces, kwargs...)
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
# TODO: *Beware* of using ForwardDiff to derive this function with complex numbers, use
# multiplications and not powers (https://github.com/JuliaDiff/ForwardDiff.jl/issues/324).
# `q` is the phonon `q`-point (`Vec3`), and `ph_disp` a list of `Vec3` displacements to
# compute the Fourier transform of the force constant matrix.
function energy_pairwise(lattice, symbols, positions, V, params;
                         max_radius=1000, forces=nothing, ph_disp=nothing, q=nothing)
    @assert length(symbols) == length(positions)

    T = eltype(lattice)
    if ph_disp !== nothing
        @assert q !== nothing
        T = promote_type(complex(T), eltype(ph_disp[1]))
        @assert size(ph_disp) == size(positions)
    end

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
                tj = positions[j] + R
                # Phonons `q` points
                if !isnothing(ph_disp)
                    ti += ph_disp[i] # * cis(2T(π)*dot(q, zeros(3))) === 1
                                     #  as we use the forces at the nuclei in the unit cell
                    tj += ph_disp[j] * cis(2T(π)*dot(q, R))
                end
                ai, aj = minmax(symbols[i], symbols[j])
                param_ij =params[(ai, aj)]

                Δr = lattice * (ti .- tj)
                dist = norm_cplx(Δr)

                # the potential decays very quickly, so cut off at some point
                abs(dist) > max_radius && continue

                any_term_contributes = true
                energy_contribution = real(V(dist, param_ij))
                sum_pairwise += energy_contribution
                if forces !== nothing
                    dE_ddist = ForwardDiff.derivative(real(zero(eltype(dist)))) do ε
                        res = V(dist + ε, param_ij)
                        [real(res), imag(res)]
                    end |> x -> complex(x...)
                    dE_dti = lattice' * ((dE_ddist / dist) * Δr)
                    # We need to "break" the symmetry for phonons; at equilibrium, expect
                    # the forces to be zero at machine precision.
                    forces_pairwise[i] -= dE_dti
                end
            end # i,j
        end # R
        rsh += 1
    end
    energy = sum_pairwise / 2  # Divide by 2 (because of double counting)
    if forces !== nothing
        forces .= forces_pairwise
    end
    energy
end
