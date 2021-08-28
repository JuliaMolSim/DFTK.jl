using SpecialFunctions: erfc

"""
Ewald term: electrostatic energy per unit cell of the array of point
charges defined by `model.atoms` in a uniform background of
compensating charge yielding net neutrality.
"""
struct Ewald end
(::Ewald)(basis) = TermEwald(basis)

struct TermEwald <: Term
    basis::PlaneWaveBasis
    energy::Real  # precomputed energy
end
function TermEwald(basis::PlaneWaveBasis)
    # precompute Ewald energy
    energy = energy_ewald(basis.model)
    TermEwald(basis, energy)
end

function ene_ops(term::TermEwald, ψ, occ; kwargs...)
    ops = [NoopOperator(term.basis, kpoint) for kpoint in term.basis.kpoints]
    (E=term.energy, ops=ops)
end

@timing "forces: Ewald" function compute_forces(term::TermEwald, ψ, occ; kwargs...)
    T = eltype(term.basis)
    atoms = term.basis.model.atoms
    # TODO this could be precomputed
    # Compute forces in the "flat" representation used by ewald
    forces_ewald = zeros(Vec3{T}, sum(length(positions) for (elem, positions) in atoms))
    energy_ewald(term.basis.model; forces=forces_ewald)
    # translate to the "folded" representation
    f = [zeros(Vec3{T}, length(positions)) for (type, positions) in atoms]
    count = 0
    for i = 1:length(atoms)
        for j = 1:length(atoms[i][2])
            count += 1
            f[i][j] += forces_ewald[count]
        end
    end
    @assert count == sum(at -> length(at[2]), atoms)
    f
end

function energy_ewald(model::Model; kwargs...)
    charges   = [charge_ionic(elem) for (elem, positions) in model.atoms for pos in positions]
    positions = [pos for (_, positions) in model.atoms for pos in positions]
    isempty(charges) && return zero(eltype(model.lattice))

    # DFTK currently assumes that the compensating charge in the electronic and nuclear
    # terms is equal and of opposite sign. See also the PSP correction term, where n_electrons
    # is used synonymously for sum of charges
    @assert sum(charges) == model.n_electrons
    energy_ewald(model.lattice, charges, positions; kwargs...)
end

"""
Compute the electrostatic interaction energy per unit cell between point
charges in a uniform background of compensating charge to yield net
neutrality. the `lattice` and `recip_lattice` should contain the
lattice and reciprocal lattice vectors as columns. `charges` and
`positions` are the point charges and their positions (as an array of
arrays) in fractional coordinates. If `forces` is not nothing, minus the derivatives
of the energy with respect to `positions` is computed.
"""
function energy_ewald(lattice, charges, positions; η=nothing, forces=nothing)
    T = eltype(lattice)

    for i=1:3
        if norm(lattice[:,i]) == 0
            ## TODO should something more clever be done here? For now
            ## we assume that we are not interested in the Ewald
            ## energy of non-3D systems
            return T(0)
        end
    end
    energy_ewald(lattice, compute_recip_lattice(lattice), charges, positions; η, forces)
end

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

    #
    # Numerical cutoffs
    #
    # The largest argument to the exp(-x) function to obtain a numerically
    # meaningful contribution. The +5 is for safety.
    max_exponent = -log(eps(T)) + 5

    # The largest argument to the erfc function for various precisions.
    # To get an idea:
    #   erfc(5) ≈ 1e-12,  erfc(8) ≈ 1e-29,  erfc(10) ≈ 2e-45,  erfc(14) ≈ 3e-87
    max_erfc_arg = 100
    try
        max_erfc_arg = Dict(Float32 => 5, Float64 => 8, BigFloat => 14)[T]
    catch KeyError
        # Fallback for not yet implemented cutoffs
        max_erfc_arg = something(findfirst(arg -> 100 * erfc(arg) < eps(T), 1:100), 100)
    end

    #
    # Reciprocal space sum
    #
    # Initialize reciprocal sum with correction term for charge neutrality
    sum_recip::T = - (sum(charges)^2 / 4η^2)

    # Function to return the indices corresponding
    # to a particular shell
    # TODO switch to an O(N) implementation
    function shell_indices(ish)
        [[i,j,k] for i in -ish:ish for j in -ish:ish for k in -ish:ish
         if maximum(abs.([i,j,k])) == ish]
    end

    # Loop over reciprocal-space shells
    gsh = 1 # Exclude G == 0
    any_term_contributes = true
    while any_term_contributes
        any_term_contributes = false

        # Compute G vectors and moduli squared for this shell patch
        for G in shell_indices(gsh)
            Gsq = sum(abs2, recip_lattice * G)

            # Check if the Gaussian exponent is small enough
            # for this term to contribute to the reciprocal sum
            exponent = Gsq / 4η^2
            if exponent > max_exponent
                continue
            end

            cos_strucfac = sum(Z * cos(2T(π) * dot(r, G)) for (r, Z) in zip(positions, charges))
            sin_strucfac = sum(Z * sin(2T(π) * dot(r, G)) for (r, Z) in zip(positions, charges))
            sum_strucfac = cos_strucfac^2 + sin_strucfac^2

            any_term_contributes = true
            sum_recip += sum_strucfac * exp(-exponent) / Gsq

            if forces !== nothing
                for (ir, r) in enumerate(positions)
                    Z = charges[ir]
                    dc = -Z*2T(π)*G*sin(2T(π) * dot(r, G))
                    ds = +Z*2T(π)*G*cos(2T(π) * dot(r, G))
                    dsum = 2cos_strucfac*dc + 2sin_strucfac*ds
                    forces_recip[ir] -= dsum * exp(-exponent)/Gsq
                end
            end
        end
        gsh += 1
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

    # Loop over real-space shells
    rsh = 0 # Include R = 0
    any_term_contributes = true
    while any_term_contributes || rsh <= 1
        any_term_contributes = false

        # Loop over R vectors for this shell patch
        for R in shell_indices(rsh)
            for i = 1:length(positions), j = 1:length(positions)
                ti = positions[i]
                Zi = charges[i]
                tj = positions[j]
                Zj = charges[j]

                # Avoid self-interaction
                if rsh == 0 && ti == tj
                    continue
                end

                Δr = lattice * (ti - tj - R)
                dist = norm(Δr)

                # erfc decays very quickly, so cut off at some point
                if η * dist > max_erfc_arg
                    continue
                end

                any_term_contributes = true
                energy_contribution = Zi * Zj * erfc(η * dist) / dist 
                sum_real += energy_contribution
                if forces !== nothing
                    # `dE_ddist` is the derivative of `energy_contribution` w.r.t. `dist`
                    dE_ddist = Zi * Zj * η * (-2exp(-(η * dist)^2) / sqrt(T(π)))
                    dE_ddist -= energy_contribution
                    dE_dti = lattice' * ((dE_ddist / dist^2) * Δr)
                    forces_real[i] -= dE_dti
                    forces_real[j] += dE_dti
                end
            end # i,j
        end # R
        rsh += 1
    end
    energy = (sum_recip + sum_real) / 2  # Divide by 2 (because of double counting)
    if forces !== nothing
        forces .= (forces_recip .+ forces_real) ./ 2
    end
    energy
end
