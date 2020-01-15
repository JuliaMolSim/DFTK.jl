using SpecialFunctions: erfc

"""
    energy_ewald(lattice, [recip_lattice, ]charges, positions; η=nothing)

Compute the electrostatic interaction energy per unit cell between point
charges in a uniform background of compensating charge to yield net
neutrality. the `lattice` and `recip_lattice` should contain the
lattice and reciprocal lattice vectors as columns. `charges` and
`positions` are the point charges and their positions (as an array of
arrays) in fractional coordinates.
"""
function energy_ewald(lattice, charges, positions; η=nothing)
    T = eltype(lattice)

    for i=1:3
        @assert norm(lattice[:,i]) != 0
        # Otherwise the formula for the reciprocal lattice
        # computation is not correct
    end
    energy_ewald(lattice, T(2π) * inv(lattice'), charges, positions, η=η)
end
function energy_ewald(lattice, recip_lattice, charges, positions; η=nothing)
    T = eltype(lattice)
    @assert T == eltype(recip_lattice)
    @assert(length(charges) == length(positions),
            "Length of charges and positions does not match")
    if η === nothing
        # Balance between reciprocal summation and real-space summation
        # with a slight bias towards reciprocal summation
        η = sqrt(sqrt(T(1.69) * norm(recip_lattice ./ 2T(π)) / norm(lattice))) / 2
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
    # Initialise reciprocal sum with correction term for charge neutrality
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
        end
        gsh += 1
    end
    # Amend sum_recip by proper scaling factors:
    sum_recip = sum_recip * 4T(π) / abs(det(lattice))

    #
    # Real-space sum
    #
    # Initialise real-space sum with correction term for uniform background
    sum_real::T = -2η / sqrt(T(π)) * sum(Z -> Z^2, charges)

    # Loop over real-space shells
    rsh = 0 # Include R = 0
    any_term_contributes = true
    while any_term_contributes || rsh <= 1
        any_term_contributes = false

        # Loop over R vectors for this shell patch
        for R in shell_indices(rsh)
            for (ti, Zi) in zip(positions, charges), (tj, Zj) in zip(positions, charges)
                # Avoid self-interaction
                if rsh == 0 && ti == tj
                    continue
                end

                dist = norm(lattice * (ti - tj - R))

                # erfc decays very quickly, so cut off at some point
                if η * dist > max_erfc_arg
                    continue
                end

                any_term_contributes = true
                sum_real += Zi * Zj * erfc(η * dist) / dist
            end # iat, jat
        end # R
        rsh += 1
    end
    (sum_recip + sum_real) / 2  # Divide by 1/2 (because of double counting)
end
