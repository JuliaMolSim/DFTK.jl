struct PspHgh
    Zion::Int            # Ionic charge (Z - valence electrons)
    rloc                 # Range of local Gaussian charge distribution
    cloc::SVector{4}     # Coefficients for the local part
    lmax::Int            # Maximal angular momentum in the non-local part
    rp::Vector           # Projector radius for each angular momentum
    h::Vector            # Projector coupling coefficients per AM channel
    identifier::String   # String identifying the PSP
    description::String  # Descriptive string
end


"""
    PspHgh(Zion::Number, rloc::Number, cloc::Vector, rp::Vector, h::Vector;
           identifier="", description="")

Construct a Hartwigsen, Goedecker, Teter, Hutter separable dual-space
Gaussian pseudopotential (1998). The required parameters are the ionic
charge `Zion` (total charge - valence electrons), the range for the local
Gaussian charge distribution `rloc`, the coefficients for the local part
`cloc`, the projector radius `rp` (one per AM channel) and the non-local
coupling coefficients between the projectors `h` (one matrix per AM channel).
"""
function PspHgh(Zion, rloc, cloc, rp, h::Vector{Matrix{T}};
                identifier="", description="") where T
    @assert length(rp) == length(h) "Length of rp and h do not agree"
    lmax = length(h) - 1

    @assert length(cloc) <= 4 "length(cloc) > 4 not supported."
    if length(cloc) < 4
        n_extra = 4 - length(cloc)
        cloc = [cloc; zeros(n_extra)]
    end

    PspHgh(Zion, rloc, cloc, lmax, rp, h, identifier, description)
end


"""
    parse_hgh_file(path; identifier="")

Parse an HGH pseudopotential file and construct the PspHgh object.
If `identifier` is given, this identifier will be set.
"""
function parse_hgh_file(path; identifier="")
    lines = readlines(path)
    description = lines[1]

    # lines[2] contains the number of projectors for each AM channel
    m = match(r"^ *(([0-9]+ *)+)", lines[2])
    n_elec = [parse(Int, part) for part in split(m[1])]
    Zion = sum(n_elec)
    lmax = length(n_elec) - 1

    # lines[3] contains rloc nloc and coefficients for it
    m = match(r"^ *([-.0-9]+) *([0-9]+) *(([-.0-9]+ *)+)", lines[3])
    rloc = parse(Float64, m[1])
    nloc = parse(Int, m[2])
    cloc = [parse(Float64, part) for part in split(m[3])]
    @assert length(cloc) == nloc

    # lines[4] contains (lmax + 1) again
    m = match(r"^ *([0-9]+)", lines[4])
    @assert lmax == parse(Int, m[1]) - 1

    rp = Vector{Float64}(undef, lmax + 1)
    h = Vector{Matrix{Float64}}(undef, lmax + 1)
    cur = 5  # Current line to parse
    for l in 0:lmax
        m = match(r"^ *([-.0-9]+) *([0-9]+) *(([-.0-9]+ *)+)", lines[cur])
        rp[l + 1] = parse(Float64, m[1])
        nproj = parse(Int, m[2])
        h[l + 1] = Matrix{Float64}(undef, nproj, nproj)

        hcoeff = [parse(Float64, part) for part in split(m[3])]
        for i in 1:nproj
            for j in i:nproj
                h[l + 1][j, i] = h[l + 1][i, j] = hcoeff[j - i + 1]
            end

            cur += 1
            if cur > length(lines)
                break
            end
            m = match(r"^ *(([-.0-9]+ *)+)", lines[cur])
            hcoeff = [parse(Float64, part) for part in split(m[1])]
        end
    end

    PspHgh(Zion, rloc, cloc, rp, h; identifier=identifier,
           description=description)
end


"""
    eval_psp_local_fourier

Evaluate the local part of the pseudopotential in reciprocal space.
Computes <e_G|Vloc|e_{G+ΔG}> without taking into account the structure factor
and the (4π / Ω) spherical Hankel transform prefactor.
`ΔG` should be in cartesian coordinates.
"""
function eval_psp_local_fourier(psp::PspHgh, ΔG::AbstractVector)
    # TODO Use Fractional coordinates here ?
    Gsq = sum(abs2, ΔG)
    Grsq = Gsq * psp.rloc^2

    convert(eltype(ΔG),
        - psp.Zion / Gsq * exp(-Grsq / 2)
        + sqrt(π/2) * psp.rloc^3 * exp(-Grsq / 2) * (
            + psp.cloc[1]
            + psp.cloc[2] * (  3 -       Grsq                       )
            + psp.cloc[3] * ( 15 -  10 * Grsq +      Grsq^2         )
            + psp.cloc[4] * (105 - 105 * Grsq + 21 * Grsq^2 - Grsq^3)
        )
    )
end


"""
    eval_psp_local_real(psp, r)

Evaluate the local part of the pseudopotential in real space.
The vector `r` should be given in cartesian coordinates.
"""
function eval_psp_local_real(psp::PspHgh, r::AbstractVector)
    # TODO Use Fractional coordinates here ?
    cloc = psp.cloc
    rrsq = sum(abs2, r) / psp.rloc

    convert(eltype(r),
        - psp.Zion / norm(r) * erf(norm(r) / sqrt(2) / psp.rloc)
        + exp(-rrsq / 2) * (cloc[1] + cloc[2] * rrsq + cloc[3] * rrsq^2 + cloc[4] * rrsq^3)
    )
end


"""
    eval_psp_projection_radial(psp::PspHgh, i, l, qsq::Number)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
at the reciprocal lattice point with modulus squared `qsq`.
Compared to the expressions in the GTH and HGH papers, this
expression misses a factor of 1/sqrt(Ω).
"""
function eval_psp_projection_radial(psp::PspHgh, i, l, qsq::Number)
    T = eltype(qsq)
    rp = psp.rp[l + 1]
    q = sqrt.(qsq)
    qrsq::T = qsq .* rp^2
    common::T = 4π^(5 / 4) * sqrt(2^(l + 1) * rp^(2 * l + 3)) * exp.(-qrsq / 2)

    if l == 0
        if i == 1 return @. common end
        # Note: In the next case the HGH paper has an error.
        #       The first 8 in equation (8) should not be under the sqrt-sign
        #       This is the right version (as shown in the GTH paper)
        if i == 2 return @. common *    2  / sqrt(15)  * (3  -   qrsq         ) end
        if i == 3 return @. common * (4/3) / sqrt(105) * (15 - 10qrsq + qrsq^2) end
    end

    if l == 1  # verify expressions
        if i == 1 return @. common * 1     /    sqrt(3) * q end
        if i == 2 return @. common * 2     /  sqrt(105) * q * ( 5 -   qrsq         ) end
        if i == 3 return @. common * 4 / 3 / sqrt(1155) * q * (35 - 14qrsq + qrsq^2) end
    end

    error("Did not implement case of i == $i and l == $l")
end


#=  TODO Not sure if this should belong here. Commented out for now
"""
Evaluate the electrostatic energy contribution of the pseudopotential core.

This is equivalent to the contribution to the DC Fourier component of the
pseudopotential required to not make the electrostatic integal blow up as G -> 0.
"""
function compute_energy_psp_core(psp_::PspHgh, system::System)
    # TODO This routine assumes that there is only one species
    #      with exactly one psp applied for all of them
    pspmap = Dict(system.Zs[1] =>  psp_)

    # Total number of explicitly treated electrons
    # (in neutral crystal with pseudopotentials equal to total ionic charge)
    Nelec = sum(pspmap[Z].Zion for Z in system.Zs)

    ene = 0.0
    for (iat, Z) in enumerate(system.Zs)
        psp = pspmap[Z]
        Zion = psp.Zion
        rloc = psp.rloc
        C(idx) = idx <= length(psp.cloc) ? psp.cloc[idx] : 0.0

        term = (
              Zion * rloc^2 / 2
            + sqrt(π/2) * rloc^3 * (C(1) + 3*C(2) + 15*C(3) + 105*C(4))
        )
        ene += 4π * Nelec / system.unit_cell_volume * term
    end
    return ene
end
=#
