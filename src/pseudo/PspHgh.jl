using SpecialFunctions: erf

struct PspHgh
    Zion::Int                   # Ionic charge (Z - valence electrons)
    rloc::Float64               # Range of local Gaussian charge distribution
    cloc::SVector{4,Float64}    # Coefficients for the local part
    lmax::Int                   # Maximal angular momentum in the non-local part
    rp::Vector{Float64}         # Projector radius for each angular momentum
    h::Vector{Matrix{Float64}}  # Projector coupling coefficients per AM channel
    identifier::String          # String identifying the PSP
    description::String         # Descriptive string
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

    # lines[2] contains the number of electrons (and the AM channel in which they sit)
    m = match(r"^ *(([0-9]+ *)+)", lines[2])
    n_elec = [parse(Int, part) for part in split(m[1])]
    Zion = sum(n_elec)

    # lines[3] contains rloc nloc and coefficients for it
    m = match(r"^ *([-.0-9]+) *([0-9]+) *(([-.0-9]+ *)+)", lines[3])
    rloc = parse(Float64, m[1])
    nloc = parse(Int, m[2])
    cloc = [parse(Float64, part) for part in split(m[3])]
    @assert length(cloc) == nloc

    # lines[4] contains the maximal AM channel
    m = match(r"^ *([0-9]+)", lines[4])
    lmax = parse(Int, m[1]) - 1

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
    eval_psp_local_fourier(psp, ΔG)

Evaluate the local part of the pseudopotential in reciprocal space.

This function computes
V(q) = ∫_R^3 Vloc(r) e^{-iqr} dr
     = 4π ∫_{R+} sin(qr)/q r e^{-iqr} dr

[GTH98] (6) except they do it with plane waves normalized by 1/sqrt(Ω).
"""
function eval_psp_local_fourier(psp::PspHgh, ΔG::AbstractVector{T}) where T
    Gsq = sum(abs2, ΔG)
    Grsq::T = Gsq * psp.rloc^2

    4T(π) * (
        - psp.Zion / Gsq * exp(-Grsq / 2)
        + sqrt(T(π)/2) * psp.rloc^3 * exp(-Grsq / 2) * (
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

[GTH98] (1)
"""
function eval_psp_local_real(psp::PspHgh, r::AbstractVector{T}) where T
    cloc = psp.cloc
    rrsq = sum(abs2, r) / psp.rloc

    convert(T,
        - psp.Zion / norm(r) * erf(norm(r) / sqrt(T(2)) / psp.rloc)
        + exp(-rrsq / 2) * (cloc[1] + cloc[2] * rrsq + cloc[3] * rrsq^2 + cloc[4] * rrsq^3)
    )
end


"""
    eval_psp_projection_radial(psp::PspHgh, i, l, qsq::Number)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
at the reciprocal lattice point with modulus squared `qsq`.

p(qsq) = ∫_{R+} r^2 p(r) j_l(q r) dr

[HGH98] (7-15) except they do it with plane waves normalized by 1/sqrt(Ω).
"""
function eval_psp_projection_radial(psp::PspHgh, i, l, qsq::T) where {T <: Real}
    rp = psp.rp[l + 1]
    q = sqrt.(qsq)
    qrsq::T = qsq .* rp^2
    common::T = 4T(π)^(5 / 4) * sqrt(2^(l + 1) * rp^(2 * l + 3)) * exp.(-qrsq / 2)

    if l == 0
        if i == 1 return @. common end
        # Note: In the next case the HGH paper has an error.
        #       The first 8 in equation (8) should not be under the sqrt-sign
        #       This is the right version (as shown in the GTH paper)
        if i == 2 return @. common *    2     / sqrt(T(15))  * (3  -   qrsq         ) end
        if i == 3 return @. common * (4/T(3)) / sqrt(T(105)) * (15 - 10qrsq + qrsq^2) end
    end

    if l == 1  # verify expressions
        if i == 1 return @. common * 1     /    sqrt(T(3)) * q end
        if i == 2 return @. common * 2     /    sqrt(T(105)) * q * ( 5 -   qrsq         ) end
        if i == 3 return @. common * 4 / T(3) / sqrt(T(1155)) * q * (35 - 14qrsq + qrsq^2) end
    end

    error("Did not implement case of i == $i and l == $l")
end


"""
    eval_psp_energy_correction([T=Float64,] psp, n_electrons)

Evaluate the energy correction to the Ewald electrostatic interaction energy of one unit
cell, which is required compared the Ewald expression for point-like nuclei. `n_electrons`
is the number of electrons per unit cell. This defines the uniform compensating background
charge, which is assumed here.

Notice: The returned result is the *energy per unit cell* and not the energy per volume.
To obtain the latter, the caller needs to divide by the unit cell volume.
"""
function eval_psp_energy_correction(T, psp::PspHgh, n_electrons)
    # By construction we need to compute the DC component of the difference
    # of the Coulomb potential (-Z/G^2 in Fourier space) and the pseudopotential
    # i.e. -Z/(ΔG)^2 -  eval_psp_local_fourier(psp, ΔG) for ΔG → 0. This is:
    difference_DC = psp.Zion * psp.rloc^2 / 2 + sqrt(T(π)/2) * psp.rloc^3 * (
        psp.cloc[1] + 3 * psp.cloc[2] + 15 * psp.cloc[3] + 105 * psp.cloc[4]
    )

    # Multiply by number of electrons and 4π (spherical Hankel prefactor)
    # to get energy per unit cell
    4T(π) * n_electrons * difference_DC
end
eval_psp_energy_correction(psp::PspHgh, n_electrons) =
    eval_psp_energy_correction(Float64, psp, n_electrons)
