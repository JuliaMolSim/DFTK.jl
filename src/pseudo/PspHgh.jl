using SpecialFunctions: erf
using SpecialFunctions: gamma
using Polynomials

struct PspHgh <: NormConservingPsp
    Zion::Int                   # Ionic charge (Z - valence electrons)
    rloc::Float64               # Range of local Gaussian charge distribution
    cloc::SVector{4,Float64}    # Coefficients for the local part
    lmax::Int                   # Maximal angular momentum in the non-local part
    rp::Vector{Float64}         # Projector radius for each angular momentum
    h::Vector{Matrix{Float64}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
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


function parse_hgh_file(path; identifier=path)
    lines = readlines(path)
    description = lines[1]

    # lines[2] contains the number of electrons (and the AM channel in which they sit)
    m = match(r"^ *(([0-9]+ *)+)", lines[2])
    n_elec = [parse(Int, part) for part in split(m[1])]
    Zion = sum(n_elec)

    # lines[3] contains rloc nloc and coefficients for it
    m = match(r"^ *([-.0-9]+) +([0-9]+)( +([-.0-9]+ *)+)? *", lines[3])
    rloc = parse(Float64, m[1])
    nloc = parse(Int, m[2])

    cloc = []
    m[3] !== nothing && (cloc = [parse(Float64, part) for part in split(m[3])])
    @assert length(cloc) == nloc

    # lines[4] contains the maximal AM channel
    m = match(r"^ *([0-9]+)", lines[4])
    lmax = parse(Int, m[1]) - 1

    rp = Vector{Float64}(undef, lmax + 1)
    h = Vector{Matrix{Float64}}(undef, lmax + 1)
    cur = 5  # Current line to parse

    for l in 0:lmax
        # loop over all AM channels and extract projectors,
        # these are given in blocks like
        #
        #    0.42273813    2     5.90692831    -1.26189397
        #                                       3.25819622
        #    0.48427842    1     2.72701346
        #
        # In each such blocks, the first number is the rp, the next is the number
        # of projectors and the following numbers (including the indented continuation
        # in the next lines) is the upper triangle of the coupling matrix h for this AM.
        # Then the next block commences unindented.

        # Match the first line of a block:
        m = match(r"^ *([-.0-9]+) +([0-9]+)( +([-.0-9]+ *)+)? *", lines[cur])
        rp[l + 1] = parse(Float64, m[1])
        nproj = parse(Int, m[2])
        h[l + 1] = Matrix{Float64}(undef, nproj, nproj)

        # If there are no projectors for this AM channel nproj is zero
        # and we can just increase cur and move on to the next block.
        if nproj == 0
            cur += 1
            continue
        end

        # Else we have to parse the extra parts of the hcoeff matrix.
        # This is done here.
        hcoeff = [parse(Float64, part) for part in split(m[3])]
        for i in 1:nproj
            for j in i:nproj
                h[l + 1][j, i] = h[l + 1][i, j] = hcoeff[j - i + 1]
            end

            # Parse the next (indented) line
            cur += 1
            cur > length(lines) && break
            m = match(r"^ *(([-.0-9]+ *)+)", lines[cur])
            hcoeff = [parse(Float64, part) for part in split(m[1])]
        end
    end

    PspHgh(Zion, rloc, cloc, rp, h; identifier, description)
end

@doc raw"""
The local potential of a HGH pseudopotentials in reciprocal space
can be brought to the form ``Q(t) / (t^2 exp(t^2 / 2))``
where ``t = r_\text{loc} q`` and `Q`
is a polynomial of at most degree 8. This function returns `Q`.
"""
@inline function psp_local_polynomial(T, psp::PspHgh, t=Polynomial{T}([0, 1]))
    rloc::T = psp.rloc
    Zion::T = psp.Zion

    # The polynomial prefactor P(t) (as used inside the { ... } brackets of equation
    # (5) of the HGH98 paper)
    P = (  psp.cloc[1]
         + psp.cloc[2] * (  3 -    t^2              )
         + psp.cloc[3] * ( 15 -  10t^2 +   t^4      )
         + psp.cloc[4] * (105 - 105t^2 + 21t^4 - t^6))

    4T(π) * rloc^2 * (-Zion + sqrt(T(π) / 2) * rloc * t^2 * P)
end

# [GTH98] (6) except they do it with plane waves normalized by 1/sqrt(Ω).
function eval_psp_local_fourier(psp::PspHgh, q::T) where {T <: Real}
    t::T = q * psp.rloc
    psp_local_polynomial(T, psp, t) * exp(-t^2 / 2) / t^2
end


@doc raw"""
Estimate an upper bound for the argument `q` after which
`abs(eval_psp_local_fourier(psp, q))` is a strictly decreasing function.
"""
function qcut_psp_local(T, psp::PspHgh)
    Q = DFTK.psp_local_polynomial(T, psp)  # polynomial in t = q * rloc

    # Find the roots of the derivative polynomial:
    res = roots(Polynomial([0, 1]) * derivative(Q) - Polynomial([2, 0, 1]) * Q)
    res = T[r for r in res if abs(imag(r)) < 1e-14]
    if length(res) > 0
        maximum(res) / psp.rloc
    else
        zero(T)
    end
end


# [GTH98] (1)
function eval_psp_local_real(psp::PspHgh, r::T) where {T <: Real}
    cloc = psp.cloc
    rr = r / psp.rloc
    convert(T,
        - psp.Zion / r * erf(rr / sqrt(T(2)))
        + exp(-rr^2 / 2) * (cloc[1] + cloc[2] * rr^2 + cloc[3] * rr^4 + cloc[4] * rr^6)
    )
end


@doc raw"""
The nonlocal projectors of a HGH pseudopotentials in reciprocal space
can be brought to the form ``Q(t) exp(-t^2 / 2)`` where ``t = r_l q``
and `Q` is a polynomial. This function returns `Q`.
"""
@inline function psp_projector_polynomial(T, psp::PspHgh, i, l, t=Polynomial{T}([0, 1]))
    @assert 0 <= l <= length(psp.rp) - 1
    @assert i > 0
    rp::T = psp.rp[l + 1]
    common::T = 4T(π)^(5 / T(4)) * sqrt(T(2^(l + 1)) * rp^3)

    # Note: In the (l == 0 && i == 2) case the HGH paper has an error.
    #       The first 8 in equation (8) should not be under the sqrt-sign
    #       This is the right version (as shown in the GTH paper)
    (l == 0 && i == 1) && return convert(typeof(t), common)
    (l == 0 && i == 2) && return common * 2 /  sqrt(T(  15))       * ( 3 -   t^2      )
    (l == 0 && i == 3) && return common * 4 / 3sqrt(T( 105))       * (15 - 10t^2 + t^4)
    #
    (l == 1 && i == 1) && return common * 1 /  sqrt(T(   3)) * t
    (l == 1 && i == 2) && return common * 2 /  sqrt(T( 105)) * t   * ( 5 -   t^2)
    (l == 1 && i == 3) && return common * 4 / 3sqrt(T(1155)) * t   * (35 - 14t^2 + t^4)
    #
    (l == 2 && i == 1) && return common * 1 /  sqrt(T(  15)) * t^2
    (l == 2 && i == 2) && return common * 2 / 3sqrt(T( 105)) * t^2 * ( 7 -   t^2)
    #
    (l == 3 && i == 1) && return common * 1 /  sqrt(T( 105)) * t^3
end


@doc raw"""
Estimate an upper bound for the argument `q` after which
`eval_psp_projector_fourier(psp, q)` is a strictly decreasing function.
"""
function qcut_psp_projector(T, psp::PspHgh, i, l)
    Q = DFTK.psp_projector_polynomial(T, psp, i, l)  # polynomial in q * rp[l + 1]

    # Find the roots of the derivative polynomial:
    res = roots(derivative(Q) - Polynomial([0, 1]) * Q)
    res = T[r for r in res if abs(imag(r)) < 1e-14]
    if length(res) > 0
        maximum(res) / psp.rp[l + 1]
    else
        zero(T)
    end
end


# [HGH98] (7-15) except they do it with plane waves normalized by 1/sqrt(Ω).
function eval_psp_projector_fourier(psp::PspHgh, i, l, q::T) where {T <: Real}
    @assert 0 <= l <= length(psp.rp) - 1
    t::T = q * psp.rp[l + 1]
    psp_projector_polynomial(T, psp, i, l, t) * exp(-t^2 / 2)
end


# [HGH98] (3)
function eval_psp_projector_real(psp::PspHgh, i, l, r::T) where {T <: Real}
    @assert 0 <= l <= length(psp.rp) - 1
    @assert i > 0
    rp = T(psp.rp[l + 1])
    ired = (4i - 1) / T(2)
    sqrt(T(2)) * r^(l + 2(i - 1)) * exp(-r^2 / 2rp^2) / rp^(l + ired) / sqrt(gamma(l + ired))
end

function eval_psp_energy_correction(T, psp::PspHgh, n_electrons)
    # By construction we need to compute the DC component of the difference
    # of the Coulomb potential (-Z/G^2 in Fourier space) and the pseudopotential
    # i.e. -Z/(ΔG)^2 -  eval_psp_local_fourier(psp, ΔG) for ΔG → 0. This is:
    cloc_coeffs = T[1, 3, 15, 105]
    difference_DC = (T(psp.Zion) * T(psp.rloc)^2 / 2
                     + sqrt(T(π)/2) * T(psp.rloc)^3 * T(sum(cloc_coeffs .* psp.cloc)))

    # Multiply by number of electrons and 4π (spherical Hankel prefactor)
    # to get energy per unit cell
    4T(π) * n_electrons * difference_DC
end
