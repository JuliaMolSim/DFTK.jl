# TODO There is https://github.com/cortner/SphericalHarmonics.jl,
#      which should be used, once it's ready.

"""
Returns the ``(l,m)`` real spherical harmonic ``Y_l^m(r)``. Consistent with
[Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics).
"""
function ylm_real(l::Integer, m::Integer, rvec::AbstractVector{T}) where {T}
    @assert 0 ≤ l
    @assert -l ≤ m ≤ l
    @assert length(rvec) == 3
    x, y, z = rvec
    r = norm(rvec)

    if l == 0  # s
        (m ==  0) && return sqrt(1 / 4T(π))
    end

    # Catch cases of numerically very small r
    if r <= 10 * eps(eltype(rvec))
        return zero(T)
    end

    if l == 1  # p
        (m == -1) && return sqrt(3 / 4T(π)) * y / r
        (m ==  0) && return sqrt(3 / 4T(π)) * z / r
        (m ==  1) && return sqrt(3 / 4T(π)) * x / r
    end

    if l == 2  # d
        (m == -2) && return sqrt(15 /  4T(π)) * (x / r) * (y / r)
        (m == -1) && return sqrt(15 /  4T(π)) * (y / r) * (z / r)
        (m ==  0) && return sqrt( 5 / 16T(π)) * (2z^2 - x^2 - y^2) / r^2
        (m ==  1) && return sqrt(15 /  4T(π)) * (x / r) * (z / r)
        (m ==  2) && return sqrt(15 / 16T(π)) * (x^2 - y^2) / r^2
    end

    if l == 3  # f
        (m == -3) && return sqrt( 35 / 32T(π)) * (3x^2 - y^2) * y / r^3
        (m == -2) && return sqrt(105 /  4T(π)) * x * y * z / r^3
        (m == -1) && return sqrt( 21 / 32T(π)) * y * (4z^2 - x^2 - y^2) / r^3
        (m ==  0) && return sqrt(  7 / 16T(π)) * z * (2z^2 - 3x^2 - 3y^2) / r^3
        (m ==  1) && return sqrt( 21 / 32T(π)) * x * (4z^2 - x^2 - y^2) / r^3
        (m ==  2) && return sqrt(105 / 16T(π)) * (x^2 - y^2) * z / r^3
        (m ==  3) && return sqrt( 35 / 32T(π)) * (x^2 - 3y^2) * x / r^3
    end

    if l == 4
        (m == -4) && return sqrt( 315 / 16T(π)) * x * y * (x^2 - y^2) / r^4
        (m == -3) && return sqrt( 315 / 32T(π)) * y * (3x^2 - y^2) * z / r^4
        (m == -2) && return sqrt(  45 / 16T(π)) * x * y * (7z^2 - r^2) / r^4
        (m == -1) && return sqrt(  45 / 32T(π)) * y * (7z^3 - 3*z*r^2) / r^4
        (m ==  0) && return sqrt(   9 /256T(π)) * (35z^4 - 30*z^2*r^2 + 3*r^4) / r^4
        (m ==  1) && return sqrt(  45 / 32T(π)) * x * (7z^3 - 3*z*r^2) / r^4
        (m ==  2) && return sqrt(  45 / 64T(π)) * (x^2 - y^2) * (7z^2 - r^2) / r^4
        (m ==  3) && return sqrt( 315 / 32T(π)) * x * (x^2 - 3y^2) * z / r^4
        (m ==  4) && return sqrt( 315 /256T(π)) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2)) / r^4
    end

    error("The case l = $l and m = $m is not implemented")
end

function pack_lm(l::Integer, m::Integer)
    @assert 0 ≤ l
    @assert -l ≤ m ≤ l
    return l^2 + l + m + 1
end

# TODO: check this
function unpack_lm(lm::Integer)
    @assert lm ≥ 1
    l = floor(Int, sqrt(lm - 1))
    m = lm - (l^2 + l + 1)
    return l, m
end

function gaunt_coefficients(T::Type, l::Integer)
    nharm = (2l+1)^2
    # Sample points randomly on a sphere
    rng = Xoshiro(42)
    points = rand(rng, T, 3, nharm)
    points ./= sqrt.(sum(points.^2, dims=1))
    # Evaluate spherical harmonics at these points
    Y = zeros(T, nharm, nharm)
    for j in 1:nharm
        for i in 1:nharm
            Y[i, j] = ylm_real(unpack_lm(j)..., points[:, i])
        end
    end
    fact = factorize(Y)

    coefs = zeros(T, nharm, (l+1)^2, (l+1)^2)

    for j in 1:(l+1)^2
        for i in 1:(l+1)^2
            coefs[:, i, j] = fact \ (Y[:, i] .* Y[:, j])
        end
    end

    coefs
end
