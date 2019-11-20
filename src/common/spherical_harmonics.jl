# TODO There is https://github.com/cortner/SphericalHarmonics.jl,
#      which should be used, once it's ready.

"""
Returns the (l,m) real spherical harmonic Y_lm(r). Consistent with
https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
"""
function ylm_real(l, m, rvec::AbstractVector{T}) where T
    @assert 0 ≤ l
    @assert -l ≤ m ≤ l
    @assert length(rvec) == 3
    x, y, z = rvec
    r = norm(rvec)

    if l == 0 # s
        if m ==  0 return sqrt(1 / 4T(π)) end
    end

    # Catch cases of numerically very small r
    if r <= 10 * eps(eltype(rvec))
        return T(0.0)
    end

    if l == 1 # p
        if m == -1 return sqrt(3 / 4T(π)) * y / r end
        if m ==  0 return sqrt(3 / 4T(π)) * z / r end
        if m ==  1 return sqrt(3 / 4T(π)) * x / r end
    end

    if l == 2 # d
        if m == -2 return sqrt(15 / 4T(π)) * (x / r) * (y / r) end
        if m == -1 return sqrt(15 / 4T(π)) * (y / r) * (z / r) end
        if m ==  0 return sqrt(5 / 16T(π)) * (2z^2 - x^2 - y^2) / r^2 end
        if m ==  1 return sqrt(15 / 4T(π)) * (x / r) * (z / r) end
        if m ==  2 return sqrt(15 / 4T(π)) * (x^2 - y^2) / r^2 end
    end

    error("The case l = $l and m = $m is not implemented")
end
