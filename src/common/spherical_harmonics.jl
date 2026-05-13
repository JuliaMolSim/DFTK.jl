# TODO There is https://github.com/cortner/SphericalHarmonics.jl,
#      which should be used, once it's ready.

"""
Returns the ``(l,m)`` real spherical harmonic ``Y_l^m(r)``. Consistent with
[Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics).
"""
function ylm_real(l::Integer, m::Integer, rvec::AbstractVector{T}) where {T}
    if l == 0 && m == 0
        return sqrt(1 / 4T(π))
    end

    r = norm(rvec)
    # Catch cases of numerically very small r
    if r <= 10 * eps(T)
        return zero(T)
    end

    solid_harmonic_real(l, m, rvec / r)
end

"""
Returns the ``(l,m)`` real solid harmonic, defined as ``R_l^m(r) = r^l Y_l^m(r)``.
The solid harmonics are homogeneous polynomials of degree l,
which makes them cleanly defined and differentiable at the origin,
unlike the spherical harmonics.
"""
function solid_harmonic_real(l::Integer, m::Integer, rvec::AbstractVector{T}) where {T}
    @assert 0 ≤ l
    @assert -l ≤ m ≤ l
    @assert length(rvec) == 3
    x, y, z = rvec

    if l == 0  # s
        (m ==  0) && return sqrt(1 / 4T(π))
    end

    if l == 1  # p
        (m == -1) && return sqrt(3 / 4T(π)) * y
        (m ==  0) && return sqrt(3 / 4T(π)) * z
        (m ==  1) && return sqrt(3 / 4T(π)) * x
    end

    if l == 2  # d
        (m == -2) && return sqrt(15 /  4T(π)) * x * y
        (m == -1) && return sqrt(15 /  4T(π)) * y * z
        (m ==  0) && return sqrt( 5 / 16T(π)) * (2z^2 - x^2 - y^2)
        (m ==  1) && return sqrt(15 /  4T(π)) * x * z
        (m ==  2) && return sqrt(15 / 16T(π)) * (x^2 - y^2)
    end

    if l == 3  # f
        (m == -3) && return sqrt( 35 / 32T(π)) * (3x^2 - y^2) * y
        (m == -2) && return sqrt(105 /  4T(π)) * x * y * z
        (m == -1) && return sqrt( 21 / 32T(π)) * y * (4z^2 - x^2 - y^2)
        (m ==  0) && return sqrt(  7 / 16T(π)) * z * (2z^2 - 3x^2 - 3y^2)
        (m ==  1) && return sqrt( 21 / 32T(π)) * x * (4z^2 - x^2 - y^2)
        (m ==  2) && return sqrt(105 / 16T(π)) * (x^2 - y^2) * z
        (m ==  3) && return sqrt( 35 / 32T(π)) * (x^2 - 3y^2) * x
    end

    throw(BoundsError()) # specific (l,m) pair not implemented
end

"""
 This function returns the Wigner D matrix for real spherical harmonics, 
 for a given l and orthogonal matrix, solving a randomized linear system.
 Such D matrix gives the decomposition of a spherical harmonic after application
 of an orthogonal matrix back into the basis of spherical harmonics.
    
                       Yₗₘ₁(Wr) = Σₘ₂ D(l,R̂)ₘ₁ₘ₂ * Yₗₘ₂(r)
"""
function wigner_d_matrix(l::Integer, Wcart::AbstractMatrix{T}) where {T}
    if l == 0 # In this case no computation is needed
        return [one(T);;]
    end
    rng = Xoshiro(1234)
    neq = 2l+2 # We need at least 2l+1 equations, we add one for numerical stability
    B = Matrix{T}(undef, 2l+1, neq)
    A = Matrix{T}(undef, 2l+1, neq)
    for n in 1:neq
        r = rand(rng, T, 3)
        r = r / norm(r)
        r0 =  Wcart * r
        for m in -l:l
            B[m+l+1, n] = ylm_real(l, m, r0)
            A[m+l+1, n] = ylm_real(l, m, r)
        end
    end
    κ = cond(A)
    @assert κ < 100.0 "The Wigner matrix computation is badly conditioned. κ(A)=$(κ)"
    B / A
end