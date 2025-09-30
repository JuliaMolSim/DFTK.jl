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

    error("The case l = $l and m = $m is not implemented")
end

"""
 This function returns the Wigner D matrix for real spherical harmonics, 
    for a given l and symmetry operation, solving a randomized linear system.
    Such matrix gives the decomposition of a spherical harmonic after application
    of a symmetry operation back into the basis of spherical harmonics.
    
                       Yₗₘ₁(R̂r) = Σₘ₂ D(l,R̂)ₘ₁ₘ₂ * Yₗₘ₂(r)

 The lattice is needed to convert reduced symmetries to Cartesian space.
"""
function wigner_d_matrix(l::Integer, Wcart::AbstractMatrix{T}) where {T}
    D = Matrix{T}(undef, 2*l+1, 2*l+1)
    if l == 0
        return D .= 1
    end
    rng = Xoshiro(1234)
    neq = 4*(2*l+1)
    for m1 in -l:l
        b = Vector{T}(undef, neq)
        A = Matrix{T}(undef, neq, 2*l+1)
        for n in 1:neq
            r = rand(rng, T, 3)
            r = r / norm(r)
            r0 =  Wcart * r
            b[n] = DFTK.ylm_real(l, m1, r0)
            for m2 in -l:l
                A[n,m2+l+1] = DFTK.ylm_real(l, m2, r)
            end
        end
        κ = cond(A)
        @assert κ < 10.0 "The Wigner matrix computation is badly conditioned. κ(A)=$(κ)"
        D[m1+l+1,:] = A\b
    end

    return D
end