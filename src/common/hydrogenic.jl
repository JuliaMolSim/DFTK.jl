@doc raw"""
Radial functions from solutions of Hydrogenic Schrödinger equation.
Same as Wannier90 user guide Table 3.3.
# Arguments
- `r`: radial grid
- `n`: principal quantum number
- `α`: diffusivity, ``\frac{Z}{a}`` where ``Z`` is the atomic number and
    ``a`` is the Bohr radius.
"""
function radial_hydrogenic(r::AbstractVector{T}, n::Integer, α::Real=one(T)) where {T<:Real}
    # T(...) is used for some literals to ensure that the output is of type T
    @assert n > 0
    if n == 1
        f = 2 * α^(3/T(2)) * exp.(-α * r)
    elseif n == 2
        f = 2^(-3/T(2)) * α^(3/T(2)) * (2 .- α * r) .* exp.(-α * r/2)
    elseif n == 3
        f = sqrt(4/T(27)) * α^(3/T(2)) * (1 .- 2/T(3) * α * r .+ 2/T(27) * α^2 * r.^2) .* exp.(-α * r/3)
    else
        error("n = $n is not supported")
    end
    f
end
