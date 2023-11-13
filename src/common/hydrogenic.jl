@doc raw"""
Radial functions from solutions of Hydrogenic Schrödinger equation.
Same as Wannier90 user guide Table 3.3.
# Arguments
- `r`: radial grid
- `n`: principal quantum number
- `α`: diffusivity, ``\frac{Z}/{a}`` where ``Z`` is the atomic number and
    ``a`` is the Bohr radius.
"""
function radial_hydrogenic(r::AbstractVector{T}, n::Integer, α::Real=1.0) where {T<:Real}
    @assert n > 0
    if n == 1
        f = 2 * α^(3/2) * exp.(-α * r)
    elseif n == 2
        f = 2^(-3/2) * α^(3/2) * (2 .- α * r) .* exp.(-α * r/2)
    elseif n == 3
        f = sqrt(4/27) * α^(3/2) * (1 .- 2/3 * α * r .+ 2/27 * α^2 * r.^2) .* exp.(-α * r/3)
    else
        error("n = $n is not supported")
    end
    f
end