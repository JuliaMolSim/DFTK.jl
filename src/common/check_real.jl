check_real(A::AbstractArray{T}) where {T <: Real} = nothing
function check_real(A::AbstractArray)
    discrepancy = norm(imag(A) ./ abs.(A))
    if discrepancy > 1000 * eps(real(eltype(A)))
        @warn "Large imaginary part" discrepancy
    end
end
