check_real(A::AbstractArray) = nothing
function check_real(A::AbstractArray{Complex{T}}) where T
    discrepancy = norm(imag(A) ./ abs.(A))
    if discrepancy > 1000 * eps(real(eltype(A)))
        @warn "Large imaginary part" discrepancy
    end
end
