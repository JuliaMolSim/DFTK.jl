check_real(A::AbstractArray) = nothing
function check_real(A::AbstractArray{Complex{T}}) where T
    epsT = eps(real(eltype(A)))
    error = imag(A) ./ abs.(A)
    error[map(x -> abs(imag(x)) < 100epsT, A)] .= 0

    discrepancy = norm(error)
    if discrepancy > 1000epsT
        @warn "Large imaginary part" discrepancy
    end
end
