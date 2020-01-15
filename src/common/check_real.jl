check_real(A::AbstractArray) = nothing
function check_real(A::AbstractArray{Complex{T}}) where T
    epsT = eps(real(eltype(A)))
    rtol = 1000epsT
    atol = 100epsT

    if any(abs(imag(x)) > rtol * abs(x) && abs(imag(x)) > atol for x in A)
        relerror = imag(A) ./ abs.(A)
        relerror[map(x -> abs(imag(x)) < atol, A)] .= 0
        @warn "Large imaginary part" norm(relerror)
    end
end
