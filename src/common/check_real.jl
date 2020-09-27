real_checked(A::AbstractArray) = A
function real_checked(A::AbstractArray{Complex{T}}) where {T}
    epsT = eps(real(eltype(A)))
    rtol = 1000epsT
    atol = 100epsT

    if any(abs(imag(x)) > rtol * abs(x) && abs(imag(x)) > atol for x in A)
        relerror = imag(A) ./ abs.(A)
        relerror[map(x -> abs(imag(x)) < atol, A)] .= 0
        @warn "Large imaginary part" norm(relerror)
    end

    real(A)
end

"""
Zero all elements of a Fourier space array which have no complex-conjugate partner
and may thus lead to an imaginary component in real space (after an iFFT).
"""
function clear_without_conjugate!(A::AbstractArray{T,3}) where {T<:Complex}
    fft_size = size(A)
    # For even-length grids the element with largest negative integer coordinate
    # in G_vectors has no complex-conjugated partner. In order to avoid having
    # a spurious imaginary component in real-space, we set these elements to zero
    iseven(fft_size[1]) && (A[div(end, 2)+1, :, :] .= 0)
    iseven(fft_size[2]) && (A[:, div(end, 2)+1, :] .= 0)
    iseven(fft_size[3]) && (A[:, :, div(end, 2)+1] .= 0)

    A
end
