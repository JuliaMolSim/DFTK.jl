using LinearAlgebra

# https://github.com/JuliaGPU/CUDA.jl/issues/1572
function LinearAlgebra.eigen(A::Hermitian{T,AT}) where {T<:Complex,AT<:CUDA.CuArray}
    vals, vects = CUDA.CUSOLVER.heevd!('V', 'U', A.data)
    (vectors = vects, values = vals)
end

function LinearAlgebra.eigen(A::Hermitian{T,AT}) where {T<:Real,AT<:CUDA.CuArray}
    vals, vects = CUDA.CUSOLVER.syevd!('V', 'U', A.data)
    (vectors = vects, values = vals)
end
