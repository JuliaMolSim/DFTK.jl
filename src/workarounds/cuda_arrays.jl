using LinearAlgebra

# https://github.com/JuliaGPU/CUDA.jl/issues/1572
function LinearAlgebra.eigen(A::Hermitian{<:Complex,<:CUDA.CuArray})
    values, vectors = CUDA.CUSOLVER.heevd!('V', 'U', A.data)
    (; values, vectors)
end

function LinearAlgebra.eigen(A::Hermitian{<:Real,<:CUDA.CuArray})
    values, vectors = CUDA.CUSOLVER.syevd!('V', 'U', A.data)
    (; values, vectors)
end

synchronize_device(::GPU{<:CUDA.CuArray}) = CUDA.synchronize()

for fun in (:potential_terms, :kernel_terms)
    @eval function DftFunctionals.$fun(fun::DispatchFunctional,
                                       ρ::CUDA.CuMatrix{Float64}, args...)
        @assert Libxc.has_cuda()
        $fun(fun.inner, ρ, args...)
    end
end
