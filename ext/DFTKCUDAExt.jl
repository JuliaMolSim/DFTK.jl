module DFTKCUDAExt
using CUDA
import DFTK: GPU, DispatchFunctional
using DftFunctionals
using DFTK
using Libxc

DFTK.synchronize_device(::GPU{<:CUDA.CuArray}) = CUDA.synchronize()

for fun in (:potential_terms, :kernel_terms)
    @eval function DftFunctionals.$fun(fun::DispatchFunctional,
                                       ρ::CUDA.CuMatrix{Float64}, args...)
        @assert Libxc.has_cuda()
        $fun(fun.inner, ρ, args...)
    end
end

end
