module DFTKAMDGPUExt
using AMDGPU
import DFTK: GPU
using DFTK

DFTK.synchronize_device(::GPU{<:AMDGPU.ROCArray}) = AMDGPU.synchronize()

end
