synchronize_device(::GPU{<:AMDGPU.ROCArray}) = AMDGPU.Device.sync_workgroup()
