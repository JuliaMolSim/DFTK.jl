synchronize_device(::GPU{<:CUDA.CuArray}) = CUDA.synchronize()

@static if @load_preference("libxc_cuda", "true") == "true"
    for fun in (:potential_terms, :kernel_terms)
        @eval function DftFunctionals.$fun(fun::DispatchFunctional,
                                           ρ::CUDA.CuMatrix{Float64}, args...)
            @assert Libxc.has_cuda()
            $fun(fun.inner, ρ, args...)
        end
    end
end

# Note: This preference is temporary for now and can disappear at any time.
function disable_libxc_cuda!()
    @set_preferences!("libxc_cuda" => "false")
end
