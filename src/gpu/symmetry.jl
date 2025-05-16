using GPUArraysCore

function accumulate_over_symmetries!(ρaccu::AbstractArray, ρin::AbstractArray,
    basis::PlaneWaveBasis{T}, symmetries) where {T}
    Gs = reshape(G_vectors(basis), size(ρaccu))
    fft_size = basis.fft_size

    symm_invS = to_device(basis.architecture, [Mat3{Int}(inv(symop.S)) for symop in symmetries])
    symm_τ = to_device(basis.architecture, [symop.τ for symop in symmetries])
    n_symm = length(symmetries)

    map!(ρaccu, Gs) do G
        acc = zero(complex(T))
        # Explicit loop over indicies because AMDGPU does not support zip() in map!
        for i_symm in 1:n_symm
            invS = symm_invS[i_symm]
            τ = symm_τ[i_symm]
            idx = index_G_vectors(fft_size, invS * G)
            acc += isnothing(idx) ? zero(complex(T)) : cis2pi(-T(dot(G, τ))) * ρin[idx]
        end
        acc
    end
    ρaccu
end

function lowpass_for_symmetry!(ρ::AbstractGPUArray, basis::PlaneWaveBasis{T};
                               symmetries=basis.symmetries) where {T}
    all(isone, symmetries) && return ρ

    Gs = reshape(G_vectors(basis), size(ρ))
    fft_size = basis.fft_size
    ρtmp = similar(ρ)

    symm_S = to_device(basis.architecture, [symop.S for symop in symmetries])

    map!(ρtmp, ρ, Gs) do ρ_i, G
        acc = ρ_i
	    for S in symm_S
            idx = index_G_vectors(fft_size, S * G)
            acc *= isnothing(idx) ? zero(complex(T)) : one(complex(T))
        end
        acc
    end
    ρ .= ρtmp
    ρ
end
