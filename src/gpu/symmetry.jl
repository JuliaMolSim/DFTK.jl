using GPUArraysCore

function accumulate_over_symmetries!(ρaccu::AbstractGPUArray, ρin::AbstractGPUArray,
                                     basis::PlaneWaveBasis{T}, symmetries) where {T}
    if all(isone, symmetries)
        ρaccu .+= ρin
        return ρaccu
    end

    Gs = vec(G_vectors(basis))
    ρtmp = similar(ρaccu)
    fft_size = basis.fft_size

    for symop in symmetries
        if isone(symop)
            ρaccu .+= ρin
            continue
        end

        invS = Mat3{Int}(inv(symop.S))
        map!(ρtmp, Gs) do G
            idx = index_G_vectors(fft_size, invS * G)
            isnothing(idx) ? zero(complex(T)) : cis2pi(-T(dot(G, symop.τ))) * ρin[idx]
        end
        ρaccu .+= ρtmp
    end
    ρaccu
end
        
function lowpass_for_symmetry!(ρ::AbstractGPUArray, basis::PlaneWaveBasis{T};
                               symmetries=basis.symmetries) where {T}
    all(isone, symmetries) && return ρ

    Gs = vec(G_vectors(basis))
    fft_size = basis.fft_size
    ρtmp = similar(ρ)

    for symop in symmetries
        isone(symop) && continue
        map!(ρtmp, ρ, Gs) do ρ_i, G
            idx = index_G_vectors(fft_size, symop.S * G)
            isnothing(idx) ? zero(complex(T)) : ρ_i
        end
        ρ .= ρtmp
    end
    ρ
end