using ChainRulesCore

# TODO rrule for r_to_G

function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, f_real::AbstractArray)
    f_fourier = r_to_G(basis, f_real)
    function r_to_G_pullback(Δf_fourier)
        ∂f_real = real(basis.opBFFT_unnormalized * Δf_fourier) * (sqrt(basis.model.unit_cell_volume) / length(basis.opFFT_unnormalized))
        return NoTangent(), @not_implemented("TODO ∂basis"), ∂f_real
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, f_fourier::AbstractArray; kwargs...)
    f_real = G_to_r(basis, f_fourier; kwargs...)
    function G_to_r_pullback(Δf_real)
        ∂f_fourier = (basis.opFFT * Δf_real) * (length(basis.opFFT) / (basis.model.unit_cell_volume))
        return NoTangent(), @not_implemented("TODO ∂basis"), ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

# add workaround rrules for mpi ?
