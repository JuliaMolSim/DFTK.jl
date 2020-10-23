import Base: @kwdef

# structs defining terms of a composable model for the independent-particle
# susceptibility χ0. The struct define a call operator, which does some setup
# and returns an `apply!(δρ, δV)` function to add the χ0 term to a preallocated
# array `δρ`. `δV[1]` holds the spin-up (or total) potential change, `δV[2]` the
# spin-down potential as a `RealFourierArray`. `δρ[:, :, :, 1]` is the spin-up
# or the total density change and `δρ[:, :, :, 2]` the spin-down density change.

abstract type χ0Model end

@doc raw"""
Represents the LDOS-based ``χ_0`` model
```math
χ_0(r, r') = (-D_\text{loc}(r) δ(r, r') + D_\text{loc}(r) D_\text{loc}(r') / D)
```
where ``D_\text{loc}`` is the local density of states and ``D`` the density of states.
For details see Herbst, Levitt 2020 arXiv:2009.01665
"""
struct LdosModel <: χ0Model end
function (::LdosModel)(basis; eigenvalues, ψ, εF, kwargs...)
    n_spin = basis.model.n_spin_components
    dVol   = basis.model.unit_cell_volume / prod(basis.fft_size)

    # Catch cases that will yield no contribution
    iszero(basis.model.temperature) && return nothing
    ldos = [LDOS(εF, basis, eigenvalues, ψ, spins=[σ]) for σ in 1:n_spin]
    if maximum(maximum(abs, ldos[σ]) for σ in 1:n_spin) < eps(eltype(basis))
        return nothing
    end

    dos = sum(sum, ldos) * dVol  # Integrate LDOS to form total DOS
    function apply!(δρ, δV)
        dotldosδV = sum(dot(ldos[σ], δV[σ].real) for σ = 1:n_spin)
        for σ in 1:n_spin
            δρ[:, :, :, σ] .-= (-ldos[σ] .* δV[σ].real
                                .+ ldos[σ] .* dotldosδV .* dVol ./ dos)
        end
        δρ
    end
end

@doc raw"""
A localised dielectric model for ``χ_0``:
```math
\sqrt{L(x)} \text{IFFT} \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_{TF}^2)} \text{FFT} \sqrt{L(x)}
```
where ``C_0 = 1 - ε_r``, `L(r)` is a real-space localisation function
and otherwise the same conventions are used as in [`DielectricMixing`](@ref).
"""
@kwdef struct DielectricModel <: χ0Model
    εr::Real  = 10
    kTF::Real = 0.8
    localisation::Function = identity
end
function (χ0::DielectricModel)(basis; kwargs...)
    T   = eltype(basis)
    εr  = T(χ0.εr)
    kTF = T(χ0.kTF)
    C0  = 1 - εr
    n_spin = basis.model.n_spin_components
    iszero(C0) && return nothing  # Will yield no contribution

    Gsq = [sum(abs2, G) for G in G_vectors_cart(basis)]
    apply_sqrtL = identity
    if χ0.localisation != identity
        sqrtL = sqrt.(χ0.localisation.(r_vectors(basis)))
        apply_sqrtL = x -> from_real(basis, sqrtL .* x.real)
    end

    function apply!(δρ, δV)
        for σ in 1:n_spin
            loc_δV = apply_sqrtL(δV[σ]).fourier
            dielectric_loc_δV = @. C0 * kTF^2 * Gsq / 4T(π) / (kTF^2 - C0 * Gsq) * loc_δV
            δρ[:, :, :, σ] .-= apply_sqrtL(from_fourier(basis, dielectric_loc_δV)).real
        end
        δρ
    end
end

"""
Full χ0 application, optionally dropping terms or disabling Sternheimer.
All keyword arguments passed to [`apply_χ0`](@ref).
"""
struct Applyχ0Model <: χ0Model
    kwargs_apply_χ0
end
Applyχ0Model(; kwargs_apply_χ0...) = Applyχ0Model(kwargs_apply_χ0)
function (χ0::Applyχ0Model)(basis; ham, eigenvalues, ψ, εF, n_ep_extra, kwargs...)
    # self_consistent_field uses a few extra bands, which are not converged by the eigensolver
    # For the χ0 application, bands need to be perfectly converged, so we discard them here
    ψ_cvg = [@view ψk[:, 1:end-n_ep_extra]  for ψk in ψ]
    eigenvalues_cvg = [εk[1:end-n_ep_extra] for εk in eigenvalues]

    function apply!(δρ, δV)
        # χ0δV[1] is total, χ0δV[2] is spin
        χ0δV = apply_χ0(ham, ψ_cvg, εF, eigenvalues_cvg, δV...; χ0.kwargs_apply_χ0...)
        if basis.model.n_spin_components == 1
            δρ[:, :, :, 1] .+= χ0δV[1].real
        else
            δρ[:, :, :, 1] .+= (χ0δV[1].real .+ χ0δV[2].real) ./ 2
            δρ[:, :, :, 2] .+= (χ0δV[1].real .- χ0δV[2].real) ./ 2
        end
        δρ
    end
end
