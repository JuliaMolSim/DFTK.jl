import Base: @kwdef

# structs defining terms of a composable model for the independent-particle
# susceptibility χ0. The struct define a call operator, which does some setup
# and returns an `apply!(δρ, δV, α=1)` function to form `δρ .+= α * χ0 * δV`.

abstract type χ0Model end

@doc raw"""
Represents the LDOS-based ``χ_0`` model
```math
χ_0(r, r') = (-D_\text{loc}(r) δ(r, r') + D_\text{loc}(r) D_\text{loc}(r') / D)
```
where ``D_\text{loc}`` is the local density of states and ``D`` the density of states.
For details see Herbst, Levitt 2020 arXiv:2009.01665
"""
@kwdef struct LdosModel <: χ0Model
    adjust_temperature = IncreaseMixingTemperature()
    characteristic_length = 0  # in Bohr (0 means low-pass filtering is disabled)
                               # Good values for filtering are between 2 and 0.1 Bohr
                               # Fluctuations of the LDOS below this length are smoothened.
end
function (χ0::LdosModel)(basis; eigenvalues, ψ, εF, kwargs...)
    n_spin = basis.model.n_spin_components

    # Catch cases that will yield no contribution
    temperature = χ0.adjust_temperature(basis.model.temperature; kwargs...)
    iszero(temperature) && return nothing
    ldos = compute_ldos(εF, basis, eigenvalues, ψ; temperature)
    if maximum(abs, ldos) < eps(eltype(basis))
        return nothing
    end

    if χ0.characteristic_length > 0
        wavelength = 2π / χ0.characteristic_length
        G₀ = 2wavelength / 3
        Gslope = wavelength / 6
        f_lowpass(G) = erfc((G - G₀) / Gslope) / 2

        lowpass = [f_lowpass(norm(G)) for G in G_vectors_cart(basis)]
        ldos = G_to_r(basis, r_to_G(basis, ldos) .* lowpass)
    end

    tdos = sum(sum, ldos) * basis.dvol  # Integrate LDOS to form total DOS
    function apply!(δρ, δV, α=1)
        δεF = dot(ldos, δV) .* basis.dvol
        δρ .+= α .* (-ldos .* δV .+ ldos .* δεF ./ tdos)
    end
end

@doc raw"""
A localised dielectric model for ``χ_0``:
```math
\sqrt{L(x)} \text{IFFT} \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_{TF}^2)} \text{FFT} \sqrt{L(x)}
```
where ``C_0 = 1 - ε_r``, `L(r)` is a real-space localization function
and otherwise the same conventions are used as in [`DielectricMixing`](@ref).
"""
@kwdef struct DielectricModel <: χ0Model
    εr::Real  = 10
    kTF::Real = 0.8
    localization::Function = identity
end
function (χ0::DielectricModel)(basis; kwargs...)
    T   = eltype(basis)
    εr  = T(χ0.εr)
    kTF = T(χ0.kTF)
    C0  = 1 - εr
    iszero(C0) && return nothing  # Will yield no contribution

    Gsq = [sum(abs2, G) for G in G_vectors_cart(basis)]
    apply_sqrtL = identity
    if χ0.localization != identity
        sqrtL = sqrt.(χ0.localization.(r_vectors(basis)))
        apply_sqrtL = x -> sqrtL .* x
    end

    # TODO simplify apply_sqrtL
    function apply!(δρ, δV, α=1)
        loc_δV = r_to_G(basis, apply_sqrtL(δV))
        dielectric_loc_δV = @. C0 * kTF^2 * Gsq / 4T(π) / (kTF^2 - C0 * Gsq) * loc_δV
        δρ .+= α .* apply_sqrtL(G_to_r(basis, dielectric_loc_δV))
        δρ
    end
end

"""
Full χ0 application. All keyword arguments passed to [`apply_χ0`](@ref).
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

    function apply!(δρ, δV, α=1)
        χ0δV = apply_χ0(ham, ψ_cvg, εF, eigenvalues_cvg, δV; χ0.kwargs_apply_χ0...)
        δρ .+= α .* χ0δV
    end
end
