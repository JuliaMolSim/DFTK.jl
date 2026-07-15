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
For details see [Herbst, Levitt 2020](https://arxiv.org/abs/2009.01665).

By default the LdosModel is constructed using the smearing of
`basis.model.smearing` and a temperature of `min(50basis.model.temperature, 0.1)`,
but this may be changed using the `smearing` and `temperature` arguments.
"""
@kwdef struct LdosModel <: χ0Model
    smearing::Union{Nothing,Smearing.SmearingFunction} = nothing
    temperature::Union{Nothing,Float64} = nothing
end
function (χ0::LdosModel)(basis::PlaneWaveBasis{T}; eigenvalues, ψ, εF, kwargs...) where {T}
    defaults = default_smearing_temperature(basis.model)
    temperature = @something(χ0.temperature, defaults.temperature)
    smearing    = @something(χ0.smearing,    defaults.smearing)
    @debug "Mixing smearing and temperature: $smearing $temperature"

    # Catch cases without contribution
    iszero(temperature) && return nothing
    ldos = compute_ldos(εF, basis, eigenvalues, ψ; smearing, temperature)
    maximum(abs, ldos) < sqrt(eps(T)) && return nothing

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

    Gsq = norm2.(G_vectors_cart(basis))
    apply_sqrtL = identity
    if χ0.localization != identity
        sqrtL = sqrt.(χ0.localization.(r_vectors(basis)))
        apply_sqrtL = x -> sqrtL .* x
    end

    # TODO simplify apply_sqrtL
    function apply!(δρ, δV, α=1)
        loc_δV = fft(basis, apply_sqrtL(δV))
        dielectric_loc_δV = @. C0 * kTF^2 * Gsq / 4T(π) / (kTF^2 - C0 * Gsq) * loc_δV
        δρ .+= α .* apply_sqrtL(irfft(basis, dielectric_loc_δV))
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
function (χ0::Applyχ0Model)(basis; ham, eigenvalues, ψ, occupation, εF,
                            kwargs...)
    defaults = default_smearing_temperature(basis.model)
    temperature = @something(χ0.temperature, defaults.temperature)
    smearing    = @something(χ0.smearing,    defaults.smearing)
    function apply!(δρ, δV, α=1)
        χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV;
                        smearing, temperature, χ0.kwargs_apply_χ0...).δρ
        δρ .+= α .* χ0δV
    end
end
