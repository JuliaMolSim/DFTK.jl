@doc raw"""
Quick and dirty Anderson implementation. Not particularly optimised.

Accelerates the iterative solution of ``f(x) = 0`` according to a
damped preconditioned scheme
```math
   xₙ₊₁ = xₙ + αₙ P⁻¹ f(xₙ)
```
Where ``f(x)`` computes the residual (e.g. `SCF(x) - x`)
Further define
   - preconditioned residual  ``Pf(x) = P⁻¹ f(x)``
   - fixed-point map          ``g(x)  = x + α Pf(x)``
where the ``α`` may vary between steps.

Finds the linear combination ``xₙ₊₁ = g(xₙ) + ∑ᵢ βᵢ (g(xᵢ) - g(xₙ))``
such that ``|Pf(xₙ) + ∑ᵢ βᵢ (Pf(xᵢ) - Pf(xₙ))|²`` is minimal.

While doing this `AndersonAcceleration` ensures that the history size (number of ``g(xᵢ)``
considered) never exceeds `m`. This value should ideally be chosen to be the maximal
value fitting in memory as we use other measures on top to take care of conditioning issues,
namely:
- We monitor the conditioning of the Anderson linear system and to drop the
  oldest entries as soon as its condition number exceeds `maxcond`.
- We follow [^CDLS21] (adaptive Anderson acceleration) and drop iterates, which do not satisfy
  ```math
      \|P⁻¹ f(xᵢ)\| < \text{errorfactor} minᵢ \|P⁻¹ f(xᵢ)\|.
  ```
  This means the best way to save memory is to reduce `errorfactor` to `1e3` or `100`,
  which reduces the effective window size.
  Note that in comparison to the adaptive damping reference implementation of [^CDLS21], we
  use ``\text{errorfactor} = 1/δ``.

[^CDLS21]: Chupin, Dupuy, Legendre, Séré. Math. Model. Num. Anal. **55**, 2785 (2021) dDOI [10.1051/m2an/2021069](https://doi.org/10.1051/m2an/2021069)
"""
@kwdef struct AndersonAcceleration
    iterates::Vector  = []    # xₙ
    residuals::Vector = []    # Pf(xₙ)
    errors::Vector    = []    # ||Pf(xₙ)||
    m::Int            = 10    # Maximal history size
    # TODO If adaptive depth anderson has proven itself, increase this default value to
    # 20 or so --- provided that we do not use too much memory if 20 steps are actually taken.
    maxcond::Real     = 1e6   # Maximal condition number for Anderson matrix
    errorfactor::Real = 1e4   # Maximal error factor for iterate to be kept
end

function Base.deleteat!(anderson::AndersonAcceleration, idx)
    deleteat!(anderson.iterates,  idx)
    deleteat!(anderson.residuals, idx)
    deleteat!(anderson.errors,    idx)
    anderson
end
function Base.popfirst!(anderson::AndersonAcceleration)
    popfirst!(anderson.iterates)
    popfirst!(anderson.residuals)
    popfirst!(anderson.errors)
    anderson
end

function Base.push!(anderson::AndersonAcceleration, xₙ, αₙ, Pfxₙ)
    push!(anderson.iterates,  vec(xₙ))
    push!(anderson.residuals, vec(Pfxₙ))
    push!(anderson.errors,    norm(Pfxₙ))
    length(anderson.iterates) > anderson.m && popfirst!(anderson)
    @debug "Anderson depth: $(length(anderson.iterates))"
    @assert length(anderson.iterates) <= anderson.m
    @assert length(anderson.iterates) == length(anderson.residuals)
    @assert length(anderson.iterates) == length(anderson.errors)
    anderson
end

"""
Accelerate the fixed-point scheme
```math
   xₙ₊₁ = xₙ + αₙ P⁻¹ f(xₙ)
```
using Anderson acceleration. Requires `Pfxₙ` is ``P⁻¹ f(xₙ)``, ``xₙ`` and ``αₙ``
and returns ``xₙ₊₁``.
"""
@timing "Anderson acceleration" function (anderson::AndersonAcceleration)(xₙ, αₙ, Pfxₙ)
    if anderson.m == 0 || anderson.errorfactor ≤ 1 || anderson.maxcond ≤ 1
        return xₙ .+ αₙ .* Pfxₙ  # Disables Anderson
    end

    # Adaptive damping Anderson: Ensure δ |Pfxᵢ| ≤ minᵢ |Pfxᵢ|
    min_error = minimum(anderson.errors; init=norm(Pfxₙ))
    dropindices = findall(anderson.errors .> anderson.errorfactor * min_error)
    deleteat!(anderson, dropindices)
    if isempty(anderson.iterates)
        push!(anderson, xₙ, αₙ, Pfxₙ)
        return xₙ .+ αₙ .* Pfxₙ
    end

    # Actual acceleration, keeping an eye on maxcond
    xs   = anderson.iterates
    Pfxs = anderson.residuals

    M = hcat(Pfxs...) .- vec(Pfxₙ)  # Mᵢⱼ = (Pfxⱼ)ᵢ - (Pfxₙ)ᵢ
    # We need to solve 0 = M' Pfxₙ + M'M βs <=> βs = - (M'M)⁻¹ M' Pfxₙ

    # Ensure the condition number of M stays below maxcond, else prune the history
    # TODO This is too be tested, but in theory the adaptive-depth DIIS mechanism
    #      we implement, should ensure the condition number to stay bounded as well.
    Mfac = qr(M)
    while size(M, 2) > 1 && cond(Mfac.R) > anderson.maxcond
        M = M[:, 2:end]  # Drop oldest entry in history
        popfirst!(anderson)
        Mfac = qr(M)
    end

    xₙ₊₁ = vec(xₙ) .+ αₙ .* vec(Pfxₙ)
    βs   = -(Mfac \ vec(Pfxₙ))
    βs = to_cpu(βs)  # GPU computation only : get βs back on the CPU so we can iterate through it
    for (iβ, β) in enumerate(βs)
        xₙ₊₁ .+= β .* (xs[iβ] .- vec(xₙ) .+ αₙ .* (Pfxs[iβ] .- vec(Pfxₙ)))
    end

    push!(anderson, xₙ, αₙ, Pfxₙ)
    reshape(xₙ₊₁, size(xₙ))
end
