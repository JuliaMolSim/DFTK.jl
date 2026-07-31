@doc raw"""
Agnostic Accelerator framework to work with Anderson & PCDIIS

To be used together with scf_accelerated_solver() in scf_solvers.jl

The history management is shared, while the function accelerate() dispatches
to the different kinds of acceleration.
"""

#struct to hold the memory of the Accelerator 
#It also includes the variables needed to determine which iterates to keep 
#and which to drop
@kwdef struct AccelerationMemory
    iterates::Vector        = []
    residuals::Vector       = []
    errors::Vector          = []
    depth::Int              
    errorfactor::Float64    
    maxcond::Float64        
end

Base.length(mem::AccelerationMemory)  = length(mem.iterates)
Base.isempty(mem::AccelerationMemory) = isempty(mem.iterates)

function Base.deleteat!(mem::AccelerationMemory, idx) 
    deleteat!(mem.iterates,  idx)
    deleteat!(mem.residuals, idx)
    deleteat!(mem.errors,    idx)
    mem
end

function Base.popfirst!(mem::AccelerationMemory) 
    popfirst!(mem.iterates)
    popfirst!(mem.residuals)
    popfirst!(mem.errors)
    mem
end

function Base.push!(mem::AccelerationMemory, xₙ, rₙ, eₙ) 
    push!(mem.iterates, xₙ)
    push!(mem.residuals, rₙ)
    push!(mem.errors, eₙ)
    length(mem) > mem.depth && popfirst!(mem)
    @debug "Accelerator depth: $(length(mem.iterates))"
    @assert length(mem) <= mem.depth
    @assert length(mem) == length(mem.residuals) == length(mem.errors)
    mem
end

function adapt_memory(mem::AccelerationMemory)
    min_error = minimum(mem.errors)
    dropindices = findall(mem.errors[1:end-1] .> mem.errorfactor * min_error)
    if !isempty(dropindices)
        @debug "Accelerator adaptive depth: Deleting $dropindices"
        deleteat!(mem, dropindices)
    end
end

function solve_for_βs(M, rhs, mem::AccelerationMemory; droplines=true)
    # Ensure the condition number of M stays below maxcond, else prune the history
    Mfac = qr(M)
    while size(M, 2) > 1 && cond(Mfac.R) > mem.maxcond
        # Drop the entry with largest error, but keep the (n-1)-st entry in any case.
        error_max, idrop = findmax(mem.errors[1:end-1])
        @debug "Accelerator cond(M) = $(cond(Mfac.R)): Dropping $idrop, error=$error_max"
        deleteat!(mem, idrop)
        kept_cols = collect(1:size(M, 2))
        deleteat!(kept_cols, idrop)
        #This is ugly, implements different pruning for Anderson and PCDIIS.
        #Default is PCDIIS
        if droplines
            M = @view M[kept_cols, kept_cols]
            rhs = @view rhs[kept_cols]
        else
            M = @view M[:, kept_cols]
        end
        Mfac = qr(M)
    end
    - (Mfac \ rhs)
end

#Types for dispatch logic
abstract type AccelerationType end
struct PcdiisType   <: AccelerationType end
struct AndersonType <: AccelerationType end

struct Acceleration{T<:AccelerationType}
    type::T
    memory::AccelerationMemory
    specifics::Dict{Symbol,Any} 
end

function Acceleration(a_type::AccelerationType; depth::Integer=10, errorfactor::Real=1e5, maxcond::Real=1e6, kwargs...)
    memory = AccelerationMemory(; depth, errorfactor, maxcond)
    specifics = init_specifics(a_type; kwargs...)
    Acceleration(a_type, memory, specifics)
end

@timing "Acceleration" function (acc::Acceleration)(xₙ::StateType, fxₙ::StateType, info)
    accelerate(acc, xₙ, fxₙ, info)
end

#####Dispatch to PCDIIS##############
 
@doc raw"""
Rudimentary PCDIIS implementation

PCDIIS stands for Projected-Commutator-DIIS and is an attempt to make CDIIS feasible for large basis sets.
Instead of computing e = [H,ρ] in the full basis, this is only done in a subspace: ē = <ψ_ref|e|ψ_ref>
Instead of mixing full density or Fock matrices, 
the occupied states are mixed after gauge fixing (gf): ψ_gf = |ψ_occ><ψ_occ|ψ_ref_occ>

[^HLY17]: Hu, Lin, Yang. Journal of chemical theory and computation **13.11**, 5458-5467 (2017) DOI [10.1021/acs.jctc.7b00892](https://doi.org/10.1021/acs.jctc.7b00892) 
"""
 
init_specifics(::PcdiisType; ψ_ref=nothing, kwargs...) = 
    Dict{Symbol,Any}(:ψ_ref => ψ_ref)

function accelerate(acc::Acceleration{PcdiisType}, xₙ::OrbitalType, fxₙ::OrbitalType, info)
    if acc.memory.depth == 0 || acc.memory.errorfactor ≤ 1 || acc.memory.maxcond ≤ 1 || isnothing(xₙ.ψ) || isnothing(xₙ.occupation)
        return fxₙ
    end

    if isnothing(acc.specifics[:ψ_ref]) 
        acc.specifics[:ψ_ref] = [deepcopy(ψk[:, 1:size(ψk,2)-3]) for ψk in xₙ.ψ]
        return fxₙ
    end

    ψ_ref = acc.specifics[:ψ_ref]

    mask = xₙ.occupation[1] .> 0
	old_length = length(mask)
	mask_ref = deepcopy(mask)
	resize!(mask_ref, size(ψ_ref[1],2))
	mask_ref[old_length+1:end] .= false

    #compute iterates and residuals from input
	k_errors::Vector = []
	k_states::Vector = []

	for ik in 1:length(fxₙ.ψ)
        #compute gauge-fixed states (iterate)
		push!(k_states, fxₙ.ψ[ik][:,mask] * (fxₙ.ψ[ik][:,mask]' * ψ_ref[ik][:,mask_ref]))

        #Compute and save commutator (residual)
		ψ_ref_H_ψ_old = ψ_ref[ik]' * (info.ham.blocks[ik] * xₙ.ψ[ik][:,mask])
		ψ_old_ψ_ref   = xₙ.ψ[ik][:,mask]' * ψ_ref[ik]
		C = ψ_ref_H_ψ_old * ψ_old_ψ_ref - ψ_old_ψ_ref' * ψ_ref_H_ψ_old'
		push!(k_errors, C)
	end

    #Fill history and delete iterates with error > min(errors) * errorfactor
    push!(acc.memory, k_states, k_errors, sum([norm(C)^2 for C in k_errors]))
    if length(acc.memory) < 2
        return fxₙ
    end

    adapt_memory(acc.memory)

    #build matrix Mij & vector bj
    Rs = acc.memory.residuals
    M = Vector{Any}()
    b = Vector{Any}()
    Y = [Rs[i-1] .- Rs[i] for i in 2:length(Rs)] 
    n = length(Y)
	for ik in 1:length(fxₙ.ψ)
        push!(b, vec([real(dot(Y[i][ik], Rs[end][ik])) for i in 1:n]))
        m = zeros(Float64, n, n)
        for i in 1:n, j in i:n
            m[i,j] = m[j,i] =  real(dot(Y[i][ik], Y[j][ik]))
        end
        push!(M, m)
	end

    #for now only one kpoint, later need to implement proper k-weights
    βs = solve_for_βs(M[1], b[1], acc.memory)

    Φs = acc.memory.iterates
    ΔΦ = [Φs[i-1] .- Φs[i] for i in 2:length(Φs)] 
	for ik in 1:length(fxₙ.ψ)
		#mixing of pw coefficients of the gauge fixed states ψ_gf
        Φmix = copy(Φs[end][ik])
		for iβ in 1:length(βs)
            axpy!(βs[iβ], ΔΦ[iβ][ik], Φmix)
		end

		fxₙ.ψ[ik][:,mask] = Φmix
		fxₙ.ψ[ik] = Matrix(qr(fxₙ.ψ[ik]).Q)
	end

    fxₙ.ρ = compute_density(info.basis, fxₙ.ψ, fxₙ.occupation)
    fxₙ
end

##### Dispatch to Anderson ##############
 
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
- We follow [^CDLS21] (adaptive Anderson acceleration) and drop iterates, which do not satisfy
  ```math
      \|P⁻¹ f(xᵢ)\| < \text{errorfactor} minᵢ \|P⁻¹ f(xᵢ)\|.
  ```
  This means the best way to save memory is to reduce `errorfactor` to `1e3` or `100`,
  which reduces the effective window size.
  Note that in comparison to the adaptive depth reference implementation of [^CDLS21], we
  use ``\text{errorfactor} = 1/δ``.
- Additionally we monitor the conditioning of the Anderson linear system and if it exceeds
  `maxcond` we drop the entries with largest residual norm ``\|P⁻¹ f(xᵢ)\|``
  (but never the most recent, i.e. ``n-1``-st, iterate).

[^CDLS21]: Chupin, Dupuy, Legendre, Séré. Math. Model. Num. Anal. **55**, 2785 (2021) dDOI [10.1051/m2an/2021069](https://doi.org/10.1051/m2an/2021069)
"""

init_specifics(::AndersonType; α::Real=1.0, kwargs...) =
    Dict{Symbol,Any}(:α => α)

function accelerate(acc::Acceleration{AndersonType}, xₙ::DensityType, fxₙ::DensityType, info)
    α = acc.specifics[:α]
    if acc.memory.depth == 0 || acc.memory.errorfactor ≤ 1 || acc.memory.maxcond ≤ 1
        return xₙ + α * (fxₙ - xₙ) # Disables Anderson
    elseif length(acc.memory) < 1
        Pfxₙ = fxₙ-xₙ
        push!(acc.memory, vec(xₙ), vec(Pfxₙ), norm(Pfxₙ))
        return  xₙ + α * (fxₙ - xₙ)
    end
    
    Pfxₙ = vec(fxₙ-xₙ)
    xₙ = vec(xₙ)

    # Adaptive depth Anderson: Ensure |Pfxᵢ| ≤ errorfactor * minᵢ |Pfxᵢ|, but keep
    # (n-1)-st entry in any case.
    adapt_memory(acc.memory)

    Pfxs = acc.memory.residuals
    xs = acc.memory.iterates
    
    M = stack(Pfxs) .- Pfxₙ  # Mᵢⱼ = (Pfxⱼ)ᵢ - (Pfxₙ)ᵢ
    # We need to solve 0 = M' Pfxₙ + M'M βs <=> βs = - (M'M)⁻¹ M' Pfxₙ
     
    # TODO If memory pressure is high, automatically drop old iterates here
    #      (Note: This quickly happens in GPU scenarios)
    βs = solve_for_βs(M, Pfxₙ, acc.memory, droplines=false)
    βs = to_cpu(βs)  # GPU computation only : get βs back on the CPU so we can iterate through it

    xₙ₊₁ = xₙ .+ α .* Pfxₙ
    
    for (iβ, β) in enumerate(βs)
        xₙ₊₁ .+= β .* (xs[iβ] .- xₙ .+ α .* (Pfxs[iβ] .- Pfxₙ))
    end

    push!(acc.memory, xₙ, Pfxₙ, norm(Pfxₙ))
    fxₙ.ρ = reshape(xₙ₊₁, size(fxₙ.ρ))
    return fxₙ
end
