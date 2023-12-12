import ForwardDiff
import AbstractFFTs

# original PR by mcabbott: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495

ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) = Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N

ForwardDiff.tagtype(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = T
ForwardDiff.tagtype(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = T

# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin
        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region; kwargs...)
    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region; kwargs...)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin
        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region; kwargs...)
    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:ForwardDiff.Dual}) =
            (Y .= _apply_plan(p, X))

        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            (Y .= _apply_plan(p, X))
    end
end

LinearAlgebra.mul!(Y::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, X::AbstractArray{<:ComplexF64}) where {T,P} =
    (Y .= _apply_plan(p, X))

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{T}}}) where {T}
    # TODO do we want x::AbstractArray{<:ForwardDiff.Dual{T}} too?
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual{T}(real(val), map(real, parts)),
            ForwardDiff.Dual{T}(imag(val), map(imag, parts)),
        )
    end
end

function _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray) where {T,P}
    _apply_plan(p.p, p.scale * x) # for when p.scale is Dual, need out-of-place
end

# this is to avoid method ambiguities between these two:
#   _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{T}}}) where {T}
#   _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray) where {T,P}
function _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{Tg}}}) where {T,P,Tg}
    _apply_plan(p.p, p.scale * x)
end

# Convert and strip off duals if that's the only way
function convert_dual(::Type{T}, x::ForwardDiff.Dual) where {T}
    convert(T, ForwardDiff.value(x))
end
convert_dual(::Type{T}, x::ForwardDiff.Dual) where {T <: ForwardDiff.Dual} = convert(T, x)
convert_dual(::Type{T}, x) where {T} = convert(T, x)


# DFTK setup specific
default_primes(T::Type{<:ForwardDiff.Dual}) = default_primes(ForwardDiff.valtype(T))
function next_working_fft_size(T::Type{<:ForwardDiff.Dual}, size::Integer)
    next_working_fft_size(ForwardDiff.valtype(T), size)
end

next_working_fft_size(::Type{<:ForwardDiff.Dual}, size::Int) = size

function build_fft_plans!(tmp::AbstractArray{Complex{T}}) where {T<:ForwardDiff.Dual}
    opFFT  = AbstractFFTs.plan_fft(tmp)
    opBFFT = AbstractFFTs.plan_bfft(tmp)
    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)
    ipFFT, opFFT, ipBFFT, opBFFT
end

# determine symmetry operations only from primal lattice values
function symmetry_operations(lattice::AbstractMatrix{<:ForwardDiff.Dual},
                             atoms, positions, magnetic_moments=[]; kwargs...)
    positions_value = [ForwardDiff.value.(pos) for pos in positions]
    symmetry_operations(ForwardDiff.value.(lattice), atoms, positions_value,
                        magnetic_moments; kwargs...)
end

function _is_well_conditioned(A::AbstractArray{<:ForwardDiff.Dual}; kwargs...)
    _is_well_conditioned(ForwardDiff.value.(A); kwargs...)
end

value_type(T::Type{<:ForwardDiff.Dual}) = value_type(ForwardDiff.valtype(T))

# TODO Should go to Model.jl / PlaneWaveBasis.jl as a constructor.
#
# Along with it should go a nice convert function to get rid of the annoying
# conversion thing in the stress computation.
#
# Ideally also upon constructing a Model one would automatically determine that
# *some* parameter deep inside the terms, psp or sth is a dual and automatically
# convert the full thing to be based on dual numbers ... note that this requires
# exposing the element type all the way up ... which is probably needed to do these
# forward and backward conversion routines between duals and non-duals as well.
function construct_value(model::Model{T}) where {T <: ForwardDiff.Dual}
    newpositions = [ForwardDiff.value.(pos) for pos in model.positions]
    Model(ForwardDiff.value.(model.lattice),
          construct_value.(model.atoms),
          newpositions;
          model.model_name,
          model.n_electrons,
          magnetic_moments=[],  # Symmetries given explicitly
          terms=model.term_types,
          temperature=ForwardDiff.value(model.temperature),
          model.smearing,
          εF=ForwardDiff.value(model.εF),
          model.spin_polarization,
          model.symmetries,
          # Can be safely disabled: this has been checked for basis.model
          disable_electrostatics_check=true)
end

construct_value(el::Element) = el
construct_value(el::ElementPsp) = ElementPsp(el.Z, el.symbol, construct_value(el.psp))
construct_value(psp::PspHgh) = psp
function construct_value(psp::PspHgh{T}) where {T <: ForwardDiff.Dual}
    PspHgh(psp.Zion,
           ForwardDiff.value(psp.rloc),
           ForwardDiff.value.(psp.cloc),
           psp.lmax,
           ForwardDiff.value.(psp.rp),
           [ForwardDiff.value.(hl) for hl in psp.h],
           psp.identifier,
           psp.description)
end


function construct_value(basis::PlaneWaveBasis{T}) where {T <: ForwardDiff.Dual}
    # NOTE: This is a pretty slow function as it *recomputes* basically
    #       everything instead of just extracting the primal data contained
    #       already in the dualised PlaneWaveBasis data structure.
    PlaneWaveBasis(construct_value(basis.model),
                   ForwardDiff.value(basis.Ecut),
                   basis.fft_size,
                   basis.variational,
                   basis.kgrid,
                   basis.symmetries_respect_rgrid,
                   basis.use_symmetries_for_kpoint_reduction,
                   basis.comm_kpts,
                   basis.architecture)
end


function self_consistent_field(basis_dual::PlaneWaveBasis{T};
                               response=ResponseOptions(),
                               kwargs...) where {T <: ForwardDiff.Dual}
    # Note: No guarantees on this interface yet.

    # Primal pass
    basis_primal = construct_value(basis_dual)
    scfres = self_consistent_field(basis_primal; kwargs...)

    ## Compute external perturbation (contained in ham_dual) and from matvec with bands
    Hψ_dual = let
        occupation_dual = [T.(occk) for occk in scfres.occupation]
        ψ_dual = [Complex.(T.(real(ψk)), T.(imag(ψk))) for ψk in scfres.ψ]
        ρ_dual = compute_density(basis_dual, ψ_dual, occupation_dual)
        εF_dual = T(scfres.εF)  # Only needed for entropy term
        eigenvalues_dual = [T.(εk) for εk in scfres.eigenvalues]
        ham_dual = energy_hamiltonian(basis_dual, ψ_dual, occupation_dual;
                                      ρ=ρ_dual, eigenvalues=eigenvalues_dual,
                                      εF=εF_dual).ham
        ham_dual * ψ_dual
    end

    ## Implicit differentiation
    response.verbose && println("Solving response problem")
    δresults = ntuple(ForwardDiff.npartials(T)) do α
        δHextψ = [ForwardDiff.partials.(δHextψk, α) for δHextψk in Hψ_dual]
        solve_ΩplusK_split(scfres, -δHextψ; tol=scfres.norm_Δρ, response.verbose)
    end

    ## Convert and combine
    DT = ForwardDiff.Dual{ForwardDiff.tagtype(T)}
    ψ = map(scfres.ψ, getfield.(δresults, :δψ)...) do ψk, δψk...
        map(ψk, δψk...) do ψnk, δψnk...
            Complex(DT(real(ψnk), real.(δψnk)),
                    DT(imag(ψnk), imag.(δψnk)))
        end
    end
    ρ = map((ρi, δρi...) -> DT(ρi, δρi), scfres.ρ, getfield.(δresults, :δρ)...)
    eigenvalues = map(scfres.eigenvalues, getfield.(δresults, :δeigenvalues)...) do εk, δεk...
        map((εnk, δεnk...) -> DT(εnk, δεnk), εk, δεk...)
    end
    occupation = map(scfres.occupation, getfield.(δresults, :δoccupation)...) do occk, δocck...
        map((occnk, δoccnk...) -> DT(occnk, δoccnk), occk, δocck...)
    end
    εF = DT(scfres.εF, getfield.(δresults, :δεF)...)

    # TODO Could add δresults[α].δVind the dual part of the total local potential in ham_dual
    # and in this way return a ham that represents also the total change in Hamiltonian

    energies, ham = energy_hamiltonian(basis_dual, ψ, occupation; ρ, eigenvalues, εF)

    # This has to be changed whenever the scfres structure changes
    (; ham, basis=basis_dual, energies, ρ, eigenvalues, occupation, εF, ψ,
       # non-differentiable metadata:
       response=getfield.(δresults, :history),
       scfres.converged, scfres.occupation_threshold, scfres.α, scfres.n_iter,
       scfres.n_bands_converge, scfres.diagonalization, scfres.stage,
       scfres.algorithm, scfres.norm_Δρ)
end

# other workarounds

# problem: ForwardDiff of norm of SVector gives NaN derivative at zero
# https://github.com/JuliaMolSim/DFTK.jl/issues/443#issuecomment-864930410
# solution: follow ChainRules custom frule for norm
# https://github.com/JuliaDiff/ChainRules.jl/blob/52a0eeadf8d19bff491f224517b7b064ce1ba378/src/rulesets/LinearAlgebra/norm.jl#L5
# TODO delete, once forward diff AD tools use ChainRules natively
function LinearAlgebra.norm(x::SVector{S,<:ForwardDiff.Dual{Tg,T,N}}) where {S,Tg,T,N}
    x_value = ForwardDiff.value.(x)
    y = norm(x_value)
    dy = ntuple(j->real(dot(x_value, ForwardDiff.partials.(x,j))) * pinv(y), N)
    ForwardDiff.Dual{Tg}(y, dy)
end

# problem: the derivative of 1/(1+exp(x)) = -exp(x) / (1+exp(x))^2.
# When x is too large, exp(x) = Inf and this is a NaN.
function Smearing.occupation(S::Smearing.FermiDirac, d::ForwardDiff.Dual{T}) where {T}
    x = ForwardDiff.value(d)
    if exp(x) > floatmax(typeof(x)) / 1e3
        ∂occ = -zero(x)
    else
        ∂occ = -exp(x) / (1 + exp(x))^2
    end
    ForwardDiff.Dual{T}(Smearing.occupation(S, x), ∂occ * ForwardDiff.partials(d))
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.:^(x::Complex{ForwardDiff.Dual{T,V,N}}, y::Complex{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    yy = complex(ForwardDiff.value(real(y)), ForwardDiff.value(imag(y)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))
    dy = complex.(ForwardDiff.partials(real(y)), ForwardDiff.partials(imag(y)))

    expv = xx^yy
    ∂expv∂x = yy * xx^(yy-1)
    ∂expv∂y = log(xx) * expv
    dxexpv = ∂expv∂x * dx
    if iszero(xx) && ForwardDiff.isconstant(real(y)) && ForwardDiff.isconstant(imag(y)) && imag(y) === zero(imag(y)) && real(y) > 0
        dexpv = zero(expv)
    elseif iszero(xx)
        throw(DomainError(x, "mantissa cannot be zero for complex exponentiation"))
    else
        dyexpv = ∂expv∂y * dy
        dexpv = dxexpv + dyexpv
    end
    complex(ForwardDiff.Dual{T,V,N}(real(expv), ForwardDiff.Partials{N,V}(tuple(real(dexpv)...))),
            ForwardDiff.Dual{T,V,N}(imag(expv), ForwardDiff.Partials{N,V}(tuple(imag(dexpv)...))))
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.exp(x::Complex{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))

    expv = exp(xx)
    dexpv = expv * dx
    complex(ForwardDiff.Dual{T,V,N}(real(expv), ForwardDiff.Partials{N,V}(tuple(real(dexpv)...))),
            ForwardDiff.Dual{T,V,N}(imag(expv), ForwardDiff.Partials{N,V}(tuple(imag(dexpv)...))))
end

# Waiting for https://github.com/JuliaDiff/DiffRules.jl/pull/37 to be merged
function erfc(x::Complex{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))
    dgamma = -2*exp(-xx^2)/sqrt(V(π)) * dx
    complex(ForwardDiff.Dual{T,V,N}(real(erfc(xx)), ForwardDiff.Partials{N,V}(tuple(real(dgamma)...))),
            ForwardDiff.Dual{T,V,N}(imag(erfc(xx)), ForwardDiff.Partials{N,V}(tuple(imag(dgamma)...))))
end
