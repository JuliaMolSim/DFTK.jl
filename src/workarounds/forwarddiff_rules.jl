import ForwardDiff
import ForwardDiff: Dual
import AbstractFFTs

# original PR by mcabbott: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495

ForwardDiff.value(x::Complex{<:Dual}) = Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:Dual{T,V,N}}}) where {T,V,N} = N

ForwardDiff.tagtype(x::Complex{<:Dual{T,V,N}}) where {T,V,N} = T
ForwardDiff.tagtype(::Type{<:Complex{<:Dual{T,V,N}}}) where {T,V,N} = T

AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin
        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:Dual}}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region; kwargs...)
    end
end

function LinearAlgebra.mul!(y::AbstractArray{<:Union{Complex{<:Dual}}},
                            p::AbstractFFTs.Plan,
                            x::AbstractArray{<:Union{Complex{<:Dual}}})
    copyto!(y, p*x)
end
function Base.:*(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:Dual{Tg}}}) where {Tg}
    # TODO do we want x::AbstractArray{<:Dual{T}} too?
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    map(xtil, dxtils...) do val, parts...
        Complex(
            Dual{Tg}(real(val), map(real, parts)),
            Dual{Tg}(imag(val), map(imag, parts)),
        )
    end
end

function build_fft_plans!(tmp::AbstractArray{Complex{T}}) where {T<:Dual}
    opFFT  = AbstractFFTs.plan_fft(tmp)
    opBFFT = AbstractFFTs.plan_bfft(tmp)
    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)
    ipFFT, opFFT, ipBFFT, opBFFT
end

# Convert and strip off duals if that's the only way
function convert_dual(::Type{T}, x::Dual) where {T}
    convert(T, ForwardDiff.value(x))
end
convert_dual(::Type{T}, x::Dual) where {T <: Dual} = convert(T, x)
convert_dual(::Type{T}, x) where {T} = convert(T, x)


# DFTK setup specific
default_primes(T::Type{<:Dual}) = default_primes(ForwardDiff.valtype(T))
function next_working_fft_size(T::Type{<:Dual}, size::Integer)
    next_working_fft_size(ForwardDiff.valtype(T), size)
end

next_working_fft_size(::Type{<:Dual}, size::Int) = size

# determine symmetry operations only from primal lattice values
function symmetry_operations(lattice::AbstractMatrix{<:Dual},
                             atoms, positions, magnetic_moments=[]; kwargs...)
    positions_value = [ForwardDiff.value.(pos) for pos in positions]
    symmetry_operations(ForwardDiff.value.(lattice), atoms, positions_value,
                        magnetic_moments; kwargs...)
end

function _is_well_conditioned(A::AbstractArray{<:Dual}; kwargs...)
    _is_well_conditioned(ForwardDiff.value.(A); kwargs...)
end

value_type(T::Type{<:Dual}) = value_type(ForwardDiff.valtype(T))

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
function construct_value(model::Model{T}) where {T <: Dual}
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
construct_value(el::ElementPsp) = ElementPsp(el.Z, el.symbol, el.mass, construct_value(el.psp))
construct_value(psp::PspHgh) = psp
function construct_value(psp::PspHgh{T}) where {T <: Dual}
    PspHgh(psp.Zion,
           ForwardDiff.value(psp.rloc),
           ForwardDiff.value.(psp.cloc),
           psp.lmax,
           ForwardDiff.value.(psp.rp),
           [ForwardDiff.value.(hl) for hl in psp.h],
           psp.identifier,
           psp.description)
end


function construct_value(basis::PlaneWaveBasis{T}) where {T <: Dual}
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
                               kwargs...) where {T <: Dual}
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
        solve_ΩplusK_split(scfres, -δHextψ; tol=last(scfres.history_Δρ), response.verbose)
    end

    ## Convert and combine
    DT = Dual{ForwardDiff.tagtype(T)}
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
       scfres.algorithm, scfres.runtime_ns)
end

function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer, p::TT) where {TT <: ForwardDiff.Dual}
    # This custom rule uses two properties of the hankel transform:
    #   d H[f] / dp = 4\pi \int_0^∞ r^2 f(r) j_l'(p⋅r)⋅r dr
    # and that
    #   j_l'(x) = l / x * j_l(x) - j_{l+1}(x)
    # and tries to avoid allocations as much as possible, which hurt in this inner loop.
    #
    # One could implement this by custom rules in integration and spherical bessels, but
    # the tricky bit is to exploit that one needs both the j_l'(p⋅r) and j_l(p⋅r) values
    # but one does not want to precompute and allocate them into arrays
    # TODO Investigate custom rules for bessels and integration

    T  = ForwardDiff.valtype(TT)
    pv = ForwardDiff.value(p)

    jl = sphericalbesselj_fast.(l, pv .* r)
    value = 4T(π) * simpson((i, r) -> r2_f[i] * jl[i], r)

    if iszero(pv)
        return TT(value, zero(T) * ForwardDiff.partials(p))
    end
    derivative = 4T(π) * simpson(r) do i, r
        (r2_f[i] * (l * jl[i] / pv - r * sphericalbesselj_fast(l+1, pv * r)))
    end
    TT(value, derivative * ForwardDiff.partials(p))
end

# other workarounds

# problem: ForwardDiff of norm of SVector gives NaN derivative at zero
# https://github.com/JuliaMolSim/DFTK.jl/issues/443#issuecomment-864930410
# solution: follow ChainRules custom frule for norm
# https://github.com/JuliaDiff/ChainRules.jl/blob/52a0eeadf8d19bff491f224517b7b064ce1ba378/src/rulesets/LinearAlgebra/norm.jl#L5
# TODO delete, once forward diff AD tools use ChainRules natively
function LinearAlgebra.norm(x::SVector{S,<:Dual{Tg,T,N}}) where {S,Tg,T,N}
    x_value = ForwardDiff.value.(x)
    y = norm(x_value)
    dy = ntuple(j->real(dot(x_value, ForwardDiff.partials.(x,j))) * pinv(y), N)
    Dual{Tg}(y, dy)
end

# problem: the derivative of 1/(1+exp(x)) = -exp(x) / (1+exp(x))^2.
# When x is too large, exp(x) = Inf and this is a NaN.
function Smearing.occupation(S::Smearing.FermiDirac, d::Dual{T}) where {T}
    x = ForwardDiff.value(d)
    if exp(x) > floatmax(typeof(x)) / 1e3
        ∂occ = -zero(x)
    else
        ∂occ = -exp(x) / (1 + exp(x))^2
    end
    Dual{T}(Smearing.occupation(S, x), ∂occ * ForwardDiff.partials(d))
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.:^(x::Complex{Dual{T,V,N}}, y::Complex{Dual{T,V,N}}) where {T,V,N}
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
    complex(Dual{T,V,N}(real(expv), ForwardDiff.Partials{N,V}(tuple(real(dexpv)...))),
            Dual{T,V,N}(imag(expv), ForwardDiff.Partials{N,V}(tuple(imag(dexpv)...))))
end

# Fix for https://github.com/JuliaDiff/ForwardDiff.jl/issues/514
function Base.exp(x::Complex{Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))

    expv = exp(xx)
    dexpv = expv * dx
    complex(Dual{T,V,N}(real(expv), ForwardDiff.Partials{N,V}(tuple(real(dexpv)...))),
            Dual{T,V,N}(imag(expv), ForwardDiff.Partials{N,V}(tuple(imag(dexpv)...))))
end

# Waiting for https://github.com/JuliaDiff/DiffRules.jl/pull/37 to be merged
function erfc(x::Complex{Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))
    dgamma = -2*exp(-xx^2)/sqrt(V(π)) * dx
    complex(Dual{T,V,N}(real(erfc(xx)), ForwardDiff.Partials{N,V}(tuple(real(dgamma)...))),
            Dual{T,V,N}(imag(erfc(xx)), ForwardDiff.Partials{N,V}(tuple(imag(dgamma)...))))
end
