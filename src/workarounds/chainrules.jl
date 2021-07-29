using ChainRulesCore
using Zygote: @adjoint  # TODO remove, once ChainRules rrules can overrule Zygote
import AbstractFFTs

function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, f_real::AbstractArray)
    f_fourier = r_to_G(basis, f_real)
    function r_to_G_pullback(Δf_fourier)
        ∂f_real = r_to_G_normalization * real(basis.opBFFT * Δf_fourier)
        ∂normalization = sum(Δf_fourier .* f_fourier)
        ∂basis = Tangent{typeof(basis)}(;r_to_G_normalization=∂normalization)
        return NoTangent(), ∂basis, ∂f_real
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, f_fourier::AbstractArray; kwargs...)
    f_real = G_to_r(basis, f_fourier; kwargs...)
    function G_to_r_pullback(Δf_real)
        ∂f_fourier = (basis.opFFT * Δf_real) * (length(basis.opFFT) / (basis.model.unit_cell_volume))
        ∂normalization = sum(Δf_real .* f_real)
        ∂basis = Tangent{typeof(basis)}(;G_to_r_normalization=∂normalization)
        return NoTangent(), ∂basis, ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

# function ChainRulesCore.rrule(::typeof(*), P::AbstractFFTs.ScaledPlan, x)
#     y = P * x
#     function mul_pullback(Δ)
#       N = prod(size(x)[[P.p.region...]])
#       ∂x = N * (P \ Δ)
#       ∂P = Tangent{typeof(P)}(;scale=sum(Δ .* y))
#       return NoTangent(), ∂P, ∂x 
#     end
#     return y, mul_pullback
# end

# # explicit Zygote.adjoint definitions are here bc Zygote.adjoint rules 
# # take precedence over ChainRulesCore.rrule, even if rrule is more specific.
# # Currently, Zygote defines an adjoint for *(P::AbstractFFTs.Plan, xs)
# # that fails for ScaledPlan (which has no field `region`).
# # TODO fix upstream and delete here
# @adjoint function *(P::AbstractFFTs.ScaledPlan, xs)
#     @warn "custom adjoint *(P::AbstractFFTs.ScaledPlan, xs)"
#     ys = P * xs
#     return ys, function(Δ)
#         N = prod(size(xs)[[P.p.region...]])
#         return ((;scale=sum(Δ .* ys)), N * (P \ Δ)) # or dot(Δ, ys) (complex conj) ?
#     end
# end
# @adjoint function \(P::AbstractFFTs.ScaledPlan, xs)
#     @warn "custom adjoint \\(P::AbstractFFTs.ScaledPlan, xs)"
#     ys = P \ xs
#     return ys, function(Δ)
#         N = prod(size(xs)[[P.p.region...]])
#         return ((;scale=-sum(Δ .* ys)/P.scale^2), (P * Δ) / N) # TODO check scale part (complex deriv?)
#     end
# end

# workaround rrules for mpi: treat as noop
function ChainRulesCore.rrule(::typeof(mpi_sum), arr, comm)
    function mpi_sum_pullback(Δy)
        return NoTangent(), Δy, NoTangent()
    end
    return arr, mpi_sum_pullback
end

ChainRulesCore.@non_differentiable ElementPsp(::Any...)
ChainRulesCore.@non_differentiable r_vectors(::Any...)
ChainRulesCore.@non_differentiable G_vectors(::Any...)

# TODO delete
@adjoint (T::Type{<:SArray})(x...) = T(x...), y->(y,)

# TODO rrule for Model and PlaneWaveBasis constructor

# # simplified version of the Model constructor to
# # help reverse mode AD to only differentiate the relevant computations.
# # this excludes assertions (try-catch), and symmetries
function _autodiff_Model_namedtuple(lattice)
    T = eltype(lattice)
    recip_lattice = 2T(π)*inv(lattice')
    unit_cell_volume = abs(det(lattice))
    recip_cell_volume = abs(det(recip_lattice))
    (;lattice=lattice, recip_lattice=recip_lattice, unit_cell_volume=unit_cell_volume, recip_cell_volume=recip_cell_volume)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{Model}, lattice; kwargs...)
    @warn "simplified Model rrule triggered."
    model = T(lattice; kwargs...)
    _model, Model_pullback = rrule_via_ad(config, _autodiff_Model_namedtuple, lattice)
    # TODO add some assertion that model and _model agree
    return model, Model_pullback
end

# simplified version of PlaneWaveBasis constructor to
# help reverse mode AD to only differentiate the relevant computations.
# this excludes assertions (try-catch), MPI handling, and other things
function _autodiff_PlaneWaveBasis_namedtuple(model::Model{T}, basis::PlaneWaveBasis) where {T <: Real}
    dvol = model.unit_cell_volume ./ prod(basis.fft_size)
    G_to_r_normalization = 1 / sqrt(model.unit_cell_volume)
    r_to_G_normalization = sqrt(model.unit_cell_volume) / length(basis.ipFFT)

    # Create dummy terms array for _basis to handle
    terms = Vector{Any}(undef, length(model.term_types))

    # cicularity is getting complicated...
    # To correctly instantiate term types, we do need a full PlaneWaveBasis struct;
    # so we need to interleave re-computed differentiable params, and fixed params in basis
    _basis = PlaneWaveBasis{T}(
        model, basis.fft_size, dvol, 
        basis.Ecut, basis.variational,
        basis.opFFT, basis.ipFFT, basis.opBFFT, basis.ipBFFT,
        r_to_G_normalization, G_to_r_normalization,
        basis.kpoints, basis.kweights, basis.ksymops, basis.kgrid, basis.kshift,
        basis.kcoords_global, basis.ksymops_global, basis.comm_kpts, basis.krange_thisproc, basis.krange_allprocs,
        basis.symmetries, terms)

    terms = [t(_basis) for t in model.term_types]
    (;model=model, dvol=dvol, terms=terms, G_to_r_normalization=G_to_r_normalization, r_to_G_normalization=r_to_G_normalization)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{PlaneWaveBasis}, model::Model, Ecut; kwargs...)
    @warn "simplified PlaneWaveBasis rrule triggered."
    basis = T(model, Ecut; kwargs...)
    _basis, PlaneWaveBasis_pullback = rrule_via_ad(config, _autodiff_PlaneWaveBasis_namedtuple, model, basis)
    return basis, PlaneWaveBasis_pullback
end


# TODO delete (once fixed upstream in Zygote, "difftype_warn not defined")
Zygote.z2d(x::Union{AbstractZero, Tangent}, ::Any) = x

# convert generators into arrays (needed for Zygote here)
function _G_vectors_cart(basis::PlaneWaveBasis)
    [basis.model.recip_lattice * G for G in G_vectors(basis.fft_size)]
end
_G_vectors_cart(kpt::Kpoint) = [kpt.model.recip_lattice * G for G in G_vectors(kpt)]

function _autodiff_TermKinetic_namedtuple(basis; scaling_factor=1)
    kinetic_energies = [[scaling_factor * sum(abs2, G + kpt.coordinate_cart) / 2
                         for G in _G_vectors_cart(kpt)]
                        for kpt in basis.kpoints]
    (;basis=basis, kinetic_energies=kinetic_energies)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{TermKinetic}, basis::PlaneWaveBasis; kwargs...)
    @warn "simplified TermKinetic rrule triggered."
    term = T(basis; kwargs...)
    T_simple = (args...) -> _autodiff_TermKinetic_namedtuple(args...; kwargs...)
    _term, TermKinetic_pullback = rrule_via_ad(config, T_simple, basis)
    return term, TermKinetic_pullback
    # function back(Δterm)
    #     ∂T, ∂basis = TermKinetic_pullback(Δterm)
    #     @show ∂T
    #     @show ∂basis
    #     return (Tangent{typeof(T)}(;∂T...), Tangent{typeof(basis)}(;∂basis...))
    # end
    # return term, back
end


