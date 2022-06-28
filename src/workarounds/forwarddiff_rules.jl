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

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{T}}}) where T
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
#   _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{T}}}) where T
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

_fftw_flags(::Type{<:ForwardDiff.Dual}) = FFTW.MEASURE | FFTW.UNALIGNED

function build_fft_plans(T::Type{<:Union{ForwardDiff.Dual,Complex{<:ForwardDiff.Dual}}}, fft_size)
    tmp = Array{complex(T)}(undef, fft_size...) # TODO think about other Array types
    opFFT  = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    opBFFT = FFTW.plan_bfft(tmp, flags=_fftw_flags(T))

    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)
    # backward by inverting and stripping off normalizations
    ipFFT, opFFT, ipBFFT, opBFFT
end

# determine symmetry operations only from primal lattice values
function spglib_get_symmetry(lattice::AbstractMatrix{<:ForwardDiff.Dual}, atom_groups, positions,
                             magnetic_moments=[]; kwargs...)
    spglib_get_symmetry(ForwardDiff.value.(lattice), atom_groups, positions,
                        magnetic_moments; kwargs...)
end
function spglib_atoms(atom_groups,
                      positions::AbstractVector{<:AbstractVector{<:ForwardDiff.Dual}},
                      magnetic_moments)
    positions_value = [ForwardDiff.value.(pos) for pos in positions]
    spglib_atoms(atom_groups, positions_value, magnetic_moments)
end

function _is_well_conditioned(A::AbstractArray{<:ForwardDiff.Dual}; kwargs...)
    _is_well_conditioned(ForwardDiff.value.(A); kwargs...)
end

value_type(T::Type{<:ForwardDiff.Dual}) = ForwardDiff.valtype(T)

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
          model_name=model.model_name,
          n_electrons=model.n_electrons,
          magnetic_moments=[],  # Symmetries given explicitly
          terms=model.term_types,
          temperature=ForwardDiff.value(model.temperature),
          smearing=model.smearing,
          spin_polarization=model.spin_polarization,
          symmetries=model.symmetries)
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
    new_kshift = isnothing(basis.kshift) ? nothing : ForwardDiff.value.(basis.kshift)
    PlaneWaveBasis(construct_value(basis.model),
                   ForwardDiff.value(basis.Ecut),
                   map(v -> ForwardDiff.value.(v), basis.kcoords_global),
                   ForwardDiff.value.(basis.kweights_global);
                   basis.symmetries_respect_rgrid,
                   fft_size=basis.fft_size,
                   kgrid=basis.kgrid,
                   kshift=new_kshift,
                   variational=basis.variational,
                   comm_kpts=basis.comm_kpts)
end

function self_consistent_field(basis_dual::PlaneWaveBasis{T};
                               response=ResponseOptions(),
                               kwargs...) where T <: ForwardDiff.Dual
    # Note: No guarantees on this interface yet.

    # Primal pass
    basis  = construct_value(basis_dual)
    scfres = self_consistent_field(basis; kwargs...)

    # Split into truncated and extra eigenpairs
    truncated = select_occupied_orbitals(scfres; threshold=response.occupation_threshold)
    ψ_extra = [@view scfres.ψ[ik][:, length(εkTrunc)+1:end]
               for (ik, εkTrunc) in enumerate(truncated.eigenvalues)]
    ε_extra = [@view scfres.eigenvalues[ik][length(εkTrunc)+1:end]
               for (ik, εkTrunc) in enumerate(truncated.eigenvalues)]

    ## promote occupied bands to dual numbers
    occupation_dual = [T.(occₖ) for occₖ in truncated.occupation]
    ψ_dual = [Complex.(T.(real(ψₖ)), T.(imag(ψₖ))) for ψₖ in truncated.ψ]
    ρ_dual = DFTK.compute_density(basis_dual, ψ_dual, occupation_dual)
    εF_dual = T(truncated.εF)  # Only needed for entropy term
    eigenvalues_dual = [T.(εₖ) for εₖ in truncated.eigenvalues]
    energies_dual, ham_dual = energy_hamiltonian(basis_dual, ψ_dual, occupation_dual;
                                                 ρ=ρ_dual, eigenvalues=eigenvalues_dual,
                                                 εF=εF_dual)

    response.verbose && println("Solving response problem")

    ## Implicit differentiation
    hamψ_dual = ham_dual * ψ_dual
    δresults = ntuple(ForwardDiff.npartials(T)) do α
        δHψ_α = [ForwardDiff.partials.(δHψk, α) for δHψk in hamψ_dual]

        δψ_α, resp_α = solve_ΩplusK_split(truncated, -δHψ_α; tol=scfres.norm_Δρ,
                                          ψ_extra, ε_extra, response.verbose)
        δρ_α = compute_δρ(basis, truncated.ψ, δψ_α, truncated.occupation)
        δψ_α, δρ_α, resp_α
    end
    δψ       = [δψ_α   for (δψ_α, δρ_α, resp_α) in δresults]
    δρ       = [δρ_α   for (δψ_α, δρ_α, resp_α) in δresults]
    response = [resp_α for (δψ_α, δρ_α, resp_α) in δresults]

    ## Convert, combine and return
    DT = ForwardDiff.Dual{ForwardDiff.tagtype(T)}
    ψ_out = map(truncated.ψ, δψ...) do ψk, δψk...
        map(ψk, δψk...) do ψi, δψi...
            Complex(DT(real(ψi), real.(δψi)),
                    DT(imag(ψi), imag.(δψi)))
        end
    end
    ρ_out = map((ρi, δρi...) -> DT(ρi, δρi), scfres.ρ, δρ...)

    # TODO Compute eigenvalue response (return dual eigenvalues and dual εF)

    merge(truncated, (; ham=ham_dual, basis=basis_dual, energies=energies_dual, ψ=ψ_out,
                        occupation=occupation_dual, ρ=ρ_out, response))
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
