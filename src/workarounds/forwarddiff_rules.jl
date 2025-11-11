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
    ipFFT  = DummyInplace(opFFT)
    ipBFFT = DummyInplace(opBFFT)
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

# determine symmetry operations only from primal values, then filter out symmetries broken by dual part
function symmetry_operations(lattice::AbstractMatrix{<:Dual},
                             atoms, positions, magnetic_moments=[];
                             tol_symmetry=SYMMETRY_TOLERANCE, kwargs...)
    positions_value = [ForwardDiff.value.(pos) for pos in positions]
    symmetries = symmetry_operations(ForwardDiff.value.(lattice), atoms,
                                     positions_value, magnetic_moments;
                                     tol_symmetry, kwargs...)
    remove_dual_broken_symmetries(lattice, atoms, positions, symmetries; tol_symmetry)
end

function remove_dual_broken_symmetries(lattice, atoms, positions,
                                       symmetries; tol_symmetry=SYMMETRY_TOLERANCE)
    filter(symmetries) do symmetry
        !is_symmetry_broken_by_dual(lattice, atoms, positions, symmetry; tol_symmetry)
    end
end

"""
Return `true` if a symmetry that holds for the primal part is broken by
a perturbation in the lattice or in the positions, `false` otherwise.
"""
function is_symmetry_broken_by_dual(lattice, atoms, positions, symmetry::SymOp; tol_symmetry)
    # The symmetry operation is given by (W,w) where W is a 3x3 matrix and w a 3-vector,
    # both in reduced (fractional) coordinates.
    # For any lattice atom at position x, W*x + w should be in the lattice.
    # This means that for any atom position x (in reduced coordinates) in the unit cell
    # and any 3 integers u, there should be an atom at position y and 3 integers v such that:
    # W * (x + u) + w = y + v
    #
    # Additionally, Wcart = A * W * A⁻¹ (in cartesian coordinates) must be orthogonal.
    # 
    # The strategy is then to check that:
    # 1. Metric invariance: Verify that the perturbed lattice metric
    #    G = A' * A satisfies W' * G * W = G.
    #    (This is equivalent to checking that Wcart = A * W * A⁻¹ is orthogonal.)
    # 2. Atomic mapping:
    #    W is an integer matrix, such that any change in u is easily compensated for in v.
    #    Thus we only need to check that for each atomic position `x`
    #    `W*x + w = y (mod 1)` is still satisfied for some equivalent atom `y`
    #    of the same species, up to numerical tolerance.
    #    Only the dual (perturbation) components are checked,
    #    the primal part is assumed to already obey the symmetry.

    # TODO: work out the case of nested Duals for higher-order derivatives
    #       (eg. symmetry breaking at higher order)
    is_dual_nonzero(x::AbstractArray) = any(x) do xi
        maximum(abs, ForwardDiff.partials(xi)) >= tol_symmetry
    end

    W = symmetry.W
    w = symmetry.w

    # 1) Metric tensor check in fractional frame: Wᵀ G W == G (to first order)
    G = lattice'lattice  # 3x3 metric tensor on the unit cube
    ΔG = W' * G * W - G
    if is_dual_nonzero(ΔG)
        return true  # anisotropic lattice perturbation breaks this symmetry
    end

    # 2) Atomic mapping check stays purely in reduced coordinates.
    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]
    for group in atom_groups
        positions_group = positions[group]
        for y in positions_group
            i_other = find_symmetry_preimage(positions_group, y, symmetry; tol_symmetry)
            x = positions_group[i_other]
            # Check that W*x + w - y has zero dual part (primal already ok by construction)
            if is_dual_nonzero(W * x + w - y)
                return true
            end
        end
    end

    false
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
          magnetic_moments=value_type(T)[],  # Symmetries given explicitly
          terms=map(construct_term_value, model.term_types),
          temperature=ForwardDiff.value(model.temperature),
          model.smearing,
          εF=ForwardDiff.value(model.εF),
          model.spin_polarization,
          model.symmetries,
          # Can be safely disabled: this has been checked for basis.model
          disable_electrostatics_check=true)
end

construct_value(el::Element) = el
function construct_value(el::ElementPsp)
    ElementPsp(el.species, construct_value(el.psp), el.family, el.mass)
end
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
function construct_value(psp::PspUpf{T,I}) where {T <: AbstractFloat, I <: AbstractArray{<:AbstractFloat}}
    # NOTE: This permits non-Dual UPF pseudos to be used in ForwardDiff computations,
    #       but does not yet permit response derivatives w.r.t. UPF parameters.
    psp
end

# term types don't have a supertype, so we use a different function name
# by default, don't convert
construct_term_value(x::Any) = x
function construct_term_value(x::Xc)
    functionals = map(x.functionals) do f
        if f isa DispatchFunctional
            f
        else
            construct_value(f)
        end
    end
    Xc(functionals, ForwardDiff.value(x.scaling_factor), x.potential_threshold, x.use_nlcc)
end
# get the value of each field of the functional
@generated function construct_value(x::F) where {F <: DftFunctionals.Functional}
    new_arguments = map(fieldnames(F)) do fn
        :(ForwardDiff.value(x.$fn))
    end
    # Use typename(...).wrapper to reconstruct the correct parametrized type (without Duals)
    :($(Base.typename(F).wrapper)($(new_arguments...)))
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


@timing "self_consistent_field ForwardDiff" function self_consistent_field(
        basis_dual::PlaneWaveBasis{T};
        response=ResponseOptions(),
        kwargs...) where {T <: Dual}
    # Note: No guarantees on this interface yet.

    # Primal pass
    basis_primal = construct_value(basis_dual)
    scfres = self_consistent_field(basis_primal; kwargs...)

    # Compute explicit density perturbation (including strain) due to normalization
    ρ_basis = compute_density(basis_dual, scfres.ψ, scfres.occupation)

    # Compute external perturbation (contained in ham_dual)
    Hψ_dual = let
        ham_dual = energy_hamiltonian(basis_dual, scfres.ψ, scfres.occupation;
                                      ρ=ρ_basis, scfres.eigenvalues,
                                      scfres.εF).ham
        ham_dual * scfres.ψ
    end
    # Implicit differentiation
    response.verbose && println("Solving response problem")
    δresults = ntuple(ForwardDiff.npartials(T)) do α
        δHextψ = [ForwardDiff.partials.(δHextψk, α) for δHextψk in Hψ_dual]
        δtemperature = ForwardDiff.partials(basis_dual.model.temperature, α)
        solve_ΩplusK_split(scfres, δHextψ; δtemperature,
                           tol=last(scfres.history_Δρ), response.verbose)
    end

    # Convert and combine
    DT = Dual{ForwardDiff.tagtype(T)}
    ψ = map(scfres.ψ, getfield.(δresults, :δψ)...) do ψk, δψk...
        map(ψk, δψk...) do ψnk, δψnk...
            Complex(DT(real(ψnk), real.(δψnk)),
                    DT(imag(ψnk), imag.(δψnk)))
        end
    end
    eigenvalues = map(scfres.eigenvalues, getfield.(δresults, :δeigenvalues)...) do εk, δεk...
        map((εnk, δεnk...) -> DT(εnk, δεnk), εk, δεk...)
    end
    occupation = map(scfres.occupation, getfield.(δresults, :δoccupation)...) do occk, δocck...
        map((occnk, δoccnk...) -> DT(occnk, δoccnk), occk, δocck...)
    end
    εF = DT(scfres.εF, getfield.(δresults, :δεF)...)

    # For strain, basis_dual contributes an explicit lattice contribution which
    # is not contained in δresults, so we need to recompute ρ here
    ρ = compute_density(basis_dual, ψ, occupation)

    # TODO Could add δresults[α].δVind the dual part of the total local potential in ham_dual
    # and in this way return a ham that represents also the total change in Hamiltonian

    energies, ham = energy_hamiltonian(basis_dual, ψ, occupation; ρ, eigenvalues, εF)

    # This has to be changed whenever the scfres structure changes
    (; ham, basis=basis_dual, energies, ρ, eigenvalues, occupation, εF, ψ,
       scfres.τ, # TODO make τ also differentiable for meta-GGA DFPT 
       # non-differentiable metadata:
       response=getfield.(δresults, :info_gmres),
       scfres.converged, scfres.occupation_threshold, scfres.α, scfres.n_iter,
       scfres.n_bands_converge, scfres.n_matvec, scfres.diagonalization, scfres.stage,
       scfres.history_Δρ, scfres.history_Etot, scfres.timedout, scfres.mixing,
       scfres.seed, scfres.algorithm, scfres.runtime_ns)
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

# Waiting for https://github.com/JuliaDiff/DiffRules.jl/pull/37 to be merged
function erfc(x::Complex{Dual{T,V,N}}) where {T,V,N}
    xx = complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
    dx = complex.(ForwardDiff.partials(real(x)), ForwardDiff.partials(imag(x)))
    dgamma = -2*exp(-xx^2)/sqrt(V(π)) * dx
    complex(Dual{T,V,N}(real(erfc(xx)), ForwardDiff.Partials{N,V}(tuple(real(dgamma)...))),
            Dual{T,V,N}(imag(erfc(xx)), ForwardDiff.Partials{N,V}(tuple(imag(dgamma)...))))
end
