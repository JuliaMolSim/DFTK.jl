using ChainRulesCore
using Zygote: @adjoint  # TODO remove, once ChainRules rrules can overrule Zygote
import AbstractFFTs


function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, f_real::AbstractArray)
    @warn "r_to_G rrule triggered."
    f_fourier = r_to_G(basis, f_real)
    function r_to_G_pullback(Δf_fourier)
        ∂f_real = G_to_r(basis, complex(Δf_fourier)) * basis.r_to_G_normalization / basis.G_to_r_normalization
        ∂normalization = real(dot(Δf_fourier, f_fourier)) / basis.r_to_G_normalization
        ∂basis = Tangent{typeof(basis)}(;r_to_G_normalization=∂normalization)
        return NoTangent(), ∂basis, real(∂f_real)
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray)
    @warn "r_to_G kpoint rrule triggered."
    f_fourier = r_to_G(basis, kpt, f_real)
    function r_to_G_pullback(Δf_fourier)
        ∂f_real = G_to_r(basis, kpt, complex(Δf_fourier)) * basis.r_to_G_normalization / basis.G_to_r_normalization
        ∂normalization = real(dot(Δf_fourier, f_fourier)) / basis.r_to_G_normalization
        ∂basis = Tangent{typeof(basis)}(;r_to_G_normalization=∂normalization)
        return NoTangent(), ∂basis, NoTangent(), ∂f_real
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, f_fourier::AbstractArray; kwargs...)
    @warn "G_to_r rrule triggered."
    f_real = G_to_r(basis, f_fourier; kwargs...)
    function G_to_r_pullback(Δf_real)
        ∂f_fourier = r_to_G(basis, real(Δf_real)) * basis.G_to_r_normalization / basis.r_to_G_normalization
        ∂normalization = real(dot(Δf_real, f_real)) / basis.G_to_r_normalization
        ∂basis = Tangent{typeof(basis)}(;G_to_r_normalization=∂normalization)
        return NoTangent(), ∂basis, ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractVector)
    @warn "G_to_r kpoint rrule triggered."
    f_real = G_to_r(basis, kpt, f_fourier)
    function G_to_r_pullback(Δf_real)
        ∂f_fourier = r_to_G(basis, kpt, complex(Δf_real)) * basis.G_to_r_normalization / basis.r_to_G_normalization
        ∂normalization = real(dot(Δf_real, f_real)) / basis.G_to_r_normalization
        ∂basis = Tangent{typeof(basis)}(;G_to_r_normalization=∂normalization)
        return NoTangent(), ∂basis, NoTangent(), ∂f_fourier
    end
    return f_real, G_to_r_pullback
end


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
ChainRulesCore.@non_differentiable default_symmetries(::Any...) # TODO perhaps?

# demanded by Zygote
function ChainRulesCore.rrule(T::Type{Pair{ElementPsp,T2}}, el, x) where {T2}
    @warn "Pair{ElementPsp,T2} constructor rrule triggered."
    return T(el, x), ΔTx -> (NoTangent(), NoTangent(), ΔTx.second)
end

# TODO delete
@adjoint (T::Type{<:SArray})(x...) = T(x...), y->(y,)

# TODO delete, or understand why this is necessary
function ChainRulesCore.rrule(T::Type{Vector{Kpoint{Float64}}}, x)
    @warn "strange Vector{Kpoint{Float64}} rrule triggered"
    return T(x), ΔTx -> (NoTangent(), ΔTx)
end


# simplified version of the Model constructor to
# help reverse mode AD to only differentiate the relevant computations.
# this excludes assertions (try-catch), and symmetries
function _autodiff_Model_namedtuple(lattice, atoms, terms)
    T = eltype(lattice)
    recip_lattice = 2T(π)*inv(lattice')
    unit_cell_volume = abs(det(lattice))
    recip_cell_volume = abs(det(recip_lattice))
    (;lattice=lattice, recip_lattice=recip_lattice, unit_cell_volume=unit_cell_volume, 
    recip_cell_volume=recip_cell_volume, atoms=atoms, term_types=terms)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{Model}, lattice, atoms, terms; kwargs...)
    @warn "simplified Model rrule triggered."
    model = T(lattice, atoms, terms; kwargs...)
    _model, Model_pullback = rrule_via_ad(config, _autodiff_Model_namedtuple, lattice, atoms, terms)
    # TODO add some assertion that model and _model agree
    return model, Model_pullback
end

# a workaround to manually updating immutable Tangent fields
function _update_tangent(t::Tangent{P,T}, nt::NamedTuple) where {P,T}
    return Tangent{P,T}(merge(ChainRulesCore.backing(t), nt))
end

function ChainRulesCore.rrule(::typeof(build_kpoints), model::Model{T}, fft_size, kcoords, Ecut; variational=true) where T
    @warn "build_kpoints rrule triggered"
    kpoints = build_kpoints(model, fft_size, kcoords, Ecut; variational=variational)
    function build_kpoints_pullback(Δkpoints)
        ∂model = sum(Δkpoints).model # TODO double-check.
        ∂recip_lattice = ∂model.recip_lattice + sum([Δkp.coordinate_cart * kp.coordinate' for (kp, Δkp) in zip(kpoints, Δkpoints) if !(Δkp isa NoTangent)])
        # ∂model.recip_lattice += ∂recip_lattice # Tangents are immutable
        ∂model = _update_tangent(∂model, (;recip_lattice=∂recip_lattice))
        ∂kcoords = @not_implemented("TODO")
        return NoTangent(), ∂model, NoTangent(), ∂kcoords, NoTangent()
    end
    return kpoints, build_kpoints_pullback
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

    # kpoints have differentiable components (inside model and coordinate_cart)
    kcoords_thisproc = basis.kcoords_global[basis.krange_thisproc] # TODO which kcoords?
    kpoints = build_kpoints(model, basis.fft_size, kcoords_thisproc, basis.Ecut; basis.variational)

    # cicularity is getting complicated...
    # To correctly instantiate term types, we do need a full PlaneWaveBasis struct;
    # so we need to interleave re-computed differentiable params, and fixed params in basis
    _basis = PlaneWaveBasis{T}( # this shouldn't hit the rrule below a second time due to more args
        model, basis.fft_size, dvol, 
        basis.Ecut, basis.variational,
        basis.opFFT, basis.ipFFT, basis.opBFFT, basis.ipBFFT,
        r_to_G_normalization, G_to_r_normalization,
        kpoints, basis.kweights, basis.ksymops, basis.kgrid, basis.kshift,
        basis.kcoords_global, basis.ksymops_global, basis.comm_kpts, basis.krange_thisproc, basis.krange_allprocs,
        basis.symmetries, terms)

    # terms = Any[t(_basis) for t in model.term_types]
    terms = vcat([], [t(_basis) for t in model.term_types]) # hack: enforce Vector{Any} without causing reverse mutation

    (;model=model, kpoints=kpoints, dvol=dvol, terms=terms, G_to_r_normalization=G_to_r_normalization, r_to_G_normalization=r_to_G_normalization)
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
end

Base.zero(::ElementPsp) = ZeroTangent() # TODO 

function ChainRulesCore.rrule(T::Type{TermAtomicLocal}, basis, potential)
    TermAtomicLocal_pullback(Δ) = NoTangent(), Δ.basis, Δ.potential
    return T(basis, potential), TermAtomicLocal_pullback
end

function _autodiff_AtomicLocal(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model

    pot_fourier = map(G_vectors(basis)) do G
        pot = zero(T)
        for (elem, positions) in model.atoms
            form_factor::T = local_potential_fourier(elem, norm(model.recip_lattice * G))
            for r in positions
                pot += cis(-2T(π) * dot(G, r)) * form_factor
            end
        end
        pot / sqrt(model.unit_cell_volume)
    end

    pot_real = G_to_r(basis, pot_fourier)
    TermAtomicLocal(basis, real(pot_real))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, E::AtomicLocal, basis::PlaneWaveBasis{T}) where {T}
    @warn "simplified AtomicLocal rrule triggered."
    term = E(basis)
    _term, AtomicLocal_pullback = rrule_via_ad(config, _autodiff_AtomicLocal, basis)
    return term, AtomicLocal_pullback
end

# compute_density rrule

function _compute_partial_density(basis, kpt, ψk, occupation)
    # if one directly tries `sum(1:N) do n ... end` one gets "Dimension mismatch" in reverse pass
    # TODO potential bug in ChainRules sum rrule?

    ρk_real = sum(zip(eachcol(ψk), occupation)) do (ψnk, occn)
        ψnk_real_tid = G_to_r(basis, kpt, ψnk)
        occn .* abs2.(ψnk_real_tid)
    end

    # FFT and return
    r_to_G(basis, ρk_real)
end

function _accumulate_over_symmetries(ρin, basis, symmetries)
    T = eltype(basis)
    # trying sum(...) do ... directly here gives a Zygote error
    x = map(
        function (sym)
            S, τ = sym
            invS = Mat3{Int}(inv(S))
            ρS = map(G_vectors(basis)) do G
                igired = index_G_vectors(basis, invS * G)
                if isnothing(igired)
                    zero(complex(T))
                else
                    cis(-2T(π) * dot(G, τ)) * ρin[igired] 
                    # TODO: indexing into large arrays can cause OOM in reverse
                end
            end
            ρS
        end,
        symmetries
    )
    ρaccu = sum(x)
    ρaccu
end

function _lowpass_for_symmetry(ρ, basis; symmetries=basis.model.symmetries)
    ρnew = deepcopy(ρ)
    ρnew = lowpass_for_symmetry!(ρnew, basis; symmetries)
    ρnew
end

function ChainRulesCore.rrule(::typeof(_lowpass_for_symmetry), ρ, basis; symmetries=basis.model.symmetries)
    @warn "_lowpass_for_symmetry rrule triggered."
    ρnew = _lowpass_for_symmetry(ρ, basis; symmetries)
    function lowpass_for_symmetry_pullback(Δρ)
        ∂ρ = _lowpass_for_symmetry(Δρ, basis; symmetries)
        return NoTangent(), ∂ρ, NoTangent()
    end
    return ρnew, lowpass_for_symmetry_pullback
end

function _autodiff_compute_density(basis::PlaneWaveBasis, ψ, occupation)
    # try one kpoint only (for debug) TODO re-enable all kpoints
    ρsum = _compute_partial_density(basis, basis.kpoints[1], ψ[1], occupation[1])
    ρ = reshape(ρsum, size(ρsum)..., 1)

    # ρsum = map(eachindex(basis.kpoints)) do ik # TODO re-write without indexing (causes OOM in reverse)
    #     kpt = basis.kpoints[ik]
    #     ρ_k = _compute_partial_density(basis, kpt, ψ[ik], occupation[ik])
    #     ρ_k = _lowpass_for_symmetry(ρ_k, basis)
    #     ρaccu = _accumulate_over_symmetries(ρ_k, basis, basis.ksymops[ik])
    #     ρaccu
    # end
    # ρsum = sum(ρsum)
    # ρaccus = [reshape(ρsum, size(ρsum)..., 1)] # TODO handle n_spin > 1

    # # Count the number of k-points modulo spin
    # n_spin = basis.model.n_spin_components
    # count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    # count = mpi_sum(count, basis.comm_kpts)

    # ρ = sum(ρaccus) ./ count
    # # mpi_sum!(ρ, basis.comm_kpts)

    G_to_r(basis, ρ)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(compute_density), basis::PlaneWaveBasis, ψ, occupation)
    @warn "simplified compute_density rrule triggered."
    ρ = compute_density(basis, ψ, occupation)
    _ρ, compute_density_pullback = rrule_via_ad(config, _autodiff_compute_density, basis, ψ, occupation)
    return ρ, compute_density_pullback
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(self_consistent_field), basis::PlaneWaveBasis; kwargs...)
    @warn "self_consistent_field rrule triggered."
    scfres = self_consistent_field(basis; kwargs...)
    ψ = scfres.ψ
    occupation = scfres.occupation
    function self_consistent_field_pullback(Δscfres)
        @show typeof(Δscfres)
        # function Hψ(basis, ψ, occupation)
        #     ρ = compute_density(basis, ψ, occupation)
        #     _, H = energy_hamiltonian(basis, ψ, occupation; ρ)
        #     H*ψ
        # end
        # δHψ = rrule_via_ad(config, Hψ, basis, ψ, occupation)[2](Δscfres.ψ)
        # return NoTangent(), ∂basis
        return NoTangent(), NoTangent()
    end
    return scfres, self_consistent_field_pullback
end
