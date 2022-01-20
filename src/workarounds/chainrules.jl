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


# constructor with all AD-compatible args as positional args
function Model(lattice, atoms, terms; kwargs...)
    return Model(lattice; atoms, terms, kwargs...)
end

# simplified version of the Model constructor to
# help reverse mode AD to only differentiate the relevant computations.
# this excludes assertions (try-catch), and symmetries
function _autodiff_Model_namedtuple(lattice, atoms, term_types)
    recip_lattice = compute_recip_lattice(lattice)
    unit_cell_volume  = compute_unit_cell_volume(lattice)
    recip_cell_volume = compute_unit_cell_volume(recip_lattice)
    (; lattice, recip_lattice, unit_cell_volume, recip_cell_volume, atoms, term_types)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{Model}, lattice, atoms, terms; kwargs...)
    @warn "simplified Model rrule triggered."
    model = T(lattice, atoms, terms; kwargs...)
    _model, Model_pullback = rrule_via_ad(config, _autodiff_Model_namedtuple, lattice, atoms, terms)
    # TODO add some assertion that model and _model agree
    return model, Model_pullback
end


function ChainRulesCore.rrule(::typeof(build_kpoints), model::Model{T}, fft_size, kcoords, Ecut; variational=true) where T
    @warn "build_kpoints rrule triggered"
    kpoints = build_kpoints(model, fft_size, kcoords, Ecut; variational=variational)
    function build_kpoints_pullback(Δkpoints)
        sum_Δkpoints = sum(Δkpoints)
        if sum_Δkpoints isa NoTangent
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        ∂recip_lattice = sum([Δkp.coordinate_cart * kp.coordinate' for (kp, Δkp) in zip(kpoints, Δkpoints) if !(Δkp isa NoTangent)])
        ∂model = Tangent{typeof(model)}(; recip_lattice=∂recip_lattice)
        ∂kcoords = @not_implemented("TODO")
        return NoTangent(), ∂model, NoTangent(), ∂kcoords, NoTangent()
    end
    return kpoints, build_kpoints_pullback
end

# explicit rule for PlaneWaveBasis inner constructor
function ChainRulesCore.rrule(PT::Type{PlaneWaveBasis{T}},
                              model::Model{T},
                              fft_size::Tuple{Int, Int, Int},
                              dvol::T,
                              Ecut::T,
                              variational::Bool,
                              opFFT,
                              ipFFT,
                              opBFFT,
                              ipBFFT,
                              r_to_G_normalization::T,
                              G_to_r_normalization::T,
                              kpoints::Vector{Kpoint},
                              kweights::Vector{T},
                              ksymops::Vector{Vector{SymOp}},
                              kgrid::Union{Nothing,Vec3{Int}},
                              kshift::Union{Nothing,Vec3{T}},
                              kcoords_global::Vector{Vec3{T}},
                              ksymops_global::Vector{Vector{SymOp}},
                              comm_kpts::MPI.Comm,
                              krange_thisproc::Vector{Int},
                              krange_allprocs::Vector{Vector{Int}},
                              symmetries::Vector{SymOp},
                              terms::Vector{Any}) where {T <: Real}
    @warn "PlaneWaveBasis inner constructor rrule triggered."
    basis = PT(
        model, fft_size, dvol, Ecut, variational, opFFT, ipFFT, opBFFT, ipBFFT,
        r_to_G_normalization, G_to_r_normalization, kpoints, kweights, ksymops,
        kgrid, kshift, kcoords_global, ksymops_global, comm_kpts, 
        krange_thisproc, krange_allprocs, symmetries, terms
    )
    function PT_pullback(Δbasis)
        return (NoTangent(), Δbasis.model, NoTangent(), Δbasis.dvol, Δbasis.Ecut, 
                NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                Δbasis.r_to_G_normalization, Δbasis.G_to_r_normalization, Δbasis.kpoints, Δbasis.kweights, Δbasis.ksymops,
                Δbasis.kgrid, Δbasis.kshift, Δbasis.kcoords_global, Δbasis.ksymops_global, Δbasis.comm_kpts, 
                NoTangent(), NoTangent(), Δbasis.symmetries, Δbasis.terms)
    end
    return basis, PT_pullback
end

# simplified version of PlaneWaveBasis outer constructor to
# help reverse mode AD to only differentiate the relevant computations.
# this excludes assertions (try-catch), MPI handling, and other things
function _autodiff_PlaneWaveBasis_namedtuple(model::Model{T}, basis::PlaneWaveBasis) where {T <: Real}
    dvol = model.unit_cell_volume ./ prod(basis.fft_size)
    G_to_r_normalization = 1 / sqrt(model.unit_cell_volume)
    r_to_G_normalization = sqrt(model.unit_cell_volume) / length(basis.ipFFT)

    # Create dummy terms array for _basis to handle
    terms = Vector{Any}(undef, length(model.term_types))

    # kpoints have differentiable components (inside coordinate_cart)
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

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{PlaneWaveBasis}, model::Model; kwargs...)
    @warn "simplified PlaneWaveBasis rrule triggered."
    basis = T(model; kwargs...)
    f(model) = _autodiff_PlaneWaveBasis_namedtuple(model, basis)
    _basis, PlaneWaveBasis_pullback = rrule_via_ad(config, f, model)
    return basis, PlaneWaveBasis_pullback
end


# convert generators into arrays (needed for Zygote here)
function _G_vectors_cart(basis::PlaneWaveBasis)
    [basis.model.recip_lattice * G for G in G_vectors(basis.fft_size)]
end
function _Gplusk_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)
    [basis.model.recip_lattice * (G + kpt.coordinate) for G in G_vectors(basis, kpt)]
end

function _autodiff_TermKinetic_namedtuple(basis, scaling_factor)
    kinetic_energies = [[scaling_factor * sum(abs2, G + kpt.coordinate_cart) / 2
                         for G in _Gplusk_vectors_cart(basis, kpt)]
                        for kpt in basis.kpoints]
    (;scaling_factor=scaling_factor, kinetic_energies=kinetic_energies)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{TermKinetic}, basis::PlaneWaveBasis, scaling_factor)
    @warn "simplified TermKinetic rrule triggered."
    term = T(basis, scaling_factor)
    _term, TermKinetic_pullback = rrule_via_ad(config, _autodiff_TermKinetic_namedtuple, basis, scaling_factor)
    return term, TermKinetic_pullback
end

Base.zero(::ElementPsp) = ZeroTangent() # TODO

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
    TermAtomicLocal(real(pot_real))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, E::AtomicLocal, basis::PlaneWaveBasis{T}) where {T}
    @warn "simplified AtomicLocal rrule triggered."
    term = E(basis)
    _term, AtomicLocal_pullback = rrule_via_ad(config, _autodiff_AtomicLocal, basis)
    return term, AtomicLocal_pullback
end


function _autodiff_TermHartree(basis::PlaneWaveBasis{T}, scaling_factor) where T
    poisson_green_coeffs = map(_G_vectors_cart(basis)) do G
        abs2G = sum(abs2, G)
        abs2G > 0. ? 4T(π)/abs2G : 0.
    end
    TermHartree(T(scaling_factor), T(scaling_factor) .* poisson_green_coeffs)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{TermHartree}, basis::PlaneWaveBasis, scaling_factor)
    @warn "TermHartree rrule triggered."
    term = T(basis, scaling_factor)
    _term, TermHartree_pullback = rrule_via_ad(config, _autodiff_TermHartree, basis, scaling_factor)
    return term, TermHartree_pullback
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

# workaround to pass rrule_via_ad kwargs
DFTK.energy_hamiltonian(basis, ψ, occ, ρ) = DFTK.energy_hamiltonian(basis, ψ, occ; ρ=ρ)

solve_ΩplusK(basis::PlaneWaveBasis, ψ, ::NoTangent, occupation) = 0ψ

function _autodiff_fast_hblock_mul(fast_hblock::NamedTuple, ψ)
    # a pure version of *(H::HamiltonianBlock, ψ)
    # TODO this currently only considers kinetic+local
    H = fast_hblock.H
    basis = H.basis
    kpt = H.kpoint
    nband = size(ψ, 2)

    potential = fast_hblock.real_op.potential
    potential = potential / prod(basis.fft_size)  # because we use unnormalized plans

    function Hψ(ψk)
        ψ_real = G_to_r(basis, kpt, ψk) .* potential ./ basis.G_to_r_normalization
        Hψ_k = r_to_G(basis, kpt, ψ_real) ./ basis.r_to_G_normalization
        Hψ_k = Hψ_k + fast_hblock.fourier_op.multiplier .* ψk
        Hψ_k
    end
    Hψ = reduce(hcat, map(Hψ, eachcol(ψ)))

    Hψ
end


# a pure version of *(H::Hamiltonian, ψ)
function _autodiff_apply_hamiltonian(H::Hamiltonian, ψ)
    return [
        _autodiff_fast_hblock_mul(
            # TODO clean up and generalize
            (fourier_op=hblock.optimized_operators[1], real_op=hblock.optimized_operators[2], H=hblock),
            ψk
        )
        for (hblock, ψk) in zip(H.blocks, ψ)
    ]
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(*), H::Hamiltonian, ψ)
    @warn "H * ψ rrule triggered."
    rrule_via_ad(config, _autodiff_apply_hamiltonian, H, ψ)
end

# avoids Energies OrderedDict struct (incompatible with Zygote)
function _autodiff_energy_hamiltonian(basis, ψ, occ, ρ)
    ene_ops_arr = [DFTK.ene_ops(term, basis, ψ, occ; ρ=ρ) for term in basis.terms]
    energies    = [eh.E for eh in ene_ops_arr]
    operators   = [eh.ops for eh in ene_ops_arr]         # operators[it][ik]

    # flatten the inner arrays in case a term returns more than one operator
    flatten(arr) = reduce(vcat, map(a -> (a isa Vector) ? a : [a], arr))
    hks_per_k   = [flatten([blocks[ik] for blocks in operators])
                   for ik = 1:length(basis.kpoints)]      # hks_per_k[ik][it]

    # Preallocated scratch arrays
    T = eltype(basis)
    scratch = (
        ψ_reals=[zeros(complex(T), basis.fft_size...) for tid = 1:Threads.nthreads()],
    )

    H = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks, scratch)
                            for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    return energies, H
end


function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{HamiltonianBlock}, basis, kpt, operators, scratch)
    @warn "HamiltonianBlock rrule triggered."
    _, optimize_operators_pullback = rrule_via_ad(config, optimize_operators_, operators)
    function T_pullback(∂hblock)
        _, ∂operators = optimize_operators_pullback(∂hblock.optimized_operators)
        ∂operators = ∂operators + ∂hblock.operators
        return NoTangent(), ∂hblock.basis, ∂hblock.kpoint, ∂operators, ∂hblock.scratch
    end
    return T(basis, kpt, operators, scratch), T_pullback
end


function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(self_consistent_field), basis::PlaneWaveBasis; kwargs...)
    @warn "self_consistent_field rrule triggered."
    scfres = self_consistent_field(basis; kwargs...)

    ψ, occupation = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)

    ## Zygote doesn't like OrderedDict in Energies
    # (energies, H), energy_hamiltonian_pullback =
    #     rrule_via_ad(config, energy_hamiltonian, basis, scfres.ψ, scfres.occupation, scfres.ρ)

    (energies, H), energy_hamiltonian_pullback =
        rrule_via_ad(config, _autodiff_energy_hamiltonian, basis, scfres.ψ, scfres.occupation, scfres.ρ)
    Hψ, mul_pullback =
        rrule(config, *, H, ψ)
    ρ, compute_density_pullback =
        rrule(config, compute_density, basis, scfres.ψ, scfres.occupation)

    function self_consistent_field_pullback(Δscfres)
        δψ = Δscfres.ψ
        δoccupation = Δscfres.occupation
        δρ = Δscfres.ρ
        δenergies = Δscfres.energies
        δbasis = Δscfres.basis
        δH = Δscfres.ham

        _, ∂basis, ∂ψ, _ = compute_density_pullback(δρ)
        ∂ψ = ∂ψ + δψ
        ∂ψ, occupation = DFTK.select_occupied_orbitals(basis, ∂ψ, occupation)

        ∂Hψ = solve_ΩplusK(basis, ψ, -∂ψ, occupation) # use self-adjointness of dH ψ -> dψ

        # TODO need to do proj_tangent on ∂Hψ
        _, ∂H, _ = mul_pullback(∂Hψ)
        ∂H = ∂H + δH
        _, ∂basis, _, _, _ = energy_hamiltonian_pullback((δenergies, ∂H))

        return NoTangent(), ∂basis
    end
    return scfres, self_consistent_field_pullback
end


# Ewald

# TODO reduce code duplication compared to primal
# TODO unspecialize from Zygote.dropgrad, Zygote.ignore -> ChainRulesCore.ignore
# TODO move shell_indices out of energy_ewald and just globally mark shell_indices 
#      as ChainRulesCore.@non_differentiable -> this should allow to delete _autodiff_energy_ewald
function _autodiff_energy_ewald(lattice, recip_lattice, charges, positions; η=nothing, args...)
    T = eltype(lattice)
    @assert T == eltype(recip_lattice)
    @assert length(charges) == length(positions)
    if η === nothing
        # Balance between reciprocal summation and real-space summation
        # with a slight bias towards reciprocal summation
        η = sqrt(sqrt(T(1.69) * norm(recip_lattice ./ 2T(π)) / norm(lattice))) / 2
    end

    #
    # Numerical cutoffs
    #
    # The largest argument to the exp(-x) function to obtain a numerically
    # meaningful contribution. The +5 is for safety.
    max_exponent = -log(eps(T)) + 5

    # The largest argument to the erfc function for various precisions.
    # To get an idea:
    #   erfc(5) ≈ 1e-12,  erfc(8) ≈ 1e-29,  erfc(10) ≈ 2e-45,  erfc(14) ≈ 3e-87
    max_erfc_arg = get(
        Dict(Float32 => 5, Float64 => 8, BigFloat => 14),
        T,
        something(findfirst(arg -> 100 * erfc(arg) < eps(T), 1:100), 100) # fallback for not yet implemented cutoffs
    )

    #
    # Reciprocal space sum
    #
    # Initialize reciprocal sum with correction term for charge neutrality
    sum_recip = - (sum(charges)^2 / 4η^2)

    # Function to return the indices corresponding
    # to a particular shell
    # TODO switch to an O(N) implementation
    function shell_indices(ish)
        [Vec3(i,j,k) for i in -ish:ish for j in -ish:ish for k in -ish:ish
        if maximum(abs.((i,j,k))) == ish]
    end

    # Loop over reciprocal-space shells
    gsh = 1 # Exclude G == 0
    any_term_contributes = true
    while any_term_contributes
        any_term_contributes = false

        # Compute G vectors and moduli squared for this shell patch
        for G in (Zygote.@ignore shell_indices(gsh))
            Gsq = sum(abs2, recip_lattice * Zygote.dropgrad(G))

            # Check if the Gaussian exponent is small enough
            # for this term to contribute to the reciprocal sum
            exponent = Gsq / 4η^2
            if exponent > max_exponent
                continue
            end

            cos_strucfac = sum(Z * cos(2T(π) * dot(r, Zygote.dropgrad(G))) for (r, Z) in zip(positions, charges))
            sin_strucfac = sum(Z * sin(2T(π) * dot(r, Zygote.dropgrad(G))) for (r, Z) in zip(positions, charges))
            sum_strucfac = cos_strucfac^2 + sin_strucfac^2

            any_term_contributes = true
            sum_recip += sum_strucfac * exp(-exponent) / Gsq

        end
        gsh += 1
    end
    # Amend sum_recip by proper scaling factors:
    sum_recip *= 4T(π) / abs(det(lattice))

    #
    # Real-space sum
    #
    # Initialize real-space sum with correction term for uniform background
    sum_real = -2η / sqrt(T(π)) * sum(Z -> Z^2, charges)

    # Loop over real-space shells
    rsh = 0 # Include R = 0
    any_term_contributes = true
    while any_term_contributes || rsh <= 1
        any_term_contributes = false

        # Loop over R vectors for this shell patch
        for R in (Zygote.@ignore shell_indices(rsh))
            for i = 1:length(positions), j = 1:length(positions)
                ti = positions[i]
                Zi = charges[i]
                tj = positions[j]
                Zj = charges[j]

                # Avoid self-interaction
                if rsh == 0 && ti == tj
                    continue
                end

                dist = norm(lattice * (ti - tj - Zygote.dropgrad(R)))

                # erfc decays very quickly, so cut off at some point
                if η * dist > max_erfc_arg
                    continue
                end

                any_term_contributes = true
                sum_real += Zi * Zj * erfc(η * dist) / dist
            end # i,j
        end # R
        rsh += 1
    end
    energy = (sum_recip + sum_real) / 2  # Divide by 2 (because of double counting)
    energy
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(energy_ewald), lattice, recip_lattice, charges, positions; args...)
    @warn "simplified energy_ewald rrule triggered."
    energy = energy_ewald(lattice, recip_lattice, charges, positions; args...)
    f(lattice, recip_lattice, charges, positions) = _autodiff_energy_ewald(lattice, recip_lattice, charges, positions; args...)
    _energy, ewald_pullback = rrule_via_ad(config, f, lattice, recip_lattice, charges, positions)
    energy, ewald_pullback
end

