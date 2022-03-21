using ChainRulesCore
using Zygote: @adjoint  # TODO remove, once ChainRules rrules can overrule Zygote
import AbstractFFTs


function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, f_real::AbstractArray)
    @warn "r_to_G rrule triggered."
    f_fourier = r_to_G(basis, f_real)
    function r_to_G_pullback(∂f_fourier)
        ∂f_real = G_to_r(basis, complex(∂f_fourier)) * basis.r_to_G_normalization / basis.G_to_r_normalization
        ∂normalization = real(dot(∂f_fourier, f_fourier)) / basis.r_to_G_normalization
        ∂basis = Tangent{typeof(basis)}(;r_to_G_normalization=∂normalization)
        return NoTangent(), ∂basis, real(∂f_real)
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray)
    @warn "r_to_G kpoint rrule triggered."
    f_fourier = r_to_G(basis, kpt, f_real)
    function r_to_G_pullback(∂f_fourier)
        ∂f_real = G_to_r(basis, kpt, complex(∂f_fourier)) * basis.r_to_G_normalization / basis.G_to_r_normalization
        ∂normalization = real(dot(∂f_fourier, f_fourier)) / basis.r_to_G_normalization
        ∂basis = Tangent{typeof(basis)}(;r_to_G_normalization=∂normalization)
        return NoTangent(), ∂basis, NoTangent(), ∂f_real
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, f_fourier::AbstractArray; kwargs...)
    @warn "G_to_r rrule triggered."
    f_real = G_to_r(basis, f_fourier; kwargs...)
    function G_to_r_pullback(∂f_real)
        ∂f_fourier = r_to_G(basis, real(∂f_real)) * basis.G_to_r_normalization / basis.r_to_G_normalization
        ∂normalization = real(dot(∂f_real, f_real)) / basis.G_to_r_normalization
        ∂basis = Tangent{typeof(basis)}(;G_to_r_normalization=∂normalization)
        return NoTangent(), ∂basis, ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractVector)
    @warn "G_to_r kpoint rrule triggered."
    f_real = G_to_r(basis, kpt, f_fourier)
    function G_to_r_pullback(∂f_real)
        ∂f_fourier = r_to_G(basis, kpt, complex(∂f_real)) * basis.G_to_r_normalization / basis.r_to_G_normalization
        ∂normalization = real(dot(∂f_real, f_real)) / basis.G_to_r_normalization
        ∂basis = Tangent{typeof(basis)}(;G_to_r_normalization=∂normalization)
        return NoTangent(), ∂basis, NoTangent(), ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

# workaround rrules for mpi: treat as noop
function ChainRulesCore.rrule(::typeof(mpi_sum), arr, comm)
    @warn "mpi_sum (ignored) rrule triggered."
    function mpi_sum_pullback(∂y)
        return NoTangent(), ∂y, NoTangent()
    end
    return arr, mpi_sum_pullback
end

ChainRulesCore.@non_differentiable ElementPsp(::Any...)
ChainRulesCore.@non_differentiable r_vectors(::Any...)
ChainRulesCore.@non_differentiable G_vectors(::Any...)
ChainRulesCore.@non_differentiable default_symmetries(::Any...) # TODO perhaps?
ChainRulesCore.@non_differentiable shell_indices(::Any)  # Ewald
ChainRulesCore.@non_differentiable cond(::Any)
ChainRulesCore.@non_differentiable isempty(::Any)

# TODO delete
@adjoint (T::Type{<:SArray})(x...) = T(x...), y->(y,)

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
    function build_kpoints_pullback(∂kpoints)
        sum_∂kpoints = sum(∂kpoints)
        if sum_∂kpoints isa NoTangent
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        ∂recip_lattice = sum([∂kp.coordinate_cart * kp.coordinate' for (kp, ∂kp) in zip(kpoints, ∂kpoints) if !(∂kp isa NoTangent)])
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
    function PT_pullback(∂basis)
        return (NoTangent(), ∂basis.model, NoTangent(), ∂basis.dvol, ∂basis.Ecut, 
                NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                ∂basis.r_to_G_normalization, ∂basis.G_to_r_normalization, ∂basis.kpoints, ∂basis.kweights, ∂basis.ksymops,
                ∂basis.kgrid, ∂basis.kshift, ∂basis.kcoords_global, ∂basis.ksymops_global, ∂basis.comm_kpts, 
                NoTangent(), NoTangent(), ∂basis.symmetries, ∂basis.terms)
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

function _autodiff_TermKinetic_namedtuple(basis, scaling_factor)
    kinetic_energies = [[scaling_factor * sum(abs2, G + kpt.coordinate_cart) / 2
                         for G in Gplusk_vectors_cart(basis, kpt)]
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
    poisson_green_coeffs = map(G_vectors_cart(basis)) do G
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
    function lowpass_for_symmetry_pullback(∂ρ)
        ∂ρ = _lowpass_for_symmetry(∂ρ, basis; symmetries)
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

# fast version
function _autodiff_hblock_mul(hblock::DftHamiltonianBlock, ψ)
    # a pure version of *(H::DftHamiltonianBlock, ψ)
    # TODO this currently only considers kinetic+local
    basis = hblock.basis
    kpt = hblock.kpoint

    potential = hblock.local_op.potential
    potential = potential / prod(basis.fft_size)  # because we use unnormalized plans
    fourier_op_multiplier = hblock.fourier_op.multiplier

    function apply_H(ψk)
        ψ_real = G_to_r(basis, kpt, ψk) .* potential ./ basis.G_to_r_normalization
        Hψ_k = r_to_G(basis, kpt, ψ_real) ./ basis.r_to_G_normalization
        Hψ_k += fourier_op_multiplier .* ψk
        reshape(Hψ_k, :, size(Hψ_k, 2)) # if Hψ_k a vector, promote to matrix
    end
    Hψ = mapreduce(apply_H, hcat, eachcol(ψ))
    Hψ
end

# slow fallback version
function _autodiff_hblock_mul(hblock::GenericHamiltonianBlock, ψ)
    basis = hblock.basis
    T = eltype(basis)
    kpt = hblock.kpoint

    function apply_H(ψk)
        ψ_real = G_to_r(basis, kpt, ψk)
        Hψ_fourier = zero(ψ[:, 1])
        Hψ_real = zeros(complex(T), basis.fft_size...)
        for op in hblock.optimized_operators
            # is the speedup in forward really worth the effort?
            if op isa RealSpaceMultiplication
                Hψ_real += op.potential .* ψ_real
            elseif op isa FourierMultiplication
                Hψ_fourier += op.multiplier .* ψk
            elseif op isa NonlocalOperator
                Hψ_fourier += op.P * (op.D * (op.P' * ψk))
            else
                @error "op not reversed"
            end
        end
        Hψ_k = Hψ_fourier + r_to_G(basis, kpt, Hψ_real)
        reshape(Hψ_k, :, size(Hψ_k, 2)) # if Hψ_k a vector, promote to matrix
    end
    Hψ = mapreduce(apply_H, hcat, eachcol(ψ))
    Hψ
end

# a pure version of *(H::Hamiltonian, ψ)
function _autodiff_apply_hamiltonian(H::Hamiltonian, ψ)
    return [_autodiff_hblock_mul(hblock, ψk) for (hblock, ψk) in zip(H.blocks, ψ)]
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

    function self_consistent_field_pullback(∂scfres)
        ∂ψ = ∂scfres.ψ
        ∂occupation = ∂scfres.occupation
        ∂ρ = ∂scfres.ρ
        ∂energies = ∂scfres.energies
        ∂basis = ∂scfres.basis
        ∂H = ∂scfres.ham

        _, ∂basis, ∂ψ_density_pullback, _ = compute_density_pullback(∂ρ)
        ∂ψ = ∂ψ_density_pullback + ∂ψ
        ∂ψ, occupation = DFTK.select_occupied_orbitals(basis, ∂ψ, occupation)

        ∂Hψ = solve_ΩplusK(basis, ψ, -∂ψ, occupation).δψ # use self-adjointness of dH ψ -> dψ

        # TODO need to do proj_tangent on ∂Hψ
        _, ∂H_mul_pullback, _ = mul_pullback(∂Hψ)
        ∂H = ∂H_mul_pullback + ∂H
        _, ∂basis, _, _, _ = energy_hamiltonian_pullback((∂energies, ∂H))

        return NoTangent(), ∂basis
    end
    return scfres, self_consistent_field_pullback
end
