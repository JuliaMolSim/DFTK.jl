using ChainRulesCore
using Infiltrator

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

ChainRulesCore.@non_differentiable r_vectors(::Any...)
ChainRulesCore.@non_differentiable G_vectors(::Any...)
ChainRulesCore.@non_differentiable default_n_electrons(::Any...)
ChainRulesCore.@non_differentiable default_symmetries(::Any...) # TODO perhaps?
ChainRulesCore.@non_differentiable build_kpoints(::Any...)

# https://github.com/doddgray/OptiMode.jl/blob/main/src/grad_lib/StaticArrays.jl
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )

# simplified version of PlaneWaveBasis outer constructor to
# help reverse mode AD to only differentiate the relevant computations.
# this excludes assertions (try-catch), MPI handling, and other things
function _autodiff_PlaneWaveBasis_namedtuple(model::Model{T, VT}, basis::PlaneWaveBasis) where {T <: Real, VT <: Real}
    dvol = model.unit_cell_volume ./ prod(basis.fft_size)

    # TODO new volumes (and more)

    G_to_r_normalization = 1 / sqrt(model.unit_cell_volume)
    r_to_G_normalization = sqrt(model.unit_cell_volume) / length(basis.ipFFT)

    # Create dummy terms array for _basis to handle
    terms = Vector{Any}(undef, length(model.term_types))

    # cicularity is getting complicated...
    # To correctly instantiate term types, we do need a full PlaneWaveBasis struct;
    # so we need to interleave re-computed differentiable params, and fixed params in basis
    _basis = PlaneWaveBasis{T,VT}( # this shouldn't hit the rrule below a second time due to more args
        model, basis.fft_size, dvol,
        basis.Ecut, basis.variational,
        basis.opFFT, basis.ipFFT, basis.opBFFT, basis.ipBFFT,
        r_to_G_normalization, G_to_r_normalization,
        basis.G_vectors, basis.r_vectors,
        basis.kpoints, basis.kweights, basis.kgrid, basis.kshift,
        basis.kcoords_global, basis.kweights_global, basis.comm_kpts, basis.krange_thisproc, basis.krange_allprocs,
        basis.symmetries, basis.symmetries_respect_rgrid, terms)

    # terms = Any[t(_basis) for t in model.term_types]
    terms = vcat([], [t(_basis) for t in model.term_types]) # hack: enforce Vector{Any} without causing reverse mutation

    (; model=model, dvol=dvol, terms=terms, G_to_r_normalization=G_to_r_normalization, r_to_G_normalization=r_to_G_normalization)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{PlaneWaveBasis}, model::Model; kwargs...)
    @warn "simplified PlaneWaveBasis rrule triggered."
    basis = T(model; kwargs...)
    f(model) = _autodiff_PlaneWaveBasis_namedtuple(model, basis)
    _basis, PlaneWaveBasis_pullback = rrule_via_ad(config, f, model)
    return basis, PlaneWaveBasis_pullback
end

Base.zero(::ElementPsp) = NoTangent() # TODO

# TODO this has changed on master
function _autodiff_compute_density(basis::PlaneWaveBasis, ψ, occupation)
    # try one kpoint only (for debug) TODO re-enable all kpoints
    kpt = basis.kpoints[1]
    ψk = ψ[1]
    occk = occupation[1]
    ρ = sum(zip(eachcol(ψk), occk)) do (ψnk, occn)
        ψnk_real_tid = G_to_r(basis, kpt, ψnk)
        occn .* abs2.(ψnk_real_tid)
    end
    ρ = reshape(ρ, size(ρ)..., 1)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(compute_density), basis::PlaneWaveBasis, ψ, occupation)
    @warn "simplified compute_density rrule triggered."
    ρ = compute_density(basis, ψ, occupation)
    _ρ, compute_density_pullback = rrule_via_ad(config, _autodiff_compute_density, basis, ψ, occupation)
    return ρ, compute_density_pullback
end

# workaround to pass rrule_via_ad kwargs
DFTK.energy_hamiltonian(basis, ψ, occ, ρ) = DFTK.energy_hamiltonian(basis, ψ, occ; ρ=ρ)

# fast version
function _autodiff_hblock_mul(hblock::DftHamiltonianBlock, ψ)
    # a pure version of *(H::DftHamiltonianBlock, ψ)
    # TODO this currently only considers kinetic+local+nonlocal
    basis = hblock.basis
    kpt = hblock.kpoint

    potential = hblock.local_op.potential
    potential = potential / prod(basis.fft_size)  # because we use unnormalized plans
    fourier_op_multiplier = hblock.fourier_op.multiplier
    nonlocal_op = hblock.nonlocal_op

    function apply_H(ψk)
        ψ_real = G_to_r(basis, kpt, ψk) .* potential ./ basis.G_to_r_normalization
        Hψ_k = r_to_G(basis, kpt, ψ_real) ./ basis.r_to_G_normalization
        Hψ_k += fourier_op_multiplier .* ψk
        if !isnothing(nonlocal_op)
            Hψ_k += nonlocal_op.P * (nonlocal_op.D * (nonlocal_op.P' * ψk))
        end
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

function ChainRulesCore.rrule(TE::Type{Energies{T}}, energies) where T
    @warn "Energies{T} constructor rrule triggered."
    E = TE(energies)
    TE_pullback(∂E::AbstractZero) = NoTangent(), NoTangent()
    TE_pullback(∂E) = NoTangent(), ∂E.energies
    return E, TE_pullback
end

function ChainRulesCore.rrule(TH::Type{Hamiltonian}, basis, blocks)
    @warn "Hamiltonian constructor rrule triggered."
    H = TH(basis, blocks)
    TH_pullback(∂H::AbstractZero) = NoTangent(), NoTangent(), NoTangent()
    TH_pullback(∂H) = NoTangent(), ∂H.basis, ∂H.blocks
    return H, TH_pullback
end

function eigenvalues_rayleigh_ritz(ψ, Hψ)
    eigenvalues = map(zip(ψ, Hψ)) do (ψik, Hψik)
        F = eigen(Hermitian(ψik'Hψik))
        F.values
    end
    return eigenvalues
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(self_consistent_field), basis::PlaneWaveBasis{T,VT}; kwargs...) where {T,VT}
    @warn "self_consistent_field rrule triggered."
    scfres = self_consistent_field(basis; kwargs...)
    project_to_scfres = ProjectTo(scfres)

    (; E, H), energy_hamiltonian_pullback =
        rrule_via_ad(config, energy_hamiltonian, basis, scfres.ψ, scfres.occupation, scfres.ρ)
    Hψ, mul_pullback =
        rrule(config, *, H, scfres.ψ)
    ρ, compute_density_pullback =
        rrule(config, compute_density, basis, scfres.ψ, scfres.occupation)

    eigenvalues, eigenvalues_pullback = rrule_via_ad(config, eigenvalues_rayleigh_ritz, scfres.ψ, Hψ)

    function self_consistent_field_pullback(∂scfres)
        ∂scfres = project_to_scfres(∂scfres)
        ∂ψ = ∂scfres.ψ
        ∂occupation = ∂scfres.occupation
        ∂ρ = ∂scfres.ρ
        ∂energies = ∂scfres.energies
        ∂basis1 = Tangent{PlaneWaveBasis{T,VT}}(; ChainRulesCore.backing(∂scfres.basis)...)
        ∂H = ∂scfres.ham
        ∂eigenvalues = ∂scfres.eigenvalues

        _, ∂basis2, ∂ψ_density_pullback, _ = compute_density_pullback(∂ρ)
        ∂ψ += ∂ψ_density_pullback
        _, ∂ψ_rayleigh_ritz, ∂Hψ_rayleigh_ritz = eigenvalues_pullback(∂eigenvalues)
        ∂ψ += ∂ψ_rayleigh_ritz

        # Otherwise there is no contribution to ∂basis, by linearity.
        # This also excludes the case when ∂ψ is a NoTangent().
        if !iszero(∂ψ)
            occupation_threshold = scfres.occupation_threshold
            ∂Hψ = solve_ΩplusK_split(basis, scfres.ψ, -∂ψ, scfres.occupation; occupation_threshold).δψ # use self-adjointness of dH ψ -> dψ
            #∂Hψ += ∂Hψ_rayleigh_ritz
            _, ∂H_mul_pullback, _ = mul_pullback(∂Hψ)
            ∂H = ∂H_mul_pullback + ∂H
        end

        _, ∂basis3, _, _, _ = energy_hamiltonian_pullback((; E=∂energies, H=∂H))
        
        # ∂basis = ∂basis1 + ∂basis2 + ∂basis3
        ∂basis = ∂basis2 + ∂basis3

        return NoTangent(), ∂basis
    end
    return scfres, self_consistent_field_pullback
end


function _autodiff_LibxcDensities(basis, max_derivative::Integer, ρ, τ)
    model = basis.model
    @assert max_derivative in (0, 1, 2)

    n_spin    = model.n_spin_components
    σ_real    = nothing
    ∇ρ_real   = nothing
    Δρ_real   = nothing

    # compute ρ_real and possibly ρ_fourier
    ρ_real = permutedims(ρ, (4, 1, 2, 3))  # ρ[x, y, z, σ] -> ρ_real[σ, x, y, z]
    if max_derivative > 0
        ρf = r_to_G(basis, ρ)
        ρ_fourier = permutedims(ρf, (4, 1, 2, 3))  # ρ_fourier[σ, x, y, z]
    end

    # compute ∇ρ and σ
    if max_derivative > 0
        n_spin_σ = div((n_spin + 1) * n_spin, 2)
        ∇ρ_real = similar(ρ_real, n_spin, basis.fft_size..., 3)
        σ_real  = similar(ρ_real, n_spin_σ, basis.fft_size...)
        # ∇ρ_real ∈ n_spin x (N x N x N) x 3
        ∇ρ_real = 
        mapreduce((x...) -> cat(x...; dims=4), α for α in 1:3) do α
            iGα = [im * G[α] for G in G_vectors_cart(basis)]
            mapreduce((x...) -> cat(x...; dims=1), σ for σ = 1:n_spin) do σ
                G_to_r(basis, iGα .* @view ρ_fourier[σ, :, :, :])
            end
        end
        # restore leading dimension
        ∇ρ_real = reshape(∇ρ_real, (n_spin, basis.fft_size..., 3)) 
        
        #for α in 1:3
        #    iGα = [im * G[α] for G in G_vectors_cart(basis)]
        #    for σ = 1:n_spin
        #        ∇ρ_real[σ, :, :, :, α] .= G_to_r(basis, iGα .* @view ρ_fourier[σ, :, :, :])
        #    end
        #end

        tσ = DftFunctionals.spinindex_σ  # Spin index transformation (s, t) => st as expected by Libxc
        #σ_real .= 0
        @assert(n_spin == 1)
        σ_real = mapreduce(+, α for α in 1:3) do α
            ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[1, :, :, :, α]
        end
        σ_real = reshape(σ_real,(n_spin, basis.fft_size...))

        #σ_real .= 0
        #@views for α in 1:3
        #    σ_real[tσ(1, 1), :, :, :] .+= ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[1, :, :, :, α]
        #    if n_spin > 1
        #        σ_real[tσ(1, 2), :, :, :] .+= ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[2, :, :, :, α]
        #        σ_real[tσ(2, 2), :, :, :] .+= ∇ρ_real[2, :, :, :, α] .* ∇ρ_real[2, :, :, :, α]
        #    end
        #end
    end

    # Compute Δρ (currently not used apparently)
    if max_derivative > 1
        Δρ_real = similar(ρ_real, n_spin, basis.fft_size...)
        mG² = [-sum(abs2, G) for G in G_vectors_cart(basis)]
        for σ = 1:n_spin
            Δρ_real[σ, :, :, :] .= G_to_r(basis, mG² .* @view ρ_fourier[σ, :, :, :])
        end
    end

    # τ[x, y, z, σ] -> τ_Libxc[σ, x, y, z]
    τ_Libxc = isnothing(τ) ? nothing : permutedims(τ, (4, 1, 2, 3))
    LibxcDensities(basis, max_derivative, ρ_real, ∇ρ_real, σ_real, Δρ_real, τ_Libxc)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(LibxcDensities), basis, max_derivative::Integer, ρ, τ)
    @warn "LibxcDensities rrule triggered."
    rrule_via_ad(config, _autodiff_LibxcDensities, basis, max_derivative, ρ, τ)
end

# phases to differentiate
# 1. setup (build model, basis, ...)
# 2. scf
# 3. ...


# challenges:
# - foreign code (FFTW)
# - mutation
# - try-catch

# 1. direct rrules (r_to_G, ...)
# 2. alternative primal (rrule_via_ad)
# 3. SCF rrule

# TODO small
# [x] upstream @non_differentiable allunique(::Any) to ChainRules.jl
# [x] add basis contributions in SCF rrule (Tangent{Any} problem)
# [x] pull master (after Michael's atoms update)
# [x] remove Model alternative primal (or update with inv_lattice, ...)
# [x] replace OrderedDict in Energies with Vector{Pair{String,T}}
# [x] recompute rayleigh-ritz for ∂eigenvalues pullback

# TODO medium-large
# - generic HamiltonianBlock ? (low prio)
# - multiple kpoints
# - symmetries
# - more efficient compute_density rrule (probably by hand)
# - mpi
# - make forces differentiable
