### A Hamiltonian is composed of blocks (kpoints), which have a list of RealFourierOperator
# corresponding to each term
# This is the "high-level" interface, provided for convenience

struct HamiltonianBlock
    basis::PlaneWaveBasis
    kpoint::Kpoint

    # The operators are vectors of RealFourierOperator,
    # not typed because of type invariance issues.
    operators::Vector  # the original list of RealFourierOperator
                       # (as many as there are terms), kept for easier exploration
    optimized_operators::Vector  # the optimized list of RealFourierOperator, to be used for applying
    scratch  # Pre-allocated scratch arrays for fast application
end
function HamiltonianBlock(basis, kpt, operators, scratch)
    HamiltonianBlock(basis, kpt, operators, optimize_operators_(operators), scratch)
end
Base.eltype(block::HamiltonianBlock) = complex(eltype(block.basis))
Base.size(block::HamiltonianBlock, i::Integer) = i < 3 ? size(block)[i] : 1
function Base.size(block::HamiltonianBlock)
    n_G = length(G_vectors(block.basis, block.kpoint))
    (n_G, n_G)
end

struct Hamiltonian
    basis::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock}
end


# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
# This is a fallback function that works for all hamiltonians;
# for "usual cases", there is a faster implementation below
@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::HamiltonianBlock,
                                                                        ψ::AbstractArray)
    # Special-case of DFT Hamiltonian: go to a fast path
    fh = fast_hblock(H)
    fh !== nothing && return fast_hblock_mul!(Hψ, fh, ψ)

    basis = H.basis
    T = eltype(basis)
    kpt = H.kpoint
    nband = size(ψ, 2)

    Hψ_fourier = similar(Hψ[:, 1])
    ψ_real = zeros(complex(T), basis.fft_size...)
    Hψ_real = zeros(complex(T), basis.fft_size...)

    # take ψi, IFFT it to ψ_real, apply each term to Hψ_fourier and Hψ_real, and add it to Hψ
    for iband = 1:nband
        Hψ_real .= 0
        Hψ_fourier .= 0
        G_to_r!(ψ_real, basis, kpt, ψ[:, iband])
        for op in H.optimized_operators
            apply!((fourier=Hψ_fourier, real=Hψ_real),
                   op,
                   (fourier=ψ[:, iband], real=ψ_real))
        end
        Hψ[:, iband] .= Hψ_fourier
        r_to_G!(Hψ_fourier, basis, kpt, Hψ_real)
        Hψ[:, iband] .+= Hψ_fourier
    end

    Hψ
end
Base.:*(H::HamiltonianBlock, ψ) = mul!(similar(ψ), H, ψ)

# Returns a fast path hamiltonian if eligible, nothing if not
function fast_hblock(H::HamiltonianBlock)
    length(H.optimized_operators) == 3 || return nothing
    fourier_ops = filter(o -> o isa FourierMultiplication, H.optimized_operators)
    real_ops = filter(o -> o isa RealSpaceMultiplication, H.optimized_operators)
    nonlocal_ops = filter(o -> o isa NonlocalOperator, H.optimized_operators)
    (length(fourier_ops) == length(real_ops) == length(nonlocal_ops) == 1) || return nothing
    (fourier_op=only(fourier_ops), real_op=only(real_ops), nonlocal_op=only(nonlocal_ops), H=H)
end

# Fast version, specialized on DFT models with just 3 operators: real, fourier and nonlocal
# Minimizes the number of allocations
@views function fast_hblock_mul!(Hψ::AbstractArray,
                                 fast_hblock::NamedTuple,
                                 ψ::AbstractArray)
    H = fast_hblock.H
    basis = H.basis
    kpt = H.kpoint
    nband = size(ψ, 2)

    potential = fast_hblock.real_op.potential
    potential /= prod(basis.fft_size)  # because we use unnormalized plans
    @timing "kinetic+local" begin
        Threads.@threads for iband = 1:nband
            tid = Threads.threadid()
            ψ_real = H.scratch.ψ_reals[tid]

            G_to_r!(ψ_real, basis, kpt, ψ[:, iband]; normalize=false)
            ψ_real .*= potential
            r_to_G!(Hψ[:, iband], basis, kpt, ψ_real; normalize=false)  # overwrites ψ_real
            Hψ[:, iband] .+= fast_hblock.fourier_op.multiplier .* ψ[:, iband]
        end
    end

    # Apply the nonlocal operator
    @timing "nonlocal" begin
        apply!((fourier=Hψ, real=nothing), fast_hblock.nonlocal_op, (fourier=ψ, real=nothing))
    end

    Hψ
end

function LinearAlgebra.mul!(Hψ, H::Hamiltonian, ψ)
    for ik = 1:length(H.basis.kpoints)
        mul!(Hψ[ik], H.blocks[ik], ψ[ik])
    end
    Hψ
end
# need `deepcopy` here to copy the elements of the array of arrays ψ (not just pointers)
Base.:*(H::Hamiltonian, ψ) = mul!(deepcopy(ψ), H, ψ)

# Get energies and Hamiltonian
# kwargs is additional info that might be useful for the energy terms to precompute
# (eg the density ρ)
@timing function energy_hamiltonian(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    # it: index into terms, ik: index into kpoints
    @timing "ene_ops" ene_ops_arr = [ene_ops(term, basis, ψ, occ; kwargs...)
                                     for term in basis.terms]
    energies  = [eh.E for eh in ene_ops_arr]
    operators = [eh.ops for eh in ene_ops_arr]         # operators[it][ik]

    # flatten the inner arrays in case a term returns more than one operator
    function flatten(arr)
        ret = []
        for a in arr
            if a isa RealFourierOperator
                push!(ret, a)
            else
                push!(ret, a...)
            end
        end
        ret
    end
    hks_per_k   = [flatten([blocks[ik] for blocks in operators])
                   for ik = 1:length(basis.kpoints)]      # hks_per_k[ik][it]

    # Preallocated scratch arrays
    T = eltype(basis)
    scratch = (
        ψ_reals=[zeros(complex(T), basis.fft_size...) for tid = 1:Threads.nthreads()],
    )

    H = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks, scratch)
                            for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    E = Energies(basis.model.term_types, energies)
    (E=E, H=H)
end
function Hamiltonian(basis::PlaneWaveBasis; ψ=nothing, occ=nothing, kwargs...)
    _, H = energy_hamiltonian(basis, ψ, occ; kwargs...)
    H
end

import Base: Matrix, Array
function Matrix(block::HamiltonianBlock)
    sum(Matrix(h) for h in block.optimized_operators)
end
Array(block::HamiltonianBlock) = Matrix(block)

"""
Get the total local potential of the given Hamiltonian, in real space
in the spin components.
"""
function total_local_potential(ham::Hamiltonian)
    n_spin = ham.basis.model.n_spin_components
    pots = map(1:n_spin) do σ
        # Get the first Hamiltonian block of this spin component
        # (works since all local potentials are the same)
        block = ham.blocks[first(krange_spin(ham.basis, σ))]
        rs = [o for o in block.optimized_operators if o isa RealSpaceMultiplication]
        only(rs).potential
    end
    cat(pots..., dims=4)
end

"""
Returns a new Hamiltonian with local potential replaced by the given one
"""
function hamiltonian_with_total_potential(ham::Hamiltonian, V)
    @assert size(V, 4) == ham.basis.model.n_spin_components
    newblocks = [hamiltonian_with_total_potential(Hk, V[:, :, :, Hk.kpoint.spin])
                 for Hk in ham.blocks]
    Hamiltonian(ham.basis, newblocks)
end
function hamiltonian_with_total_potential(Hk::HamiltonianBlock, V)
    operators = [op for op in Hk.operators if !(op isa RealSpaceMultiplication)]
    push!(operators, RealSpaceMultiplication(Hk.basis, Hk.kpoint, V))
    HamiltonianBlock(Hk.basis, Hk.kpoint, operators, Hk.scratch)
end
