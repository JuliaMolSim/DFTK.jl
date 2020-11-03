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
Base.size(block::HamiltonianBlock) =
    (length(G_vectors(block.kpoint)), length(G_vectors(block.kpoint)))
Base.size(block::HamiltonianBlock, i::Integer) = i < 3 ? size(block)[i] : 1

struct Hamiltonian
    basis::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock}
end


# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::HamiltonianBlock,
                                                                        ψ::AbstractArray)
    # Special-case of DFT Hamiltonian: go to a fast path
    if length(H.optimized_operators) == 3 &&
        any(o -> o isa FourierMultiplication, H.optimized_operators) &&
        any(o -> o isa RealSpaceMultiplication, H.optimized_operators) &&
        any(o -> o isa NonlocalOperator, H.optimized_operators)
        return mul!_fast(Hψ, H, ψ)
    end

    basis = H.basis
    kpt = H.kpoint
    nband = size(ψ, 2)

    # Allocate another scratch array:
    Hψ_fouriers = [similar(Hψ[:, 1]) for tid = 1:Threads.nthreads()]

    # take ψi, IFFT it to ψ_real, apply each term to Hψ_temp and Hψ_real, and add it to Hψi
    Threads.@threads for iband = 1:nband
        tid = Threads.threadid()
        ψ_real = H.scratch.ψ_reals[tid]
        Hψ_real = H.scratch.Hψ_reals[tid]
        Hψ_fourier = Hψ_fouriers[tid]

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

# Fast version, specialized on DFT models with just 3 operators: real, fourier and nonlocal
# Minimizes the number of allocations
@views function mul!_fast(Hψ::AbstractArray,
                          H::HamiltonianBlock,
                          ψ::AbstractArray)
    fourier_op = only(filter(o -> o isa FourierMultiplication, H.optimized_operators))
    real_op = only(filter(o -> o isa RealSpaceMultiplication, H.optimized_operators))
    nonlocal_op = only(filter(o -> o isa NonlocalOperator, H.optimized_operators))

    basis = H.basis
    kpt = H.kpoint
    nband = size(ψ, 2)

    @timing "kinetic+local" begin
    Threads.@threads for iband = 1:nband
        tid = Threads.threadid()
        ψ_real = H.scratch.ψ_reals[tid]
        Hψ_real = H.scratch.Hψ_reals[tid]

        G_to_r!(ψ_real, basis, kpt, ψ[:, iband])
        ψ_real .*= real_op.potential
        r_to_G!(Hψ[:, iband], basis, kpt, ψ_real)  # overwrites ψ_real
        Hψ[:, iband] .+= fourier_op.multiplier .* ψ[:, iband]
    end
    end

    # Apply the nonlocal operator
    @timing "nonlocal" begin
        apply!((fourier=Hψ, real=nothing), nonlocal_op, (fourier=ψ, real=nothing))
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
    @timing "ene_ops" ene_ops_arr = [ene_ops(term, ψ, occ; kwargs...) for term in basis.terms]
    energies    = [eh.E for eh in ene_ops_arr]
    operators   = [eh.ops for eh in ene_ops_arr]         # operators[it][ik]

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
        Hψ_reals=[zeros(complex(T), basis.fft_size...) for tid = 1:Threads.nthreads()]
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
Get the total local potential of the given Hamiltonian, in real space.
"""
function total_local_potential(ham::Hamiltonian)
    @assert ham.basis.model.spin_polarization in (:none, :spinless)
    block = ham.blocks[1]  # all local potentials are the same
    rs = [o for o in block.optimized_operators if o isa RealSpaceMultiplication]
    only(rs).potential
end

"""
Returns a new Hamiltonian with local potential replaced by the given one
"""
function hamiltonian_with_total_potential(ham::Hamiltonian, V)
    @assert ham.basis.model.spin_polarization in (:none, :spinless)
    Hamiltonian(ham.basis, hamiltonian_with_total_potential.(ham.blocks, Ref(V)))
end
function hamiltonian_with_total_potential(Hk::HamiltonianBlock, V)
    @assert Hk.basis.model.spin_polarization in (:none, :spinless)
    operators = [op for op in Hk.operators if !(op isa RealSpaceMultiplication)]
    push!(operators, RealSpaceMultiplication(Hk.basis, Hk.kpoint, V))
    HamiltonianBlock(Hk.basis, Hk.kpoint, operators, Hk.scratch)
end
