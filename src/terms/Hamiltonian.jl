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
# Do this band by band to conserve memory
# As an optimization we special-case nonlocal operators to apply them
# instead on the full block and benefit from BLAS3
@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::HamiltonianBlock,
                                                                        ψ::AbstractArray)
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
            if !(op isa NonlocalOperator)
                apply!((fourier=Hψ_fourier, real=Hψ_real),
                       op,
                       (fourier=ψ[:, iband], real=ψ_real))
            end
        end
        Hψ[:, iband] .= Hψ_fourier
        r_to_G!(Hψ_fourier, basis, kpt, Hψ_real)
        Hψ[:, iband] .+= Hψ_fourier
    end

    # Apply the nonlocal operators
    for op in H.optimized_operators
        if op isa NonlocalOperator
            apply!((fourier=Hψ, real=nothing), op, (fourier=ψ, real=nothing))
        end
    end

    Hψ
end
Base.:*(H::HamiltonianBlock, ψ) = mul!(similar(ψ), H, ψ)

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
    ene_ops_arr = [ene_ops(term, ψ, occ; kwargs...) for term in basis.terms]
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
    @assert length(rs) == 1
    rs[1].potential
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
