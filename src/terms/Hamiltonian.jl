### A Hamiltonian is composed of blocks (kpoints), which have a list of RFO corresponding to each term
# This is the "high-level" interface, provided for convenience

struct HamiltonianBlock
    basis::PlaneWaveBasis
    kpoint::Kpoint
    # The Hams are vectors of RealFourierOperator, not typed because of type invariance issues.
    operators::Vector  # the original list of RealFourierOperator
                       # (as many as there are terms), kept for easier exploration
    optimized_operators::Vector  # the optimized list of RealFourierOperator, to be used for applying
    HamiltonianBlock(basis, kpt, operators) = new(basis, kpt, operators, optimize_operators_(operators))
end
struct Hamiltonian
    basis::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock}
end

# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
# Do this band by band to conserve memory
# As an optimization we special-case nonlocal operators to apply them
# instead on the full block and benefit from BLAS3
@views function LinearAlgebra.mul!(Hψ::AbstractArray, H::HamiltonianBlock, ψ::AbstractArray)
    basis = H.basis
    kpt = H.kpoint
    nband = size(ψ, 2)
    # buffer arrays
    ψ_reals = [similar(ψ[:, 1], basis.fft_size...) for tid = 1:Threads.nthreads()]
    Hψ_reals = [similar(ψ[:, 1], basis.fft_size...) for tid = 1:Threads.nthreads()]
    Hψ_fouriers = [similar(Hψ[:, 1]) for tid = 1:Threads.nthreads()]

    # take ψi, IFFT it to ψ_real, apply each term to Hψ_temp and Hψ_real, and add it to Hψi
    Threads.@threads for iband = 1:nband
        tid = Threads.threadid()
        ψ_real = ψ_reals[tid]
        Hψ_real = Hψ_reals[tid]
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
# kwargs is additional info that might be useful for the energy terms to precompute (eg the density ρ)
function energy_hamiltonian(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    nterms = length(basis.terms)
    ene_ops_arr = [ene_ops(term, ψ, occ; kwargs...) for term in basis.terms] # ene_hams[it].hams[ik]
    energies = [eh.E for eh in ene_ops_arr] # energies[it]
    ops = [[ene_ops_arr[it].ops[ik] for it = 1:nterms]
            for ik = 1:length(basis.kpoints)] # hams[ik][it]
    H = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks)
                            for (hks, kpt) in zip(ops, basis.kpoints)])
    (E=Energies(basis, energies), H=H)
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
