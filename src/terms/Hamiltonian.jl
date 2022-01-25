### A Hamiltonian is composed of blocks (kpoints), which have a list of RealFourierOperator
# corresponding to each term
# This is the "high-level" interface, provided for convenience

abstract type HamiltonianBlock end

# Generic HamiltonianBlock
struct GenericHamiltonianBlock <: HamiltonianBlock
    basis::PlaneWaveBasis
    kpoint::Kpoint

    # The operators are vectors of RealFourierOperator,
    # not typed because of type invariance issues.
    operators::Vector  # the original list of RealFourierOperator
                       # (as many as there are terms), kept for easier exploration
    optimized_operators::Vector  # Optimized list of RealFourierOperator, for application
end

# More optimized HamiltonianBlock for the important case of a DFT Hamiltonian
struct DftHamiltonianBlock <: HamiltonianBlock
    basis::PlaneWaveBasis
    kpoint::Kpoint
    operators::Vector

    # Individual operators for easy access
    fourier_op::FourierMultiplication
    real_op::RealSpaceMultiplication
    nonlocal_op::NonlocalOperator
    divAgrad_op::Union{Nothing,DivAgradOperator}

    scratch  # Pre-allocated scratch arrays for fast application
end

function HamiltonianBlock(basis, kpoint, operators, scratch=ham_allocate_scratch_(basis))
    optimized_operators = optimize_operators_(operators)
    fourier_ops  = filter(o -> o isa FourierMultiplication,   optimized_operators)
    real_ops     = filter(o -> o isa RealSpaceMultiplication, optimized_operators)
    nonlocal_ops = filter(o -> o isa NonlocalOperator,        optimized_operators)
    divAgrid_ops = filter(o -> o isa DivAgradOperator,        optimized_operators)

    is_dft_ham = (   length(fourier_ops) == length(real_ops) == length(nonlocal_ops) == 1
                     && length(divAgrid_ops) < 2)
    if is_dft_ham
        divAgrid_op = isempty(divAgrid_ops) ? nothing : only(divAgrid_ops)
        DftHamiltonianBlock(basis, kpoint, operators,
                            only(fourier_ops), only(real_ops), only(nonlocal_ops),
                            divAgrid_op, scratch)
    else
        GenericHamiltonianBlock(basis, kpoint, operators, optimized_operators)
    end
end
function ham_allocate_scratch_(basis::PlaneWaveBasis{T}) where {T}
    (ψ_reals=[zeros(complex(T), basis.fft_size...) for _ = 1:Threads.nthreads()], )
end

Base.:*(H::HamiltonianBlock, ψ) = mul!(similar(ψ), H, ψ)
Base.eltype(block::HamiltonianBlock) = complex(eltype(block.basis))
Base.size(block::HamiltonianBlock, i::Integer) = i < 3 ? size(block)[i] : 1
function Base.size(block::HamiltonianBlock)
    n_G = length(G_vectors(block.basis, block.kpoint))
    (n_G, n_G)
end

import Base: Matrix, Array
Array(block::HamiltonianBlock)  = Matrix(block)
Matrix(block::GenericHamiltonianBlock) = sum(Matrix, block.optimized_operators)
function Matrix(block::DftHamiltonianBlock)
    base = Matrix(block.fourier_op) .+ Matrix(block.real_op) .+ Matrix(block.nonlocal_op)
    isnothing(block.divAgrad_op) ? base : base .+ Matrix(block.divAgrad_op)
end

struct Hamiltonian
    basis::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock}
end

function LinearAlgebra.mul!(Hψ, H::Hamiltonian, ψ)
    for ik = 1:length(H.basis.kpoints)
        mul!(Hψ[ik], H.blocks[ik], ψ[ik])
    end
    Hψ
end
# need `deepcopy` here to copy the elements of the array of arrays ψ (not just pointers)
Base.:*(H::Hamiltonian, ψ) = mul!(deepcopy(ψ), H, ψ)

# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
# For the common DftHamiltonianBlock there is an optimized version below
@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::HamiltonianBlock,
                                                                        ψ::AbstractArray)
    T = eltype(H.basis)
    n_bands = size(ψ, 2)
    Hψ_fourier = similar(Hψ[:, 1])
    ψ_real  = zeros(complex(T), H.basis.fft_size...)
    Hψ_real = zeros(complex(T), H.basis.fft_size...)

    # take ψi, IFFT it to ψ_real, apply each term to Hψ_fourier and Hψ_real, and add it to Hψ
    for iband = 1:n_bands
        Hψ_real .= 0
        Hψ_fourier .= 0
        G_to_r!(ψ_real, H.basis, H.kpoint, ψ[:, iband])
        for op in H.optimized_operators
            apply!((fourier=Hψ_fourier, real=Hψ_real),
                   op,
                   (fourier=ψ[:, iband], real=ψ_real))
        end
        Hψ[:, iband] .= Hψ_fourier
        r_to_G!(Hψ_fourier, H.basis, H.kpoint, Hψ_real)
        Hψ[:, iband] .+= Hψ_fourier
    end

    Hψ
end

# Fast version, specialized on DFT models. Minimizes the number of FFTs and allocations
@views @timing "DftHamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                           H::DftHamiltonianBlock,
                                                                           ψ::AbstractArray)
    n_bands = size(ψ, 2)
    have_divAgrad = !isnothing(H.divAgrad_op)

    potential = H.real_op.potential
    potential /= prod(H.basis.fft_size)  # because we use unnormalized plans for extra speed
    @timing "kinetic+local$(have_divAgrad ? "+divAgrad" : "")" begin
        Threads.@threads for iband = 1:n_bands
            tid = Threads.threadid()
            ψ_real = H.scratch.ψ_reals[tid]

            G_to_r!(ψ_real, H.basis, H.kpoint, ψ[:, iband]; normalize=false)
            ψ_real .*= potential
            r_to_G!(Hψ[:, iband], H.basis, H.kpoint, ψ_real; normalize=false)  # overwrites ψ_real
            Hψ[:, iband] .+= H.fourier_op.multiplier .* ψ[:, iband]

            if !isnothing(H.divAgrad_op)
                apply!((fourier=Hψ[:, iband], real=nothing),
                       H.divAgrad_op,
                       (fourier=ψ[:, iband], real=nothing),
                       ψ_real)  # ψ_real used as scratch
            end
        end
    end

    # Apply the nonlocal operator
    @timing "nonlocal" begin
        apply!((fourier=Hψ, real=nothing), H.nonlocal_op, (fourier=ψ, real=nothing))
    end

    Hψ
end


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

    H = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks)
                            for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    E = Energies(basis.model.term_types, energies)
    (E=E, H=H)
end
function Hamiltonian(basis::PlaneWaveBasis; ψ=nothing, occ=nothing, kwargs...)
    _, H = energy_hamiltonian(basis, ψ, occ; kwargs...)
    H
end

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
