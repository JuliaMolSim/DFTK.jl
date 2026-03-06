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

    scratch  # dummy field
end

"""A more optimized HamiltonianBlock for the important case of a DFT Hamiltonian."""
struct DftHamiltonianBlock <: HamiltonianBlock
    basis::PlaneWaveBasis
    kpoint::Kpoint
    operators::Vector

    # Individual operators for easy access
    fourier_op::FourierMultiplication
    local_op::RealSpaceMultiplication
    nonlocal_op::Union{Nothing,NonlocalOperator}
    divAgrad_op::Union{Nothing,DivAgradOperator}

    scratch  # Pre-allocated scratch arrays for fast application
end

function HamiltonianBlock(basis, kpoint, operators; scratch=nothing)
    if basis.model.spin_polarization == :full
        # Pad any 3D scalar potentials (like Hartree or Local) to 4D 
        # so they can be safely summed with the 4D non-collinear XC potential.
        for i in eachindex(operators)
            if operators[i] isa RealSpaceMultiplication
                V = operators[i].potential
                if ndims(V) == 3
                    V4 = zeros(eltype(V), size(V)..., 4)
                    V4[:, :, :, 1] .= V
                    operators[i] = RealSpaceMultiplication(basis, kpoint, V4)
                end
            end
        end
    end

    optimized_operators = optimize_operators(operators)
    fourier_ops  = filter(o -> o isa FourierMultiplication,   optimized_operators)
    real_ops     = filter(o -> o isa RealSpaceMultiplication, optimized_operators)
    nonlocal_ops = filter(o -> o isa NonlocalOperator,        optimized_operators)
    divAgrad_ops = filter(o -> o isa DivAgradOperator,        optimized_operators)

    n_ops_grouped = length(fourier_ops) + length(real_ops) + length(nonlocal_ops) + length(divAgrad_ops)
    is_dft_ham = (   length(fourier_ops) == 1 && length(real_ops) == 1
                  && length(nonlocal_ops) < 2 && length(divAgrad_ops) < 2
                  && n_ops_grouped == length(optimized_operators))
    if is_dft_ham
        scratch = @something scratch _ham_allocate_scratch(basis)
        nonlocal_op = isempty(nonlocal_ops) ? nothing : only(nonlocal_ops)
        divAgrad_op = isempty(divAgrad_ops) ? nothing : only(divAgrad_ops)
        DftHamiltonianBlock(basis, kpoint, operators,
                            only(fourier_ops), only(real_ops),
                            nonlocal_op, divAgrad_op, scratch)
    else
        GenericHamiltonianBlock(basis, kpoint, operators, optimized_operators, nothing)
    end
end
function _ham_allocate_scratch(basis::PlaneWaveBasis{T}) where {T}
    n_spin_reals = basis.model.spin_polarization == :full ? 2 : 1
    [(; ψ_reals=[zeros_like(G_vectors(basis), complex(T), basis.fft_size...) for _ = 1:n_spin_reals])
     for _ = 1:Threads.nthreads()]
end

Base.:*(H::HamiltonianBlock, ψ) = mul!(similar(ψ), H, ψ)
Base.eltype(block::HamiltonianBlock) = complex(eltype(block.basis))
Base.size(block::HamiltonianBlock, i::Integer) = i < 3 ? size(block)[i] : 1
function Base.size(block::HamiltonianBlock)
    n_G = length(G_vectors(block.basis, block.kpoint))
    n_spin = block.basis.model.spin_polarization == :full ? 2 : 1
    (n_G * n_spin, n_G * n_spin)
end
function random_orbitals(hamk::HamiltonianBlock, howmany::Integer)
    random_orbitals(hamk.basis, hamk.kpoint, howmany)
end

Base.Array(block::HamiltonianBlock)  = Matrix(block)
Base.Matrix(block::HamiltonianBlock) = sum(Matrix, block.operators)
Base.Matrix(block::GenericHamiltonianBlock) = sum(Matrix, block.optimized_operators)

"""Represents a matrix-free Hamiltonian discretized in a given plane-wave basis."""
struct Hamiltonian
    basis::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock}
end

Base.getindex(ham::Hamiltonian, index) = ham.blocks[index]

function LinearAlgebra.mul!(Hψ, H::Hamiltonian, ψ)
    for ik = 1:length(H.basis.kpoints)
        mul!(Hψ[ik], H.blocks[ik], ψ[ik])
    end
    Hψ
end
function Base.:*(H::Hamiltonian, ψ)
    # This allocates new memory for the result of promoted eltype
    result = one(eltype(H.basis)) * ψ
    mul!(result, H, ψ)
end

# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
# For the common DftHamiltonianBlock there is an optimized version below
@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::GenericHamiltonianBlock,
                                                                        ψ::AbstractArray)
    if H.basis.model.spin_polarization == :full
        error("GenericHamiltonianBlock not implemented for :full spin")
    end
    function allocate_local_storage()
        T = eltype(H.basis)
        (; Hψ_fourier = similar(Hψ[:, 1]),
           ψ_real  = similar(ψ, complex(T), H.basis.fft_size...),
           Hψ_real = similar(Hψ, complex(T), H.basis.fft_size...))
    end
    parallel_loop_over_range(1:size(ψ, 2); allocate_local_storage) do iband, storage
        to = TimerOutput()  # Thread-local timer output

        # Take ψi, IFFT it to ψ_real, apply each term to Hψ_fourier and Hψ_real, and add it
        # to Hψ.
        storage.Hψ_real .= 0
        storage.Hψ_fourier .= 0
        ifft!(storage.ψ_real, H.basis, H.kpoint, ψ[:, iband])
        for op in H.optimized_operators
            @timeit to "$(nameof(typeof(op)))" begin
                apply!((; fourier=storage.Hψ_fourier, real=storage.Hψ_real),
                       op,
                       (; fourier=ψ[:, iband], real=storage.ψ_real))
            end
        end
        Hψ[:, iband] .= storage.Hψ_fourier
        fft!(storage.Hψ_fourier, H.basis, H.kpoint, storage.Hψ_real)
        Hψ[:, iband] .+= storage.Hψ_fourier

        if Threads.threadid() == 1
            merge!(DFTK.timer, to; tree_point=[t.name for t in DFTK.timer.timer_stack])
        end
    end

    Hψ
end

# Fast version, specialized on DFT models. Minimizes the number of FFTs and allocations
@views @timing "DftHamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                           H::DftHamiltonianBlock,
                                                                           ψ::AbstractArray)
    n_bands = size(ψ, 2)
    iszero(n_bands) && return Hψ  # Nothing to do if ψ empty
    n_G = length(G_vectors(H.basis, H.kpoint))
    is_full = H.basis.model.spin_polarization == :full

    have_divAgrad = !isnothing(H.divAgrad_op)
    if have_divAgrad
        is_full && error("divAgrad_op not implemented for :full")
        # TODO: It is very beneficial to precompute G_plus_k here, rather than for each band.
        #       Extra performance could probably be gained by storing this in the HamiltonianBlock
        #       as a scratch array. Is it worth the complication and extra memory use?
        # Precompute G_plus_k for DivAgradOperator
        G_plus_k = [map(p -> p[α], Gplusk_vectors_cart(H.basis, H.kpoint)) for α = 1:3]
    end

    # Notice that we use unnormalized plans for extra speed
    potential = H.local_op.potential .* H.basis.fft_grid.fft_normalization .*
                H.basis.fft_grid.ifft_normalization

    parallel_loop_over_range(1:n_bands, H.scratch) do iband, storage
        to = TimerOutput()  # Thread-local timer output

        if is_full
            ψ_real_1 = storage.ψ_reals[1]
            ψ_real_2 = storage.ψ_reals[2]

            @timeit to "local" begin
                ifft!(ψ_real_1, H.basis, H.kpoint, ψ[1:n_G, iband]; normalize=false)
                ifft!(ψ_real_2, H.basis, H.kpoint, ψ[n_G+1:end, iband]; normalize=false)

                for I in CartesianIndices(ψ_real_1)
                    Vn = potential[I, 1]
                    Vx = potential[I, 2]
                    Vy = potential[I, 3]
                    Vz = potential[I, 4]

                    # Construct the local 2x2 Pauli matrix
                    V11 = Vn + Vz
                    V22 = Vn - Vz
                    V12 = Vx - im * Vy
                    V21 = Vx + im * Vy

                    c1 = ψ_real_1[I]
                    c2 = ψ_real_2[I]

                    # Apply it to the spinor
                    ψ_real_1[I] = V11 * c1 + V12 * c2
                    ψ_real_2[I] = V21 * c1 + V22 * c2
                end

                fft!(Hψ[1:n_G, iband], H.basis, H.kpoint, ψ_real_1; normalize=false)
                fft!(Hψ[n_G+1:end, iband], H.basis, H.kpoint, ψ_real_2; normalize=false)
            end
        else
            ψ_real = storage.ψ_reals[1]
            @timeit to "local" begin
                ifft!(ψ_real, H.basis, H.kpoint, ψ[:, iband]; normalize=false)
                ψ_real .*= potential
                fft!(Hψ[:, iband], H.basis, H.kpoint, ψ_real; normalize=false)  # overwrites ψ_real
            end

            if have_divAgrad
                @timeit to "divAgrad" begin
                    apply!((; fourier=Hψ[:, iband], real=nothing),
                           H.divAgrad_op,
                           (; fourier=ψ[:, iband], real=nothing);
                           ψ_real, G_plus_k) # overwrites ψ_real
                end
            end
        end

        if Threads.threadid() == 1
            merge!(DFTK.timer, to; tree_point=[t.name for t in DFTK.timer.timer_stack])
        end
    end

    # Kinetic term
    if is_full
        Hψ[1:n_G, :] .+= H.fourier_op.multiplier .* ψ[1:n_G, :]
        Hψ[n_G+1:end, :] .+= H.fourier_op.multiplier .* ψ[n_G+1:end, :]
    else
        Hψ .+= H.fourier_op.multiplier .* ψ
    end

    # Apply the nonlocal operator (spin-diagonal without SOC)
    if !isnothing(H.nonlocal_op)
        @timing "nonlocal" begin
            if is_full
                apply!((; fourier=Hψ[1:n_G, :], real=nothing),
                       H.nonlocal_op,
                       (; fourier=ψ[1:n_G, :], real=nothing))
                apply!((; fourier=Hψ[n_G+1:end, :], real=nothing),
                       H.nonlocal_op,
                       (; fourier=ψ[n_G+1:end, :], real=nothing))
            else
                apply!((; fourier=Hψ, real=nothing),
                       H.nonlocal_op,
                       (; fourier=ψ, real=nothing))
            end
        end
    end

    Hψ
end


"""
Get energies and Hamiltonian
kwargs is additional info that might be useful for the energy terms to precompute
(eg the density ρ)
"""
@timing function energy_hamiltonian(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    # it: index into terms, ik: index into kpoints
    @timing "ene_ops" ene_ops_arr = [ene_ops(term, basis, ψ, occupation; kwargs...)
                                     for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    energy_values  = [eh.E for eh in ene_ops_arr]
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
    scratch = _ham_allocate_scratch(basis)
    hks_per_k = [flatten([blocks[ik] for blocks in operators])
                 for ik = 1:length(basis.kpoints)]      # hks_per_k[ik][it]
    ham = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks; scratch)
                              for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    energies = Energies(term_names, energy_values)
    (; energies, ham)
end

"""
Faster version than energy_hamiltonian for cases where only the energy is needed.
"""
@timing function energy(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    energy_values = [energy(term, basis, ψ, occupation; kwargs...) for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    (; energies=Energies(term_names, energy_values))
end

function Hamiltonian(basis::PlaneWaveBasis; ψ=nothing, occupation=nothing, kwargs...)
    energy_hamiltonian(basis, ψ, occupation; kwargs...).ham
end

"""
Get the total local potential of the given Hamiltonian, in real space
in the spin components.
"""
function total_local_potential(ham::Hamiltonian)
    if ham.basis.model.spin_polarization == :full
        return total_local_potential(ham.blocks[1])
    end
    n_spin = ham.basis.model.n_spin_components
    pots = map(1:n_spin) do σ
        # Get the potential from the first Hamiltonian block of this spin component
        # (works since all local potentials are the same)
        i_σ = first(krange_spin(ham.basis, σ))
        total_local_potential(ham.blocks[i_σ])
    end
    cat(pots..., dims=4)
end
total_local_potential(Hk::DftHamiltonianBlock) = Hk.local_op.potential
function total_local_potential(Hk::GenericHamiltonianBlock)
    only(o for o in Hk.optimized_operators if o isa RealSpaceMultiplication).potential
end

"""
Returns a new Hamiltonian with local potential replaced by the given one
"""
function hamiltonian_with_total_potential(ham::Hamiltonian, V)
    if ham.basis.model.spin_polarization == :full
        @assert size(V, 4) == 4
        newblocks = [hamiltonian_with_total_potential(Hk, V) for Hk in ham.blocks]
        return Hamiltonian(ham.basis, newblocks)
    end
    
    @assert size(V, 4) == ham.basis.model.n_spin_components
    newblocks = [hamiltonian_with_total_potential(Hk, V[:, :, :, Hk.kpoint.spin])
                 for Hk in ham.blocks]
    Hamiltonian(ham.basis, newblocks)
end
function hamiltonian_with_total_potential(Hk::HamiltonianBlock, V)
    operators = [op for op in Hk.operators if !(op isa RealSpaceMultiplication)]
    push!(operators, RealSpaceMultiplication(Hk.basis, Hk.kpoint, V))
    HamiltonianBlock(Hk.basis, Hk.kpoint, operators; Hk.scratch)
end
