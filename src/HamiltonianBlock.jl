# Data structures for representing a k-Point block of a one-particle Hamiltonian

struct HamiltonianBlock
    basis::PlaneWaveBasis
    kpt::Kpoint

    # Stored values representing this Hamiltonian block
    values_kinetic   # Kinetic diagonal values in B_k Fourier basis
    values_local     # Local potential values (psp, XC, Hartree)
    values_nonlocal  # Non-local operator in B_k Fourier basis
end

# Matrix-like interface for solvers
import Base: *, \, Matrix, Array

function Base.size(block::HamiltonianBlock, idx::Int)
    idx > 2 && return 1
    return size(block.values_kinetic, 1)
end
Base.size(block::HamiltonianBlock) = (size(block, 1), size(block, 2))
Base.eltype(block::HamiltonianBlock) = complex(eltype(block.values_kinetic))
*(block::HamiltonianBlock, X) = mul!(similar(X), block, X)


function Matrix(block::HamiltonianBlock)
    n_bas = length(block.kpt.basis)
    T = eltype(block)
    mat = Matrix{T}(undef, (n_bas, n_bas))
    v = fill(zero(T), n_bas)
    @inbounds for i = 1:n_bas
        v[i] = one(T)
        mul!(view(mat, :, i), block, v)
        v[i] = zero(T)
    end
    mat
end
Array(block::HamiltonianBlock) = Matrix(block::HamiltonianBlock)


function LinearAlgebra.mul!(Y, block::HamiltonianBlock, X)
    kin = block.values_kinetic
    Vnloc = block.values_nonlocal
    Vloc = block.values_local

    @assert kin isa Diagonal # for optimization

    if Vloc === nothing
        Y .= kin.diag .* X
    else
        @views Threads.@threads for n = 1:size(X, 2)
            Xreal = G_to_r(block.basis, block.kpt, X[:, n])
            Xreal .*= Vloc
            r_to_G!(Y[:, n], block.basis, block.kpt, Xreal)
            Y[:,n] .+= kin.diag .* X[:, n]
        end
    end

    Vnloc !== nothing && (Y .+= Vnloc * X)

    Y
end


"""
Generate one k-Point block of a Hamiltonian, can be used as a matrix
"""
function HamiltonianBlock(ham::Hamiltonian, kpt::Kpoint)
    HamiltonianBlock(ham.basis, kpt, kblock(ham.kinetic, kpt),
                     ham.pot_local, kblock(ham.pot_nonlocal, kpt))
end

"""
TODO docme
"""
kblock(ham::Hamiltonian, kpt::Kpoint) = HamiltonianBlock(ham, kpt)
kblock(::Nothing, kpt::Kpoint) = nothing
