@doc raw"""Compute a ``k``-Point block of an operator as a dense matrix"""
function kblock_as_matrix(ham::Hamiltonian, ik::Int, pot_hartree_values, pot_xc_values)
    # TODO This assumes a PlaneWaveBasis
    n_bas = prod(ham.basis.grid_size)
    T = eltype(ham)
    mat = Matrix{T}(undef, (n_bas, n_bas))
    v = fill(zero(T), n_bas)
    @inbounds for i = 1:n_bas
        v[i] = one(T)
        mat[:,i] .= 0
        apply_fourier!(view(mat, :, i), ham, ik, pot_hartree_values, pot_xc_values, v)
        v[i] = zero(T)
    end
return mat
end
function kblock_as_matrix(prec::PreconditionerKinetic, ik::Int)
    # TODO This assumes a PlaneWaveBasis and Float64 datatype
    n_bas = prod(prec.basis.grid_size)
    T = Float64
    mat = Matrix{T}(undef, (n_bas, n_bas))
    v = fill(zero(T), n_bas)
    @inbounds for i = 1:n_bas
        v[i] = one(T)
        ldiv!(view(mat, :, i), prec, ik, v)
        v[i] = zero(T)
    end
return mat
end
