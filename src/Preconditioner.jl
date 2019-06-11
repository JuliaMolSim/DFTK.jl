"""
Kinetic-energy based preconditioner.
Applies 1 / (|k + G|^2 + α) to the vectors, when called with ldiv!

The rationale is to dampen the high-kinetic energy parts of the
Hamiltonian and decreases their size, thus make the Hamiltonian
more well-conditioned
"""
struct PreconditionerKinetic
    basis::PlaneWaveBasis
    α::Float64
end

function PreconditionerKinetic(ham::Hamiltonian; α=0)
    PreconditionerKinetic(ham.basis, α)
end

function apply_inverse_fourier!(out_Xk, prec::PreconditionerKinetic, ik::Int, in_Xk)
    pw = prec.basis
    k = pw.kpoints[ik]

    qsq = [sum(abs2, pw.recip_lattice * (G + k)) for G in pw.wfctn_basis[ik]]
    diagonal = 1 ./ (qsq ./ 2 .+ 1e-6 .+ prec.α)
    out_Xk .= Diagonal(diagonal) * in_Xk
end

# Get a representation of the Preconditioner as a matrix
# TODO Is there a more julia-idiomatic way to do this?
function block_as_matrix(prec::PreconditionerKinetic, ik::Int)
    # TODO This assumes a PlaneWaveBasis and Float64 datatype
    n_bas = prod(prec.basis.grid_size)
    T = Float64
    mat = Matrix{T}(undef, (n_bas, n_bas))
    v = fill(zero(T), n_bas)
    @inbounds for i = 1:n_bas
        v[i] = one(T)
        apply_inverse_fourier!(view(mat, :, i), prec, ik, v)
        v[i] = zero(T)
    end
return mat
end
