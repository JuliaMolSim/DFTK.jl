# This is pretty naive and probably needs to be thought about a bit more, esp. in the periodic context.
# Right now this is sufficient to reproduce uniform fields for isolated systems.

@doc raw"""
Magnetic term ``A⋅(-i∇)``. It is assumed (but not checked) that ``∇⋅A = 0``.
"""
struct Magnetic
    Afunction::Function  # A(x,y,z) returns [Ax,Ay,Az]
                         # both [x,y,z] and [Ax,Ay,Az] are in *cartesian* coordinates
end
(M::Magnetic)(basis) = TermMagnetic(basis, M.Afunction)

struct TermMagnetic <: Term
    # Apotential[α] is an array of size fft_size for α=1:3
    Apotential::Vector{<:AbstractArray}
end
function TermMagnetic(basis::PlaneWaveBasis{T}, Afunction::Function) where T
    Apotential = [zeros(T, basis.fft_size) for α = 1:3]
    N1, N2, N3 = basis.fft_size
    rvecs = collect(r_vectors_cart(basis))
    for i = 1:N1
        for j = 1:N2
            for k = 1:N3
                Apotential[1][i, j, k],
                Apotential[2][i, j, k],
                Apotential[3][i, j, k] = Afunction(rvecs[i, j, k])
            end
        end
    end
    TermMagnetic(Apotential)
end

function ene_ops(term::TermMagnetic, basis::PlaneWaveBasis{T}, ψ, occ; kwargs...) where {T}
    ops = [MagneticFieldOperator(basis, kpoint, term.Apotential)
           for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(Inf), ops=ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            # TODO optimize this
            E += basis.kweights[ik] * occ[ik][iband] * real(dot(ψnk, ops[ik] * ψnk))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (E=E, ops=ops)
end
