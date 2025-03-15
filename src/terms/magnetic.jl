# This is pretty naive and probably needs to be thought about a bit more, esp. in the periodic context.
# Right now this is sufficient to reproduce uniform fields for isolated systems.

@doc raw"""
Magnetic term ``A⋅(-i∇)``. It is assumed (but not checked) that ``∇⋅A = 0``.
"""
struct Magnetic
    Afunction::Function  # A(x,y,z) returns [Ax,Ay,Az]
                         # both [x,y,z] and [Ax,Ay,Az] are in *Cartesian* coordinates
end
function (M::Magnetic)(basis)
    Apotential = [zeros(T, basis.fft_size) for α = 1:3]
    N1, N2, N3 = basis.fft_size
    rvecs = collect(r_vectors_cart(basis))
    for i = 1:N1, j = 1:N2, k = 1:N3
        Apotential[1][i, j, k],
        Apotential[2][i, j, k],
        Apotential[3][i, j, k] = M.Afunction(rvecs[i, j, k])
    end
    TermMagnetic(Apotential)
end

struct MagneticFromValues
    # Apotential[α] is an array of size fft_size for α=1:3
    Apotential::Vector{<:AbstractArray}
end
function (M::MagneticFromValues)(basis::PlaneWaveBasis)
    @assert length(M.Apotential) == 3
    @assert size(M.Apotential[1]) == basis.fft_size
    @assert size(M.Apotential[2]) == basis.fft_size
    @assert size(M.Apotential[3]) == basis.fft_size
    TermMagnetic(M.Apotential)
end

struct TermMagnetic <: Term
    # Apotential[α] is an array of size fft_size for α=1:3
    Apotential::Vector{<:AbstractArray}
end

function ene_ops(term::TermMagnetic, basis::PlaneWaveBasis{T}, ψ, occupation;
                 kwargs...) where {T}
    ops = [MagneticFieldOperator(basis, kpoint, term.Apotential)
           for (ik, kpoint) in enumerate(basis.kpoints)]
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(Inf), ops)
    end

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            # TODO optimize this
            E += basis.kweights[ik] * occupation[ik][iband] * real(dot(ψnk, ops[ik] * ψnk))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (; E, ops)
end
