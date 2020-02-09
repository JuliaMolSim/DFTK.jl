# EXPERIMENTAL this is very naive, probably needs to be thought about a bit more and the results have not been checked
# TODO check the formulas and document
struct Magnetic
    A::Function # A(x,y,z) returns [Ax,Ay,Az]
end
(M::Magnetic)(basis) = TermMagnetic(basis, M.A)

struct TermMagnetic <: Term
    basis::PlaneWaveBasis
    Apot::AbstractArray # Apot[i] is an array of size fft_size for i=1:3
end
function TermMagnetic(basis::PlaneWaveBasis{T}, Afun::Function) where T
    Apot = [zeros(T, basis.fft_size) for i = 1:3]
    N1, N2, N3 = basis.fft_size
    for i = 1:N1
        for j = 1:N2
            for k = 1:N3
                Apot[1][i, j, k],
                Apot[2][i, j, k],
                Apot[3][i, j, k] = 
                    Afun(basis.model.lattice * @SVector[(i-1)/N1,
                                                        (j-1)/N2,
                                                        (k-1)/N3])
            end
        end
    end
    TermMagnetic(basis, Apot)
end

term_name(term::TermMagnetic) = "Magnetic"

function ene_ops(term::TermMagnetic, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)

    ops = [MagneticFieldOperator(basis, kpoint, term.Apot) for (ik, kpoint) in enumerate(basis.kpoints)]
    ψ === nothing && return (E=T(Inf), ops=ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            psi = @views ψ[ik][:, iband]
            E += basis.kweights[ik] * occ[ik][iband] * real(dot(psi, ops[ik] * psi)) # TODO optimize this
        end
    end

    (E=E, ops=ops)
end
