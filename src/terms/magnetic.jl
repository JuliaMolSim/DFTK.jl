# EXPERIMENTAL this is very naive, probably needs to be thought about a bit more and the results have not been checked
# TODO check the formulas and document
struct Magnetic
    Afunction::Function # A(x,y,z) returns [Ax,Ay,Az]
end
function (M::Magnetic)(basis)
    @warn "Magnetic() is experimental. You're on your own (but please report bugs)."
    TermMagnetic(basis, M.Afunction)
end

struct TermMagnetic <: Term
    basis::PlaneWaveBasis
    Apotential::AbstractArray  # Apotential[α] is an array of size fft_size for α=1:3
end
function TermMagnetic(basis::PlaneWaveBasis{T}, Afunction::Function) where T
    lattice = basis.model.lattice
    Apotential = [zeros(T, basis.fft_size) for α = 1:3]
    N1, N2, N3 = basis.fft_size
    for i = 1:N1
        for j = 1:N2
            for k = 1:N3
                Apotential[1][i, j, k],
                Apotential[2][i, j, k],
                Apotential[3][i, j, k] = Afunction(lattice * @SVector[(i-1) / N1,
                                                                      (j-1) / N2,
                                                                      (k-1) / N3])
            end
        end
    end
    TermMagnetic(basis, Apotential)
end

function ene_ops(term::TermMagnetic, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)

    ops = [MagneticFieldOperator(basis, kpoint, term.Apotential)
           for (ik, kpoint) in enumerate(basis.kpoints)]
    ψ === nothing && return (E=T(Inf), ops=ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            # TODO optimize this
            E += basis.kweights[ik] * occ[ik][iband] * real(dot(ψnk, ops[ik] * ψnk))
        end
    end

    (E=E, ops=ops)
end
