# Ref https://arxiv.org/pdf/1901.10739.pdf
# Their kinetic energy is twice ours, so we must take twice their potential energy as well

struct Anyonic
    β
end
function (A::Anyonic)(basis)
    @assert length(basis.kpoints) == 1
    @assert basis.kpoints[1].coordinate == [0, 0, 0]
    @assert basis.model.dim == 2
    @assert basis.model.unit_cell[2, 1] == basis.model.unit_cell[1, 2] == 0

    TermAnyonic(basis, A.β)
end

struct TermAnyonic <: Term
    basis::PlaneWaveBasis
    β
end

function ene_ops(term::TermMagnetic, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    @assert ψ !== nothing # the hamiltonian depends explicitly on ψ

    # Compute A in Fourier domain
    # curl A = 2π ρ
    # A(G1, G2) = 2π ρ(G1, G2) * [-G2;G1;0]/(G1^2 + G2^2)
    A1 = zeros(T, basis.fft_size)
    A2 = zeros(T, basis.fft_size)
    for (iG, Gred) in enumerate(G_vectors(basis))
        G = basis.model.unit_cell * Gred
        G2 = sum(abs2, G)
        A1[iG] = -2T(π) * G[2] / G2 * ρ.fourier[iG]
        A2[iG] =  2T(π) * G[1] / G2 * ρ.fourier[iG]
    end
    Areal = [from_fourier(basis, A1).real,
             from_fourier(basis, A2).real,
             zeros(T, fft_size)]
    op = [MagneticFieldOperator(basis, kpoint, term.Apotential), RealSpaceMultiplication(basis, A[1] .* )]
    MFO = 

    ops = [
           for (ik, kpoint) in enumerate(basis.kpoints)]

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            # TODO optimize this
            E += basis.kweights[ik] * occ[ik][iband] * real(dot(ψnk, ops[ik] * ψnk))
        end
    end

    (E=E, ops=[op])
end
