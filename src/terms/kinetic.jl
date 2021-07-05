"""
Kinetic energy: 1/2 sum_n f_n ∫ |∇ψn|^2.
"""
struct Kinetic
    scaling_factor::Real
end
Kinetic() = Kinetic(1)
(K::Kinetic)(basis) = TermKinetic(basis; scaling_factor=K.scaling_factor)

struct TermKinetic <: Term
    basis::PlaneWaveBasis
    kinetic_energies::Vector{Vector} # kinetic energy 1/2|G+k|^2 for every kpoint
end
function TermKinetic(basis::PlaneWaveBasis; scaling_factor=1)
    kinetic_energies = [[scaling_factor * sum(abs2, G + kpt.coordinate_cart) / 2
                         for G in G_vectors_cart(kpt)]
                        for kpt in basis.kpoints]
    TermKinetic(basis, kinetic_energies)
end

@timing "ene_ops: kinetic" function ene_ops(term::TermKinetic, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)

    ops = [FourierMultiplication(basis, kpoint, term.kinetic_energies[ik])
           for (ik, kpoint) in enumerate(basis.kpoints)]
    ψ === nothing && return (E=T(Inf), ops=ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            E += (basis.kweights[ik] * occ[ik][iband]
                  * real(dot(ψnk, Diagonal(term.kinetic_energies[ik]), ψnk)))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (E=E, ops=ops)
end
