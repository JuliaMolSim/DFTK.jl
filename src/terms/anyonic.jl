# Ref https://arxiv.org/pdf/1901.10739.pdf
# This term does not contain the kinetic energy, which must be added separately
# /!\ They have no 1/2 factor in front of the kinetic energy,
#     so for consistency the added kinetic energy must have a scaling of 2
# Energy = <u, ((-i∇+βA)^2 + V) u>
# where ∇∧A = 2π ρ, ∇⋅A = 0 => A = x⟂/|x|² ∗ ρ
# H = (-i∇+βA)² + V - 2β x⟂/|x|² ∗ (βAρ + J)
#   = -Δ + 2β (-i∇)⋅A + β²|A|^2 - 2β x⟂/|x|² ∗ (βAρ + J)
# where only the first three terms "count for the energy", and where
# J = 1/(2i) (u* ∇u - u ∇u*)

struct Anyonic
    β
end
function (A::Anyonic)(basis)
    @assert length(basis.kpoints) == 1
    @assert basis.kpoints[1].coordinate == [0, 0, 0]
    @assert basis.model.dim == 2
    # only square lattices allowed
    # (because I can't be bothered to figure out the right formulas otherwise)
    @assert basis.model.lattice[2, 1] == basis.model.lattice[1, 2] == 0
    @assert basis.model.lattice[1, 1] == basis.model.lattice[2, 2]

    TermAnyonic(basis, A.β)
end

struct TermAnyonic <: Term
    basis::PlaneWaveBasis
    β
end

function ene_ops(term::TermAnyonic, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    β = term.β
    @assert ψ !== nothing # the hamiltonian depends explicitly on ψ

    # Compute A in Fourier domain
    # curl A = 2π ρ
    # A(G1, G2) = -2π i ρ(G1, G2) * [-G2;G1;0]/(G1^2 + G2^2)
    # A is in cartesian (not reduced) coordinates
    A1 = zeros(complex(T), basis.fft_size)
    A2 = zeros(complex(T), basis.fft_size)
    for (iG, Gred) in enumerate(G_vectors(basis))
        G = basis.model.recip_lattice * Gred
        G2 = sum(abs2, G)
        if G2 != 0
            A1[iG] = +2T(π) * G[2] / G2 * ρ.fourier[iG] * im
            A2[iG] = -2T(π) * G[1] / G2 * ρ.fourier[iG] * im
        end
    end
    Areal = [from_fourier(basis, A1).real,
             from_fourier(basis, A2).real,
             zeros(T, basis.fft_size)]

    # 2β (-i∇)⋅A + |A|^2
    ops_energy = [MagneticFieldOperator(basis, basis.kpoints[1], 2β .* Areal),
                  RealSpaceMultiplication(basis, basis.kpoints[1], β^2 .* (abs2.(Areal[1]) .+ abs2.(Areal[2])))]

    # Now compute effective local potential - 2β x⟂/|x|² ∗ (βAρ + J)
    current = compute_current(basis, ψ, occ)
    # TODO rationalize where this 1/sqrt(basis.model.recip_cell_volume) comes from
    eff_current = [1/sqrt(basis.model.recip_cell_volume)*current[α].real .+
                   β .* ρ.real .* Areal[α] for α = 1:2]
    eff_current_fourier = [from_real(basis, eff_current[α]).fourier for α = 1:2]
    # eff_pot = - 2β x⟂/|x|² ∗ eff_current
    # => ∇∧eff_pot = -4πβ eff_current
    # => eff_pot(G1, G2) = 4πβ i eff_current(G1, G2) * [-G2;G1;0]/(G1^2 + G2^2)
    eff_pot_fourier = zeros(complex(T), basis.fft_size)
    for (iG, Gred) in enumerate(G_vectors(basis))
        G = basis.model.recip_lattice * Gred
        G2 = sum(abs2, G)
        if G2 != 0
            eff_pot_fourier[iG] += -4T(π)*β * G[2] / G2 * eff_current_fourier[1][iG] * im
            eff_pot_fourier[iG] += +4T(π)*β * G[1] / G2 * eff_current_fourier[2][iG] * im
        end
    end
    eff_pot_real = from_fourier(basis, eff_pot_fourier).real
    ops_ham = [ops_energy..., RealSpaceMultiplication(basis, basis.kpoints[1], eff_pot_real)]

    E = zero(T)
    for iband = 1:size(ψ[1], 2)
        ψnk = @views ψ[1][:, iband]
        # TODO optimize this
        for op in ops_energy
            E += occ[1][iband] * real(dot(ψnk, op * ψnk))
        end
    end

    (E=E, ops=[ops_ham])
end
