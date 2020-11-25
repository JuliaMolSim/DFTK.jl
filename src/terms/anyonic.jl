# Ref https://arxiv.org/pdf/1901.10739.pdf
# This term does not contain the kinetic energy, which must be added separately
# /!\ They have no 1/2 factor in front of the kinetic energy,
#     so for consistency the added kinetic energy must have a scaling of 2
# Energy = <u, ((-i∇ + βA)^2 + V) u>
# where ∇∧A = 2π ρ, ∇⋅A = 0 => A = x^⟂/|x|² ∗ ρ
# H = (-i∇ + βA)² + V - 2β x^⟂/|x|² ∗ (βAρ + J)
#   = -Δ + 2β (-i∇)⋅A + β²|A|^2 - 2β x^⟂/|x|² ∗ (βAρ + J)
# where only the first three terms "count for the energy", and where
# J = 1/(2i) (u* ∇u - u ∇u*)


# for numerical reasons, we solve the equation ∇∧A = 2π ρ in two parts:
# ∇∧A = 2π(ρ-ρref + ρref)
# where ρref is a gaussian centered at the origin and with the same integral as ρ (=M)
# ρref(x) = M exp(-1/2 (x/σ)^2) / (σ^3 (2π)^3/2)

function ρref_real(x::T, y::T, M, σ) where {T <: Real}
    r = hypot(x, y)
    M * exp(-1/2 * (r/σ)^2) / (σ^2 * (2T(π))^(2/2))
end

function magnetic_field_produced_by_ρref(x::T, y::T, M, σ) where {T <: Real}
    # The solution of ∇∧A = 2π ρref is ϕ(r) [-y;x] where ϕ satisfies
    # ∇∧A = 2ϕ + r ϕ' => rϕ' + 2 ϕ = 2π ρref
    # Wolfram alpha (after a bit of coaxing) says the solution to
    # rϕ' + 2ϕ = C exp(-α x^2)
    # is
    # 1/x^2 (c1 - C exp(-α x^2)/2α), which coupled with smoothness at 0 gives
    # C/(2α)/x^2*(1 - exp(-α x^2))
    r = hypot(x, y)
    r == 0 && return magnetic_field_produced_by_ρref(1e-8, 0.0, M, σ) # hack

    α = 1/(2*σ^2)
    C = 2T(π)*M/(σ^2 * (2T(π))^(2/2))
    r = hypot(x, y)
    ϕ = 1/2*C/α/r^2*(1-exp(-α*r^2))
    ϕ * @SVector[-y, x]
end

function make_div_free(basis::PlaneWaveBasis{T}, A) where {T}
    out = [zeros(complex(T), basis.fft_size), zeros(complex(T), basis.fft_size)]
    A_fourier = [from_real(basis, A[α]).fourier for α = 1:2]
    for (iG, G) in enumerate(G_vectors(basis))
        vec = [A_fourier[1][iG], A_fourier[2][iG]]
        G = [G[1], G[2]]
        if iG != 1
            out[1][iG], out[2][iG] = vec - (G'vec)*G/sum(abs2, G)
        else
            out[1][iG], out[2][iG] = vec
        end
    end
    [from_fourier(basis, out[α]).real for α = 1:2]
end

function _test_ρref()
    x = 1.23
    y = -1.8
    ε = 1e-8
    M = 2.31
    σ = 1.81
    dAdx = (magnetic_field_produced_by_ρref(x+ε, y, M, σ) - magnetic_field_produced_by_ρref(x, y, M, σ))/ε
    dAdy = (magnetic_field_produced_by_ρref(x, y+ε, M, σ) - magnetic_field_produced_by_ρref(x, y, M, σ))/ε
    curlA = dAdx[2] - dAdy[1]
    divA = dAdx[1] + dAdy[2]
    @assert norm(curlA - 2π*ρref_real(x, y, M, σ)) < 1e-4
    @assert abs(divA) < 1e-6
end
_test_ρref()



struct Anyonic
    β
end
function (A::Anyonic)(basis)
    @assert length(basis.kpoints) == 1
    @assert basis.kpoints[1].coordinate == [0, 0, 0]
    @assert basis.model.dim == 2
    # only square lattices allowed
    # (because I can't be bothered to think about the right formulas otherwise,
    # although they might already be correct)
    @assert basis.model.lattice[2, 1] == basis.model.lattice[1, 2] == 0
    @assert basis.model.lattice[1, 1] == basis.model.lattice[2, 2]

    TermAnyonic(basis, eltype(basis)(A.β))
end

struct TermAnyonic{T <: Real, Tρ <: RealFourierArray, TA} <: Term
    basis::PlaneWaveBasis{T}
    β::T
    ρref::Tρ
    Aref::TA
end
function TermAnyonic(basis::PlaneWaveBasis{T}, β) where {T}
    # compute correction magnetic field
    ρref = zeros(T, basis.fft_size)
    Aref = [zeros(T, basis.fft_size), zeros(T, basis.fft_size)]
    M = basis.model.n_electrons  # put M to zero to disable
    σ = 2
    for (ir, r) in enumerate(r_vectors(basis))
        rcart = basis.model.lattice * (r - @SVector[.5, .5, .0])
        ρref[ir] = ρref_real(rcart[1], rcart[2], M, σ)
        Aref[1][ir], Aref[2][ir] = magnetic_field_produced_by_ρref(rcart[1], rcart[2], M, σ)
    end
    Aref = make_div_free(basis, Aref)
    ρref = from_real(basis, ρref)
    TermAnyonic(basis, β, ρref, Aref)
end

function ene_ops(term::TermAnyonic, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    β = term.β
    @assert ψ !== nothing # the hamiltonian depends explicitly on ψ

    # Compute A in Fourier domain
    # curl A = 2π ρ, ∇⋅A = 0
    # i [G1;G2] ∧ A(G1,G2) = 2π ρ(G1,G2), [G1;G2] ⋅ A(G1,G2) = 0
    # => A(G1, G2) = -2π i ρ(G1, G2) * [-G2;G1;0]/(G1^2 + G2^2)
    A1 = zeros(complex(T), basis.fft_size)
    A2 = zeros(complex(T), basis.fft_size)
    ρ_fourier = ρ.fourier  # unpack before hot loop for perf
    ρref_fourier = term.ρref.fourier
    for (iG, G) in enumerate(G_vectors_cart(basis))
        G2 = sum(abs2, G)
        if G2 != 0
            A1[iG] = +2T(π) * G[2] / G2 * (ρ_fourier[iG]-ρref_fourier[iG]) * im
            A2[iG] = -2T(π) * G[1] / G2 * (ρ_fourier[iG]-ρref_fourier[iG]) * im
        end
    end
    Areal = [from_fourier(basis, A1).real + term.Aref[1],
             from_fourier(basis, A2).real + term.Aref[2],
             zeros(T, basis.fft_size)]

    # 2β (-i∇)⋅A + β^2 |A|^2
    ops_energy = [MagneticFieldOperator(basis, basis.kpoints[1],
                                        2β .* Areal),
                  RealSpaceMultiplication(basis, basis.kpoints[1],
                                          β^2 .* (abs2.(Areal[1]) .+ abs2.(Areal[2])))]

    # Now compute effective local potential - 2β x^⟂/|x|² ∗ (βAρ + J)
    J = compute_current(basis, ψ, occ)
    eff_current = [J[α].real .+
                   β .* ρ.real .* Areal[α] for α = 1:2]
    eff_current_fourier = [from_real(basis, eff_current[α]).fourier for α = 1:2]
    # eff_pot = - 2β x^⟂/|x|² ∗ eff_current
    # => ∇∧eff_pot = -4πβ eff_current
    # => eff_pot(G1, G2) = 4πβ i eff_current(G1, G2) * [-G2;G1;0]/(G1^2 + G2^2)
    eff_pot_fourier = zeros(complex(T), basis.fft_size)
    for (iG, Gred) in enumerate(G_vectors(basis))
        G = basis.model.recip_lattice * Gred
        G2 = sum(abs2, G)
        if G2 != 0
            eff_pot_fourier[iG] += -4T(π)*β * im * G[2] / G2 * eff_current_fourier[1][iG]
            eff_pot_fourier[iG] += +4T(π)*β * im * G[1] / G2 * eff_current_fourier[2][iG]
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
