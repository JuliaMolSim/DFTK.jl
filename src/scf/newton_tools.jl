## Tools to perform projection on tangent spaces and compute Jacobian matrices
## for newton steps

import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

############################### TOOLS ##########################################

"""
    compute_residual(basis::PlaneWaveBasis, φ, occ)

Compute the residual associated to a set of planewave φ, that is to say
H(φ)*φ - λ*φ where λ is the set of rayleigh coefficients associated to the φ.
"""
function compute_residual(basis::PlaneWaveBasis, φ, occ)

    # necessary quantities
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ)

    # compute residual
    res = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        N = size(φk, 2)
        Hk = H.blocks[ik]
        # eigenvalues as rayleigh coefficients
        egvalk = φk'*(Hk*φk)
        # compute residual at given kpoint as H(φ)φ - λφ
        res[ik] = Hk*φk - φk*egvalk
    end
    res
end

# to project onto the space tangent to ψ, we project ϕ onto the orthogonal of ψ
# TODO : detail why
function proj(ϕ, ψ; tol_test=1e-12)

    Nk1 = size(ϕ,1)
    Nk2 = size(ψ,1)
    @assert Nk1 == Nk2
    Nk = Nk1

    Πϕ = similar(ϕ)

    for ik = 1:Nk
        ψk = ψ[ik]
        ϕk = ϕ[ik]
        Πϕk = deepcopy(ϕk)

        N1 = size(ϕk,2)
        N2 = size(ψk,2)
        @assert N1 == N2
        N = N1

        for i = 1:N, j = 1:N
            Πϕk[:,i] -= (ψk[:,j]'ϕk[:,i]) * ψk[:,j]
        end
        Πϕ[ik] = Πϕk
    end

    # test orthogonalisation
    for ik = 1:Nk
        ψk = ψ[ik]
        ϕk = ϕ[ik]
        Πϕk = Πϕ[ik]
        N = size(ψk,2)
        for i = 1:N, j = 1:N
            @assert abs(Πϕk[:,i]'ψk[:,j]) < tol_test [println(abs(Πϕk[:,i]'ψk[:,j]))]
        end
    end

    Πϕ
end

# packing routines
pack(φ) = vcat(Base.vec.(φ)...)
function packing(basis::PlaneWaveBasis{T}, φ) where T

    Nk = length(basis.kpoints)

    lengths = [length(φ[ik]) for ik = 1:Nk]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    unpack(x) = [@views reshape(x[starts[ik]:starts[ik]+lengths[ik]-1], size(φ[ik]))
                 for ik = 1:Nk]

    packed_proj(ϕ,φ) = proj(unpack(ϕ), unpack(φ))

    (pack, unpack, packed_proj)
end

# KrylovKit custom orthogonaliser to be used in KrylovKit eigsolve, svdsolve,
# linsolve, ...
struct OrthogonalizeAndProject{F, O <: Orthogonalizer, ψ} <: Orthogonalizer
    projector::F
    orth::O
    ψ::ψ
end
OrthogonalizeAndProject(projector, ψ) = OrthogonalizeAndProject(projector,
                                                                KrylovDefaults.orth,
                                                                ψ)
function KrylovKit.orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, b, x, alg.orth)
    v = reshape(v, size(alg.ψ))
    v = pack(alg.projector(v, alg.ψ))::T
    v, x
end
function KrylovKit.orthogonalize!(v::T, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, x, alg.orth)
    v = reshape(v, size(alg.ψ))
    v = pack(alg.projector(v, alg.ψ))::T
    v, x
end
function KrylovKit.gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                                 alg::OrthogonalizeAndProject)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    # for q in V # not necessary if we definitely reorthogonalize next step and previous step
    #     v, = orthogonalize!(v, q, ModifiedGramSchmidt())
    # end
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    for q in U
        r, = orthogonalize!(r, q, alg)
    end
    β = norm(r)
    return v, r, α, β
end

############################# OPERATORS ########################################

"""
    apply_Ω(basis::PlaneWaveBasis, δφ, φ, H, λ)

Compute the application of Ω to an element on the tangent space.
Here, an element on the tangent space can be written as
      δP = Σ |φi><δφi| + hc
where the δφi are of size Nb and are all orthogonal to the φj, 1 <= j <= N.
Therefore we store them in the same kind of array than φ, with
δφ[ik][:,i] = δφi for each k-point.
Therefore, computing Ωδφ can be done analitically with the formula
      Ωδφi = H*δφi - Σj <δφj|H|δφi> δφj - Σj λji δφj
where λij = <φi|H|φj>.
"""
function apply_Ω(basis::PlaneWaveBasis, δφ, φ, H, λ)
    Nk = length(basis.kpoints)
    Ωδφ = similar(φ)

    for ik = 1:Nk
        φk = φ[ik]
        δφk = δφ[ik]
        λk = λ[ik]
        Ωδφk = similar(δφk)

        N1 = size(δφk,2)
        N2 = size(φk,2)
        @assert N1 == N2
        N = N1

        # compute component on i
        for i = 1:N
            Hδφki = H.blocks[ik] * δφk[:,i]
            Ωδφk[:,i] = Hδφki
            for j = 1:N
                Ωδφk[:,i] -= (φk[:,j]'Hδφki) * φk[:,j]
                Ωδφk[:,i] -= λk[j,i] * δφk[:,j]
            end
        end
        Ωδφ[ik] = Ωδφk
    end
    # ensure proper projection onto the tangent space
    proj(Ωδφ, φ)
end

"""
    apply_K(basis::PlaneWaveBasis, δφ, φ, ρ, occ)

Compute the application of K to an element in the tangent space
with the same notations as for the computation of Ω, we have
      Kδφi = Π(dV * φi),
Π being the projection on the space tangent to the φ and
      dV = kernel(δρ)
where δρ = Σi φi*conj(δφi) + hc.
"""
function apply_K(basis::PlaneWaveBasis, δφ, φ, ρ, occ)
    Nk = length(basis.kpoints)

    δρ = compute_density(basis, φ, δφ, occ)
    dV = apply_kernel(basis, δρ; ρ=ρ)
    Kδφ = similar(φ)

    for ik = 1:Nk
        kpt = basis.kpoints[ik]
        φk = φ[ik]
        dVφk = similar(φk)

        for i = 1:size(φk,2)
            φk_r = G_to_r(basis, kpt, φk[:,i])
            dVφk_r = dV .* φk_r
            dVφk_r = total_density(dVφk_r)
            dVφk[:,i] = r_to_G(basis, kpt, dVφk_r)
        end
        Kδφ[ik] = dVφk
    end
    # ensure proper projection onto the tangent space
    proj(Kδφ, φ)
end

"""
    ΩplusK(basis::PlaneWaveBasis, δφ, φ, ρ, H, λ, occ)

Apply (Ω+K) to an element δφ of the tangent space.
"""
function ΩplusK(basis, δφ, φ, ρ, H, λ, occ)
    δφ = proj(δφ, φ)
    Kδφ = apply_K(basis, δφ, φ, ρ, occ)
    Ωδφ = apply_Ω(basis, δφ, φ, H, λ)
    ΩpKδφ = Ωδφ .+ Kδφ
    # ensure proper projection onto the tangent space
    proj(ΩpKδφ, φ)
end

