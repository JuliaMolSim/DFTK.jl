import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

############################# ERROR AND RESIDUAL ###############################

## ORBITAL FORMULATION

# compute the residual associated to a set of planewave φ, that is to say
# H(φ)*φ - λ.*φ where λ is the set of rayleigh coefficients associated to the
# φ
# we also return the egval set for further computations
function compute_residual(basis::PlaneWaveBasis{T}, φ, occ) where T

    # necessary quantities
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])

    # compute residual
    res = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        N = size(φk, 2)
        Hk = H.blocks[ik]
        # eigenvalues as rayleigh coefficients
        egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
        # compute residual at given kpoint as H(φ)φ - λφ
        res[ik] = Hk*φk - hcat([egvalk[i] * φk[:,i] for i = 1:N]...)
    end
    res
end

# compute the error on the orbitals by aligning the eigenvectors
# this is done by solving min |ϕ - ψ*U| for U unitary matrix of size NxN
# whose solution is U = M(M^*M)^-1/2 where M = ψ^*ϕ
function compute_error(basis, ϕ, ψ)

    # necessary quantites
    Nk = length(basis.kpoints)

    # compute error
    err = similar(ϕ)
    for ik = 1:Nk
        ϕk = ϕ[ik]
        ψk = ψ[ik]
        # compute overlap matrix
        M = ψk'ϕk
        U = M*(M'M)^(-1/2)
        err[ik] = ϕk - ψk*U
    end
    err
end

############################## TANGENT SPACE TOOLS #############################

# test for orthogonalisation
tol_test = 1e-8

# we project ϕ onto the orthogonal of ψ
function proj(ϕ, ψ; high_freq=false, low_freq=false, Ecut=nothing, basis=nothing)

    Nk1 = size(ϕ,1)
    Nk2 = size(ψ,1)
    @assert Nk1 == Nk2
    Nk = Nk1

    Πϕ = similar(ϕ)

    if high_freq
        Πϕ = keep_HF(ϕ, basis, Ecut)
    elseif low_freq
        Πϕ = keep_LF(ϕ, basis, Ecut)
    else
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
    end

    Πϕ
end

# packing routines
function packing(basis::PlaneWaveBasis{T}, φ;
                 high_freq=false, low_freq=false, Ecut=nothing) where T
    Nk = length(basis.kpoints)
    lengths = [length(φ[ik]) for ik = 1:Nk]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    pack(φ) = vcat(Base.vec.(φ)...)
    unpack(x) = [@views reshape(x[starts[ik]:starts[ik]+lengths[ik]-1], size(φ[ik]))
                 for ik = 1:Nk]
    if (high_freq || low_freq) && Nk == 1
        b = basis
    else
        b = nothing
    end
    packed_proj(ϕ,φ) = proj(unpack(ϕ), unpack(φ);
                            high_freq=high_freq, low_freq=low_freq,
                            Ecut=Ecut, basis=b)
    (pack, unpack, packed_proj)
end

# KrylovKit custom orthogonaliser to be used in KrylovKit eigsolve, svdsolve,
# linsolve, ...
pack(φ) = vcat(Base.vec.(φ)...)
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

############################## CHANGES OF NORMS ################################

## T = -1/2 Δ + t

function apply_inv_T(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= 1 ./ (Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
end

function apply_inv_sqrt_T(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= 1 ./ sqrt.(Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
end

function apply_sqrt_T(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= sqrt.(Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
end

function apply_M(φ, Pks, δφ)
    δφ = proj(δφ, φ)
    δφ = apply_sqrt_T(Pks, δφ)
    δφ = proj(δφ, φ)
    δφ = apply_sqrt_T(Pks, δφ)
    δφ = proj(δφ, φ)
end

function apply_sqrt_M(φ, Pks, δφ)
    δφ = proj(δφ, φ)
    δφ = apply_sqrt_T(Pks, δφ)
    δφ = proj(δφ, φ)
end

function apply_inv_sqrt_M(basis, φ, Pks, res)
    Nk = length(Pks)

    pack, unpack, packed_proj = packing(basis, φ)

    function op(x)
        δφ = unpack(x)
        δφ = apply_sqrt_M(φ, Pks, δφ)
        pack(δφ)
    end

    Res, info = linsolve(op, pack(proj(res, φ));
                         tol=tol_krylov, verbosity=0,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    unpack(Res)
end

function apply_inv_M(basis, φ, Pks, res)
    Nk = length(Pks)

    pack, unpack, packed_proj = packing(basis, φ)

    function op(x)
        δφ = unpack(x)
        δφ = apply_M(φ, Pks, δφ)
        pack(δφ)
    end

    Res, info = linsolve(op, pack(proj(res, φ));
                         tol=tol_krylov, verbosity=0,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    unpack(Res)
end



############################# OPERATORS ########################################

# for a given kpoint, we compute the
# application of Ω to an element on the tangent plane
# Here, an element on the tangent plane can be written as
#       δP = Σ |ψi><δφi| + hc
# where the δφi are of size Nb and are all orthogonal to the ψj, 1 <= j <= N
# therefore we store them in the same kind of array than ψ, with
# δφ[ik][:,i] = δφi for each k-point
# therefore, computing Ωδφ can be done analitically
function apply_Ω(basis, δφ, φ, H, egval)
    Nk = length(basis.kpoints)
    Ωδφ = similar(φ)

    δφ = proj(δφ, φ)
    for ik = 1:Nk
        δφk = δφ[ik]
        φk = φ[ik]
        egvalk = egval[ik]

        N1 = size(δφk,2)
        N2 = size(φk,2)
        @assert N1 == N2
        N = N1

        Ωδφk = similar(δφk)

        Hδφk = H.blocks[ik] * δφk

        # compute component on i
        for i = 1:N
            ε_i = egvalk[i]
            Ωδφk[:,i] = Hδφk[:,i] - ε_i * δφk[:,i]
        end
        Ωδφ[ik] = Ωδφk
    end
    proj(Ωδφ, φ)
end

# compute the application of K
function apply_K(basis, δφ, φ, ρ, occ)
    Nk = length(basis.kpoints)

    δφ = proj(δφ, φ)
    δρ = DFTK.compute_density(basis, φ, δφ, occ)
    Kδρ = apply_kernel(basis, δρ[1]; ρ=ρ)
    Kδρ_r = Kδρ[1].real
    Kδφ = similar(φ)

    for ik = 1:Nk
        kpt = basis.kpoints[ik]
        φk = φ[ik]
        Kδρφk = similar(φk)

        N = size(φk,2)
        for i = 1:N
            φk_r = G_to_r(basis, kpt, φk[:,i])
            Kδρφk_r = Kδρ_r .* φk_r
            Kδρφk[:,i] = r_to_G(basis, kpt, Kδρφk_r)
        end
        Kδφ[ik] = Kδρφk
    end
    proj(Kδφ, φ)
end

# Apply (Ω+K)δφ
function ΩplusK(basis, δφ, φ, ρ, H, egval, occ)
    Kδφ = apply_K(basis, δφ, φ, ρ, occ)
    Ωδφ = apply_Ω(basis, δφ, φ, H, egval)
    ΩpKδφ = Ωδφ .+ Kδφ
    proj(ΩpKδφ, φ)
end

