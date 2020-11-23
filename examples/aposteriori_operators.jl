import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

# This file containes functions for computing the jacobian of the direct
# minimization algorihtm on the grassman manifold, that this the operator Ω+K
# defined Cancès/Kemlin/Levitt, Convergence analysis of SCF and direct
# minization algorithms.


################################## TOOL ROUTINES ###############################

# test for orthogonalisation
tol_test = 1e-12

# we project ϕ onto the orthogonal of ψ
function proj!(ϕ, ψ)

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

# KrylovKit custom orthogonaliser to be used in KrylovKit eigsolve, svdsolve,
# linsolve, ...
pack(ψ) = vcat(Base.vec.(ψ)...)
struct OrthogonalizeAndProject{F, O <: Orthogonalizer, ψ} <: Orthogonalizer
    projector!::F
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
    v = pack(alg.projector!(v, alg.ψ))::T
    v, x
end
function KrylovKit.orthogonalize!(v::T, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, x, alg.orth)
    v = reshape(v, size(alg.ψ))
    v = pack(alg.projector!(v, alg.ψ))::T
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
    proj!(Ωδφ, φ)
end

# compute the application of K
function apply_K(basis, δφ, φ, ρ, occ)
    Nk = length(basis.kpoints)

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
    proj!(Kδφ, φ)
end

# Apply (Ω+K)δφ
function ΩplusK(basis, δφ, φ, ρ, H, egval, occ)
    Kδφ = apply_K(basis, δφ, φ, ρ, occ)
    Ωδφ = apply_Ω(basis, δφ, φ, H, egval)
    ΩpKδφ = Ωδφ .+ Kδφ
end

################################## TESTS #######################################
function test_Ω(basis::PlaneWaveBasis{T};
                ψ0=nothing) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = DFTK.filled_occupation(model)
    n_bands = div(model.n_electrons, filled_occ)

    ## number of kpoints
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]
    ## number of eigenvalue/eigenvectors we are looking for
    N = n_bands

    ortho(ψk) = Matrix(qr(ψk).Q)
    if ψ0 === nothing
        ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), n_bands))
              for kpt in basis.kpoints]
    end

    ## vec and unpack
    # length of ψ0[ik]
    lengths = [length(ψ0[ik]) for ik = 1:Nk]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    pack(ψ) = vcat(Base.vec.(ψ)...) # TODO as an optimization, do that lazily? See LazyArrays
    unpack(ψ) = [@views reshape(ψ[starts[ik]:starts[ik]+lengths[ik]-1], size(ψ0[ik]))
                 for ik = 1:Nk]

    φ = similar(scfres.ψ)
    H = scfres.ham
    for ik = 1:Nk
        φ[ik] = scfres.ψ[ik][:,1:4]
    end
    egval = [ zeros(Complex{T}, size(occupation[i])) for i = 1:length(occupation) ]
    for ik = 1:Nk
        φk = φ[ik]
        Hk = H.blocks[ik]
        egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
        egval[ik] = egvalk
    end

    function f(x)
        δφ = unpack(x)
        δφ = proj!(δφ, φ)
        ΩpKx = apply_Ω(basis, δφ, φ, H, egval)
        ΩpKx = proj!(ΩpKx, φ)
        pack(ΩpKx)
    end

    packed_proj!(ϕ,ψ) = proj!(unpack(ϕ), unpack(ψ))
    vals_Ω, vecs_Ω, info = eigsolve(f, pack(ψ0), 8, :SR;
                                    tol=1e-6, verbosity=1, eager=true,
                                    maxiter=50, krylovdim=30,
                                    orth=OrthogonalizeAndProject(packed_proj!, pack(φ)))

    display(vals_Ω)
    # should match the smallest eigenvalue of Ω
    println(scfres.eigenvalues[1][5] - scfres.eigenvalues[1][4])

    ## testing symmetry
    ψ1 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]
    ψ1 = proj!(ψ1, φ)
    ΩpKψ1 = ΩplusK(basis, ψ1, φ, ρ, H, egval, occupation)
    ψ2 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]
    ψ2 = proj!(ψ1, φ)
    ΩpKψ2 = ΩplusK(basis, ψ2, φ, ρ, H, egval, occupation)
    println(norm(ψ1'ΩpKψ2 - ψ2'ΩpKψ1))
end
