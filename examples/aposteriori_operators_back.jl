import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

################################## TOOL ROUTINES ###############################

# test for orthogonalisation
tol_test = 1e-10

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

# for a given kpoint, we compute the
# projection of a vector ϕk on the orthogonal of the eigenvectors ψk
function proj_kpt!(ϕk, ψk)

    N1 = size(ϕk,2)
    N2 = size(ψk,2)
    @assert N1 == N2
    N = N1

    Πϕk = deepcopy(ϕk)
    for i = 1:N, j = 1:N
        Πϕk[:,i] -= (ψk[:,j]'ϕk[:,i]) * ψk[:,j]
    end
    for i = 1:N, j = 1:N
        @assert abs(Πϕk[:,i]'ψk[:,j]) < tol_test [println(abs(Πϕk[:,i]'ψk[:,j]))]
    end
    ϕk = Πϕk
    ϕk
end

# KrylovKit custom orthogonaliser
struct OrthogonalizeAndProject{F, O <: Orthogonalizer, ψk} <: Orthogonalizer
    projector!::F
    orth::O
    ψ::ψk
end
OrthogonalizeAndProject(projector, ψk) = OrthogonalizeAndProject(projector,
                                                                 KrylovDefaults.orth,
                                                                 ψk)
function KrylovKit.orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, b, x, alg.orth)
    v = reshape(v, size(alg.ψ))
    v = vec(alg.projector!(v, alg.ψ))::T
    v, x
end
function KrylovKit.orthogonalize!(v::T, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, x, alg.orth)
    v = reshape(v, size(alg.ψ))
    v = vec(alg.projector!(v, alg.ψ))::T
    v, x
end

# generate random δφ that are all orthogonal to every ψi for 1 <= i <= N
function generate_δφ(ψk)

    N = size(ψk,2)

    # generate random vector and project it
    δφk = rand(typeof(ψk[1,1]), size(ψk))
    δφk = proj!(δφk, ψk)

    # normalization and test
    for i = 1:N
        δφk[:,i] /= norm(δφk[:,i])
        for j = 1:N
            @assert abs(δφk[:,i]'ψk[:,j]) < tol_test [println(abs(δφk[:,i]'ψk[:,j]))]
        end
    end
    δφk
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
    Ωδφ = similar(δφ)

    for ik = 1:Nk
        δφk = δφ[ik]
        φk = φ[ik]
        egvalk = egval[ik]

        N1 = size(δφk,2)
        N2 = size(ψk,2)
        @assert N1 == N2
        N = N1

        Ωδφk = similar(δφk)

        Hδφk = H * δφk
        Hδφk = proj!([Hδφk], [ψk])

        # compute component on i
        for i = 1:N
            ε_i = egvalk[i]
            Ωδφk[:,i] = Hδφk[:,i] - ε_i * δφk[:,i]
        end
        Ωδφ[ik] = Ωδφk
    end
    Ωδφ
end

# we compute the application of K (for all kpoints)
function apply_K(basis, δφ, φ)

    δρ = DFTK.compute_density(basis, φ, δφ, occupation)
    Kδρ = apply_kernel(basis, δρ[1]; ρ=ρ[1])
    Kδφ = similar(δφ)

    for ik = 1:Nk
        kpt = basis.kpoints[ik]
        Kδρk_r = Kδρ[ik].real
        φk = φ[ik]
        Kδρφk = similar(φk)

        N = size(φk,2)
        for i = 1:N
            φk_r = G_to_r(basis, kpt, φk[:,i])
            Kδρφk_r = Kδρk_r .* φk_r
            Kδρφk[:,i] = r_to_G(basis, kpt, Kδρφk_r)
        end
        Kδφ[ik] = Kδρφk
    end
    proj!(Kδφ, φ)
end

# Apply (Ω+K)δφ
function ΩplusK(basis, δφ, φ, H, egval)
    Kδφ = apply_K(basis, δφ, φ)
    Ωδφ = apply_Ω(basis, δφ, φ, H, egval)
    ΩpKδφ = Ωδφ .+ Kδφ
end

############################## TESTS ##########################################

# Compare eigenvalues of Ω with the gap
function validate_Ω(scfres)

    ψ = scfres.ψ
    δφ = similar(ψ)

    basis = scfres.basis
    occ = scfres.occupation
    egval = scfres.eigenvalues
    H = scfres.ham

    vecs = []
    vals = []
    gap = nothing

    for ik = 1:length(basis.kpoints)

        occk = occ[ik]
        egvalk = egval[ik]
        N = length([l for l in occk if l != 0.0])
        gap = egvalk[N+1] - egvalk[N]

        ψk = ψ[ik][:,1:N]
        Hk = H.blocks[ik]
        δφ[ik] = generate_δφ(ψk)
        δφk = δφ[ik]
        x0 = vec(δφk)

        # function we want to compute eigenvalues
        function f(x)
            ncalls += 1
            x = reshape(x, size(δφk))
            x = proj!(x, ψk)
            Ωx = apply_Ω_kpt(x, ψk, Hk, egvalk)
            Ωx = proj!(Ωx, ψk)
            vec(Ωx)
        end

        println("\n--------------------------------")
        println("Solving with KrylovKit...")
        ncalls = 0
        vals_Ω, vecs_Ω, info = eigsolve(f, x0, 8, :SR;
                                        tol=1e-6, verbosity=1, eager=true,
                                        maxiter=50, krylovdim=30,
                                        orth=OrthogonalizeAndProject(proj!, ψk))

        println("\nKryloKit calls to operator: ", ncalls)
        idx = findfirst(x -> abs(x) > 1e-6, vals_Ω)
        display(vals_Ω)
        println("\n")
        display(info.normres)
        println("\n")
        display(vals_Ω[idx])
        display(gap)
        display(norm(gap-vals_Ω[idx]))
        append!(vecs, vecs_Ω)
        append!(vals, vals_Ω)
    end

    vals, vecs, gap
end

#  vals, vecs, gap = validate_Ω(scfres)
