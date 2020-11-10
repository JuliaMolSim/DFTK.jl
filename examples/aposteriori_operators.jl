import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

################################## TOOL ROUTINES ###############################

# test for orthogonalisation
tol_test = 1e-10

# for a given kpoint, we compute the
# projection of a vector ϕk on the orthogonal of the eigenvectors ψk
function proj!(ϕk, ψk)

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
function apply_Ω_kpt(δφk, ψk, H::HamiltonianBlock, egvalk)

    N1 = size(δφk,2)
    N2 = size(ψk,2)
    @assert N1 == N2
    N = N1

    Ωδφk = similar(δφk)

    Hδφk = H * δφk
    Hδφk = proj!(Hδφk, ψk)

    # compute component on i
    for i = 1:N
        ε_i = egvalk[i]
        Ωδφk[:,i] = Hδφk[:,i] - ε_i * δφk[:,i]
    end
    Ωδφk
end

# same but for all kpoint at a time
function apply_Ω(scfres, δφ, ψ, H::Hamiltonian, egval)
    Ωδφ = similar(δφ)
    for ik = 1:length(scfres.basis.kpoints)
        N = size(δφ[ik],2)
        Ωδφ[ik] = apply_Ω_kpt(δφ[ik], ψ[ik][:,1:N], H.blocks[ik], egval[ik])
    end
    Ωδφ
end

# we compute the application of K for on kpt
function apply_K_kpt(scfres, kpt, δφk, Kδρk, ψk)

    basis = scfres.basis

    Kδρk_r = Kδρk.real
    Kδρψk = similar(δφk)

    N = size(δφk,2)

    for i = 1:N
        ψk_r = G_to_r(basis, kpt, ψk[:,i])
        Kδρψk_r = Kδρk_r .* ψk_r
        Kδρψk[:,i] = r_to_G(basis, kpt, Kδρψk_r)
    end
    proj!(Kδρψk, ψk)
end

# we compute the application of K (for all kpoints)
function apply_K(scfres, δφ, ψ)

    basis = scfres.basis
    occ = similar(scfres.occupation)

    for ik = 1:length(basis.kpoints)
        N = size(δφ[ik],2)
        occ[ik] = scfres.occupation[ik][1:N]
    end

    δρ = compute_density(basis, δφ, occ)
    Kδρ = apply_kernel(basis, δρ[1]; ρ=scfres.ρ)
    Kδφ = similar(δφ)

    for ik = 1:length(basis.kpoints)
        kpt = basis.kpoints[ik]
        Kδφ[ik] = apply_K_kpt(scfres, kpt, δφk, Kδρ[ik], ψk)
    end
    Kδφ
end

# Ω+K at a given kpt
function ΩplusK_kpt(scfres, kpt, δφk, Kδρk, ψk, Hk, egvalk)

    ΩpKk = similar(δφk)

    Kδφk = apply_K_kpt(scfres, kpt, δφk, Kδρk, ψk)
    Ωδφk = apply_Ω_kpt(δφk, ψk, Hk, egvalk)
    ΩpKδφk = Ωδφk .+ Kδφk
end

# Apply (Ω+K)δφ
function ΩplusK(scfres, δφ)

    basis = scfres.basis

    ψ = scfres.ψ
    H = scfres.ham
    occ = scfres.occupation
    egval = scfres.eigenvalues

    Kδφ = apply_K(scfres, δφ, ψ)
    Ωδφ = apply_Ω(scfres, δφ, ψ, H, egval)
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
