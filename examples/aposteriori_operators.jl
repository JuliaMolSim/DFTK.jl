import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

# This file containes functions for computing the jacobian of the direct
# minimization algorihtm on the grassman manifold, that this the operator Ω+K
# defined Cancès/Kemlin/Levitt, Convergence analysis of SCF and direct
# minization algorithms.


################################## TOOL ROUTINES ###############################

# norm of difference between the density matrices associated to ϕ and ψ
# to be consistent with |δφ|^2 = 2 Σ |δφi|^2 when δφ = Σ |φi><δφi| + hc is an
# element on the tangent space, we return (1/√2)|ϕϕ'-ψψ'| = √(N-|ϕ'ψ|^2) so that
# we can take Σ |δφi|^2 as norm of δφ for δφ an element of the tangent space
function dm_distance(ϕ, ψ)
    N = size(ϕ,2)

    # use higher precision to compute √(N-|ϕ'ψ|^2) with |ϕ'ψ|^2 close to N
    ϕ = Complex{BigFloat}.(ϕ)
    ψ = Complex{BigFloat}.(ψ)
    ortho(ψk) = Matrix(qr(ψk).Q)
    ϕ = ortho(ϕ)
    ψ = ortho(ψ)

    ϕψ = norm(ϕ'ψ)
    sqrt(abs(N - ϕψ^2))
end

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

# apply preconditioner M^{1/2}
function apply_sqrt(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= sqrt.(Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, ϕk)
    end
    ϕ
end

# apply preconditioner M^{-1/2}
function apply_inv_sqrt(Pks, res)
    Nk = length(Pks)

    R = []

    for ik = 1:Nk
        Rk = similar(res[ik])
        N = size(res[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            Rk[:,i] .= 1 ./ sqrt.(Pk.mean_kin[i] .+ Pk.kin) .* res[ik][:,i]
        end
        append!(R, Rk)
    end
    R
end

# compute operator norm of Ω+K defined at φ
function compute_normop(basis::PlaneWaveBasis{T}, φ, ρ, H, egval,
                        occupation, packing; tol_krylov=1e-12, Pks=nothing) where T

    N = size(φ[1],2)
    Nk = size(φ)
    pack, unpack, packed_proj! = packing

    ## random starting point for eigensolvers
    ortho(ψk) = Matrix(qr(ψk).Q)
    ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]

    function f(δφ)
        δφ = proj!(δφ, φ)
        ΩpKx = ΩplusK(basis, δφ, φ, ρ, H, egval, occupation)
        ΩpKx = proj!(ΩpKx, φ)
    end

    # svd solve
    function g(x,flag)
        δφ = unpack(x)
        if Pks != nothing
            δφ = proj!(δφ, φ)
            apply_sqrt(Pks, δφ)
        end
        ΩpKδφ = f(δφ)
        if Pks != nothing
            apply_sqrt(Pks, δφ)
            δφ = proj!(δφ, φ)
        end
        pack(ΩpKδφ)
    end
    svds_ΩpK, _ = svdsolve(g, pack(ψ0), 3, :SR;
                           tol=tol_krylov, verbosity=0, eager=true,
                           orth=OrthogonalizeAndProject(packed_proj!, pack(φ)))

    normop = 1. / svds_ΩpK[1]
    println("--> plus petite valeur singulière $(svds_ΩpK[1])")
    println("--> normop $(normop)")
    println("--> gap $(egval[1][N+1] - egval[1][N])")
    normop
end

############################# SCF CALLBACK ####################################

## custom callback to follow estimators
function callback_estimators(; test_newton=false, change_norm=true)

    global ite, φ_list, basis_list
    φ_list = []                 # list of φ^k
    basis_list = []
    ite = 0

    function callback(info)

        if info.stage == :finalize

            println("Starting post-treatment")

            basis_ref = info.basis
            model = info.basis.model

            ## number of kpoints
            Nk = length(basis_ref.kpoints)
            ## number of eigenvalue/eigenvectors we are looking for
            filled_occ = DFTK.filled_occupation(model)
            N = div(model.n_electrons, filled_occ)
            occupation = [filled_occ * ones(N) for ik = 1:Nk]

            φ_ref = similar(info.ψ)
            for ik = 1:Nk
                φ_ref[ik] = info.ψ[ik][:,1:N]
            end

            ## converged values
            ρ_ref = info.ρ
            H_ref = info.ham
            egval_ref = info.eigenvalues
            T = typeof(ρ_ref.real[1])


            ## packing routines to pack vectors for KrylovKit solver
            packing = packing_routines(basis_ref, φ_ref)

            ## filling residuals and errors
            err_ref_list = []
            norm_res_list = []
            if test_newton
                err_newton_list = []
                norm_δφ_list = []
            end

            ## preconditioner for changing norm if asked so
            if change_norm
                Pks = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
                for ik = 1:length(Pks)
                    DFTK.precondprep!(Pks[ik], φ_ref[ik])
                end
                norm_Pkres_list = []
                err_Pkref_list = []
            else
                Pks = nothing
            end

            println("Computing residual...")
            for i in 1:ite
                println("   iteration $(i)")
                φ = φ_list[i]
                basis = basis_list[i]
                for ik = 1:Nk
                    φ[ik] = φ[ik][:,1:N]
                end

                res, ρ, H, egval = compute_residual(basis, φ, occupation;
                                          test_newton=test_newton, φproj=φ_ref)
                append!(err_ref_list, dm_distance(φ[1], φ_ref[1]))
                append!(norm_res_list, norm(res))

                if change_norm
                    append!(norm_Pkres_list, norm(apply_inv_sqrt(Pks, res)))
                    Pkφ = apply_sqrt(Pks, φ)
                    Pkφ_ref = apply_sqrt(Pks, φ_ref)
                    append!(err_Pkref_list, dm_distance(Pkφ[1], Pkφ_ref[1]))
                end

                if test_newton
                    φ_newton, δφ = newton_step(basis_ref, φ, res, ρ_ref, H_ref,
                                               egval_ref, occupation, packing;
                                               tol_krylov=tol_krylov, φproj=φ_ref)
                    append!(err_newton_list, dm_distance(φ_newton[1], φ_ref[1]))
                    append!(norm_δφ_list, norm(δφ))
                end
            end

            ## error estimates
            println("Computing operator norm...")
            normop = compute_normop(basis_ref, φ_ref, ρ_ref, H_ref,
                                    egval_ref, occupation,
                                    packing; tol_krylov=tol_krylov, Pks=nothing)
            err_estimator = normop .* norm_res_list
            if change_norm
                normop = compute_normop(basis_ref, φ_ref, ρ_ref, H_ref,
                                        egval_ref, occupation,
                                        packing; tol_krylov=tol_krylov, Pks=Pks)
                err_Pk_estimator = normop .* norm_Pkres_list
            end

            ## plotting convergence info
            figure()
            title("error estimators vs iteration, N = $(N)")
            semilogy(1:ite, err_ref_list, "x-", label="|P-Pref|")
            semilogy(1:ite, norm_res_list, "x-", label="|res_φ|")
            semilogy(1:ite, err_estimator, "x-", label="|(Ω+K)^-1| * |res_φ|")
            if test_newton
                semilogy(1:ite, norm_δφ_list, "+-", label="|δφ|")
                semilogy(1:ite, err_newton_list, "x-", label="|P_newton-Pref|")
            end
            if change_norm
                semilogy(1:ite, err_Pkref_list, "x--", label="|M^1/2(P-Pref)|")
                semilogy(1:ite, norm_Pkres_list, "x--", label="|M^-1/2res_φ|")
                semilogy(1:ite, err_Pk_estimator, "x--", label="|M^1/2(Ω+K)^-1M^1/2| * |M^-1/2res_φ|")
            end
            legend()
            xlabel("iterations")

        else
            ite += 1
            append!(φ_list, [info.ψ])
            append!(basis_list, [info.basis])
        end
    end
    callback
end
