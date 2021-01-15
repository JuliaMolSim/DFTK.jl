using KrylovKit

# This file containes functions for computing the jacobian of the direct
# minimization algorihtm on the grassman manifold, that this the operator Ω+K
# defined Cancès/Kemlin/Levitt, Convergence analysis of SCF and direct
# minization algorithms.

############################## CHANGES OF NORMS ################################

# apply preconditioner M
function apply_M(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= (Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
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
        append!(ϕ, [ϕk])
    end
    ϕ
end

# apply preconditioner M^{-1}
function apply_inv_M(Pks, res)
    Nk = length(Pks)

    Res = []

    for ik = 1:Nk
        Rk = similar(res[ik])
        N = size(res[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            Rk[:,i] .= (1. ./ (Pk.mean_kin[i] .+ Pk.kin)) .* res[ik][:,i]
        end
        append!(Res, [Rk])
    end
    Res
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
            Rk[:,i] .= (1 ./ sqrt.(Pk.mean_kin[i] .+ Pk.kin)) .* res[ik][:,i]
        end
        append!(R, [Rk])
    end
    R
end

# compute operator norm of (Ω+K)^-1 defined at φ
function compute_normop_invΩpK(basis::PlaneWaveBasis{T}, φ, occ;
                                       tol_krylov=1e-12, Pks=nothing, change_norm=false) where T

    ## necessary quantities
    N = size(φ[1],2)
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])
    egval = [ zeros(Complex{T}, size(occ[ik])) for ik = 1:Nk ]
    for ik = 1:Nk
        Hk = H.blocks[ik]
        φk = φ[ik]
        egval[ik] = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
    end

    ## random starting point for eigensolvers
    ortho(ψk) = Matrix(qr(ψk).Q)
    ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]

    function f(δφ)
        ΩpKδφ = ΩplusK(basis, δφ, φ, ρ[1], H, egval, occ)
    end

    # packing routines
    pack, unpack, packed_proj = packing(basis, φ)

    # svd solve
    function g(x,flag)
        δφ = unpack(x)
        if Pks != nothing
            δφ = proj(δφ, φ)
            δφ = apply_sqrt(Pks, δφ)
        end
        ΩpKδφ = f(δφ)
        if Pks != nothing
            δφ = apply_sqrt(Pks, δφ)
            δφ = proj(δφ, φ)
        end
        pack(ΩpKδφ)
    end
    svd_SR, _ = svdsolve(g, pack(ψ0), 3, :SR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    svd_LR, _ = svdsolve(g, pack(ψ0), 3, :LR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))

    normop = 1. / svd_SR[1]
    if !change_norm
        println("--> plus petite valeur singulière (Ω+K) $(svd_SR[1])")
        println("--> normop (Ω+K)^-1 $(normop)")
    else
        println("--> plus petite valeur singulière M^-1/2(Ω+K)M^-1/2 $(svd_SR[1])")
        println("--> normop M^1/2(Ω+K)^-1M^1/2 $(normop)")
    end
    (normop=normop, svd_min=svd_SR[1], svd_max=svd_LR[1])
end

function compute_normop_invε(basis::PlaneWaveBasis{T}, φ, occ;
                             tol_krylov=1e-12, Pks=nothing, change_norm=false) where T

    ## necessary quantities
    N = size(φ[1],2)
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])
    egval = [ zeros(Complex{T}, size(occ[ik])) for ik = 1:Nk ]
    for ik = 1:Nk
        Hk = H.blocks[ik]
        φk = φ[ik]
        egval[ik] = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
    end

    # packing routines
    pack, unpack, packed_proj = packing(basis, φ)

    ## random starting point for eigensolvers
    ortho(ψk) = Matrix(qr(ψk).Q)
    ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]

    function invΩ(basis, δφ, φ, H, egval)
        function Ω(x)
            δϕ = unpack(x)
            Ωδϕ = apply_Ω(basis, δϕ, φ, H, egval)
            pack(Ωδϕ)
        end
        invΩδφ, info = linsolve(Ω, pack(δφ);
                                tol=tol_krylov, verbosity=1,
                                orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
        unpack(invΩδφ)
    end

    function f(δφ,flag)
        if flag
            invΩδφ = invΩ(basis, δφ, φ, H, egval)
            KinvΩδφ = apply_K(basis, invΩδφ, φ, ρ[1], occ)
            return proj(δφ .+ KinvΩδφ, φ)
        else
            Kδφ = apply_K(basis, δφ, φ, ρ[1], occ)
            invΩKδφ = invΩ(basis, Kδφ, φ, H, egval)
            return proj(δφ .+ invΩKδφ, φ)
        end
    end

    # svd solve
    function g(x,flag)
        if flag
            δφ = unpack(x)
            if Pks != nothing
                δφ = proj(δφ, φ)
                δφ = apply_sqrt(Pks, δφ)
            end
            δφ = f(δφ, flag)
            if Pks != nothing
                δφ = apply_inv_sqrt(Pks, δφ)
                δφ = proj(δφ, φ)
            end
            return pack(δφ)
        else
            δφ = unpack(x)
            if Pks != nothing
                δφ = proj(δφ, φ)
                δφ = apply_inv_sqrt(Pks, δφ)
            end
            δφ = f(δφ, flag)
            if Pks != nothing
                δφ = apply_sqrt(Pks, δφ)
                δφ = proj(δφ, φ)
            end
            return pack(δφ)
        end
    end
    svd_SR, _ = svdsolve(g, pack(ψ0), 3, :SR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    svd_LR, _ = svdsolve(g, pack(ψ0), 3, :LR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))

    normop = 1. / svd_SR[1]
    if !change_norm
        println("--> plus petite valeur singulière ε $(svd_SR[1])")
        println("--> normop ε^-1 $(normop)")
    else
        println("--> plus petite valeur singulière M^1/2εM^-1/2 $(svd_SR[1])")
        println("--> normop M^1/2ε^-1M^-1/2 $(normop)")
    end
    (normop=normop, svd_min=svd_SR[1], svd_max=svd_LR[1])
end

function compute_normop_invΩ(basis::PlaneWaveBasis{T}, φ, occ;
                             tol_krylov=1e-12, Pks=nothing, change_norm=false) where T

    ## necessary quantities
    N = size(φ[1],2)
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])
    egval = [ zeros(Complex{T}, size(occ[ik])) for ik = 1:Nk ]
    for ik = 1:Nk
        Hk = H.blocks[ik]
        φk = φ[ik]
        egval[ik] = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
    end

    # packing routines
    pack, unpack, packed_proj = packing(basis, φ)

    ## random starting point for eigensolvers
    ortho(ψk) = Matrix(qr(ψk).Q)
    ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]

    function f(δφ)
        Ωδφ = apply_Ω(basis, δφ, φ, H, egval)
    end

    # svd solve
    function g(x,flag)
        δφ = unpack(x)
        if Pks != nothing
            δφ = proj(δφ, φ)
            δφ = apply_inv_sqrt(Pks, δφ)
        end
        δφ = f(δφ)
        if Pks != nothing
            δφ = apply_inv_sqrt(Pks, δφ)
            δφ = proj(δφ, φ)
        end
        pack(δφ)
    end
    svd_SR, _ = svdsolve(g, pack(ψ0), 3, :SR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    svd_LR, _ = svdsolve(g, pack(ψ0), 3, :LR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))

    normop = 1. / svd_SR[1]
    if !change_norm
        println("--> plus petite valeur singulière Ω $(svd_SR[1])")
        println("--> normop Ω^-1 $(normop)")
    else
        println("--> plus petite valeur singulière M^-1/2ΩM^-1/2 $(svd_SR[1])")
        println("--> normop M^1/2Ω^-1M^1/2 $(normop)")
    end
    (normop=normop, svd_min=svd_SR[1], svd_max=svd_LR[1])
end

############################# SCF CALLBACK ####################################

## custom callback to follow estimators
function callback_estimators(; test_newton=false, change_norm=true)

    global ite, φ_list
    φ_list = []                 # list of φ^k
    ite = 0

    function callback(info)

        if info.stage == :finalize

            println("Starting post-treatment")

            basis = info.basis
            model = info.basis.model

            ## number of kpoints
            Nk = length(basis.kpoints)
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

            ## filling residuals and errors
            err_ref_list = []
            norm_res_list = []
            if test_newton
                err_newton_list = []
                norm_δφ_list = []
            end

            ## preconditioner for changing norm if asked so
            if change_norm
                Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
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
                for ik = 1:Nk
                    φ[ik] = φ[ik][:,1:N]
                end

                res = compute_residual(basis, φ, occupation)
                append!(err_ref_list, dm_distance(φ[1], φ_ref[1]))
                append!(norm_res_list, norm(res))

                if change_norm
                    append!(norm_Pkres_list, norm(apply_inv_sqrt(Pks, res)))
                    append!(err_Pkref_list, dm_distance(φ[1], φ_ref[1], Pks))
                end

                if test_newton
                    φ_newton, δφ = newton_step(basis, φ, res, occupation;
                                               tol_krylov=tol_krylov, φproj=φ_ref)
                    append!(err_newton_list, dm_distance(φ_newton[1], φ_ref[1]))
                    append!(norm_δφ_list, norm(δφ))
                end
            end

            ## error estimates
            println("--------------------------------")
            println("Computing operator norms...")
            println("--> gap $(egval_ref[1][N+1] - egval_ref[1][N])")
            normop_ΩpK, svd_min_ΩpK, svd_max_ΩpK = compute_normop_invΩpK(basis, φ_ref, occupation;
                                                                         tol_krylov=tol_krylov, Pks=nothing)
            err_estimator = normop_ΩpK .* norm_res_list
            if change_norm
                normop_invΩ, svd_min_Ω, svd_max_Ω = compute_normop_invΩ(basis, φ_ref, occupation;
                                                                        tol_krylov=tol_krylov, Pks=Pks)
                normop_invε, svd_min_ε, svd_max_ε = compute_normop_invε(basis, φ_ref, occupation;
                                                                        tol_krylov=tol_krylov, Pks=Pks)
                err_Pk_estimator = normop_invε .* normop_invΩ .* norm_Pkres_list
            end

            ## plotting convergence info
            figure(figsize=(10,5))
            title("GaAs
                  error estimators vs iteration, LDA, N = $(N), M = (1-Δ),
                  (Ω+K)^-1 : norm = $(@sprintf("%.3f", normop_ΩpK)), min_svd = $(@sprintf("%.3f", svd_min_ΩpK)), max_svd = $(@sprintf("%.3f", svd_max_ΩpK))
                  ε^-1 : norm = $(@sprintf("%.3f", normop_invε)), min_svd = $(@sprintf("%.3f", svd_min_ε)), max_svd = $(@sprintf("%.3f", svd_max_ε))
                  Ω^-1 : norm = $(@sprintf("%.3f", normop_invΩ)), min_svd = $(@sprintf("%.3f", svd_min_Ω)), max_svd = $(@sprintf("%.3f", svd_max_Ω))")
            semilogy(1:(ite-1), err_ref_list[1:end-1], "x-", label="|P-Pref|")
            semilogy(1:ite, norm_res_list, "x-", label="|res_φ|")
            semilogy(1:ite, err_estimator, "x-", label="|(Ω+K)^-1| * |res_φ|")
            if test_newton
                semilogy(1:ite, norm_δφ_list, "+-", label="|δφ|")
                semilogy(1:ite, err_newton_list, "x-", label="|P_newton-Pref|")
            end
            if change_norm
                semilogy(1:(ite-1), err_Pkref_list[1:end-1], "x--", label="|M^1/2(P-Pref)|")
                semilogy(1:ite, norm_Pkres_list, "x--", label="|M^-1/2res_φ|")
                semilogy(1:ite, err_Pk_estimator, "x--", label="|M^1/2(Ω+K)^-1M^1/2| * |M^-1/2res_φ|")
            end
            legend()
            xlabel("iterations")
            savefig("GaAs_SCF.pdf")
        else
            ite += 1
            append!(φ_list, [info.ψ])
        end
    end
    callback
end
