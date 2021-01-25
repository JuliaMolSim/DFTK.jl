using KrylovKit

############################### OPERATOR NORMS #################################

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
                δφ = proj(δφ, φ)
            end
            δφ = f(δφ, flag)
            if Pks != nothing
                δφ = proj(δφ, φ)
                δφ = apply_inv_sqrt(Pks, δφ)
                δφ = proj(δφ, φ)
            end
            return pack(δφ)
        else
            δφ = unpack(x)
            if Pks != nothing
                δφ = proj(δφ, φ)
                δφ = apply_inv_sqrt(Pks, δφ)
                δφ = proj(δφ, φ)
            end
            δφ = f(δφ, flag)
            if Pks != nothing
                δφ = proj(δφ, φ)
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
            δφ = proj(δφ, φ)
        end
        δφ = f(δφ)
        if Pks != nothing
            δφ = proj(δφ, φ)
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
function callback_estimators(system; test_newton=false, change_norm=true)

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
                err = compute_error(basis, φ, φ_ref)
                append!(err_ref_list, norm(err))
                append!(norm_res_list, norm(res))

                if change_norm
                    append!(norm_Pkres_list, norm(apply_inv_sqrt(Pks, res)))
                    append!(err_Pkref_list, norm(apply_sqrt(Pks, err)))
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

            h5open(system*"_SCF.h5", "w") do file
                file["Ecut_ref"] = Ecut_ref
                file["ite"] = ite
                file["kgrid"] = kgrid
                file["N"] = N
                file["gap"] = gap
                file["normop_invΩpK"] = normop_ΩpK
                file["svd_min_ΩpK"] = svd_min_ΩpK
                file["svd_max_ΩpK"] = svd_max_ΩpK
                file["normop_invΩ_kin"] = normop_invΩ
                file["svd_min_Ω_kin"] = svd_min_Ω
                file["svd_max_Ω_kin"] = svd_max_Ω
                file["normop_invε_kin"] = normop_invε
                file["svd_min_ε_kin"] = svd_min_ε
                file["svd_max_ε_kin"] = svd_max_ε
                file["norm_err_list"] = Float64.(err_ref_list)
                file["norm_res_list"] = Float64.(norm_res_list)
                file["err_estimator"] = Float64.(err_estimator)
                file["norm_Pk_kin_err_list"] = Float64.(err_Pkref_list)
                file["norm_Pk_kin_res_list"] = Float64.(norm_Pkres_list)
                file["err_Pk_estimator"] = Float64.(err_Pk_estimator)
            end

        else
            ite += 1
            append!(φ_list, [info.ψ])
        end
    end
    callback
end
