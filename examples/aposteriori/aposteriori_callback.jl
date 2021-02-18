using KrylovKit

############################### OPERATOR NORMS #################################

## projection on frequencies higher than Ecut
function keep_HF(δϕ, basis, Ecut)

    Nk = length(basis.kpoints)

    δφ = deepcopy(δϕ)

    for ik in 1:Nk
        kpt = basis.kpoints[ik]
        G_vec = G_vectors(kpt)
        recip_lat = kpt.model.recip_lattice
        N = size(δφ[ik], 2)

        for i in 1:N
            for g in 1:length(δφ[ik][:,i])
                if sum(abs2, recip_lat * (G_vec[g] + kpt.coordinate)) <= 2*Ecut
                    δφ[ik][g,i] = 0
                end
            end
        end
    end

    δφ
end

## projection on frequencies smaller than Ecut
function keep_LF(δϕ, basis, Ecut)

    Nk = length(basis.kpoints)

    δφ = deepcopy(δϕ)

    for ik in 1:Nk
        kpt = basis.kpoints[ik]
        G_vec = G_vectors(kpt)
        recip_lat = kpt.model.recip_lattice
        N = size(δφ[ik], 2)

        for i in 1:N
            for g in 1:length(δφ[ik][:,i])
                if sum(abs2, recip_lat * (G_vec[g] + kpt.coordinate)) > 2*Ecut
                    δφ[ik][g,i] = 0
                end
            end
        end
    end

    δφ
end

function compute_normop(basis::PlaneWaveBasis{T}, φ, occ;
                        tol_krylov=1e-12, Pks=nothing, change_norm=false,
                        high_freq=false, low_freq=false, Ecut=nothing, nl=false) where T

    ## ensure that high_freq and low_freq are not true at the same time or false
    ## at the same time
    @assert !(high_freq && low_freq)

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

    if high_freq
        @assert Ecut != nothing
        @assert Nk == 1 [println("high_freq only valid for 1 kpt yet")]
        println("\nComputing M^1/2Ω^-1M^1/2 on high frequencies...")
    elseif low_freq
        @assert Ecut != nothing
        @assert Nk == 1 [println("high_freq only valid for 1 kpt yet")]
        println("\nComputing M^1/2Ω^-1M^1/2 on low frequencies...")
    end

    # packing routines
    pack, unpack, packed_proj = packing(basis, φ)
    pack, unpack, packed_proj_freq = packing(basis, φ;
                                             high_freq=high_freq,
                                             low_freq=low_freq, Ecut=Ecut)

    ## random starting point for eigensolvers
    ortho(ψk) = Matrix(qr(ψk).Q)
    ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]
    if high_freq
        ψ0 = keep_HF(ψ0, basis, Ecut)
    elseif low_freq
        ψ0 = keep_LF(ψ0, basis, Ecut)
    end

    function inv_op(δφ)
        function op(x)
            δϕ = unpack(x)
            if !nl
                δϕ = apply_Ω(basis, δϕ, φ, H, egval)
            else
                δϕ = ΩplusK(basis, δϕ, φ, ρ[1], H, egval, occ)
            end
            if Pks != nothing
                δϕ = apply_inv_T(Pks, δϕ)
            end
            pack(δϕ)
        end
        if Pks != nothing
            δφ = apply_inv_T(Pks, δφ)
        end
        δφ, info = linsolve(op, pack(δφ);
                            tol=tol_krylov, verbosity=0,
                            krylovdim=2, maxiter=1,
                            orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
        unpack(δφ)
    end

    # svd solve
    function g(x,flag)
        δφ = unpack(x)
        if high_freq || (low_freq && !flag)
            δφ = keep_HF(δφ, basis, Ecut)
        elseif low_freq && flag
            δφ = keep_LF(δφ, basis, Ecut)
        end
        if Pks != nothing
            δφ = apply_sqrt_M(φ, Pks, δφ)
        else
            δφ = proj(δφ, φ)
        end
        δφ = inv_op(δφ)
        if Pks != nothing
            δφ = apply_sqrt_M(φ, Pks, δφ)
        else
            δφ = proj(δφ, φ)
        end
        if high_freq || (low_freq && flag)
            δφ = keep_HF(δφ, basis, Ecut)
        elseif low_freq && !flag
            δφ = keep_LF(δφ, basis, Ecut)
        end
        pack(δφ)
    end
    svd_LR, _ = svdsolve(g, pack(ψ0), 2, :LR;
                         tol=tol_krylov, verbosity=1, eager=true,
                         orth=OrthogonalizeAndProject(packed_proj_freq, pack(φ)))
    #  svd_SR, _ = svdsolve(g, pack(ψ0), 2, :SR;
    #                       tol=tol_krylov, verbosity=1, eager=true,
    #                       orth=OrthogonalizeAndProject(packed_proj_freq, pack(φ)))

    normop = svd_LR[1]
    if !change_norm
        println("--> normop (Ω+K)^-1 $(normop)")
    else
        println("--> normop M^1/2(Ω+K)^-1M^1/2 $(normop)")
    end
    normop
end

############################# SCF CALLBACK ####################################

## custom callback to follow estimators
#  function callback_estimators(system; test_newton=false, change_norm=true)

#      global ite, φ_list
#      φ_list = []                 # list of φ^k
#      nrj_list = []
#      ite = 0

#      function callback(info)

#          if info.stage == :finalize

#              println("Starting post-treatment")

#              basis = info.basis
#              model = info.basis.model

#              ## number of kpoints
#              Nk = length(basis.kpoints)
#              ## number of eigenvalue/eigenvectors we are looking for
#              filled_occ = DFTK.filled_occupation(model)
#              N = div(model.n_electrons, filled_occ)
#              occupation = [filled_occ * ones(N) for ik = 1:Nk]

#              φ_ref = similar(info.ψ)
#              for ik = 1:Nk
#                  φ_ref[ik] = info.ψ[ik][:,1:N]
#              end

#              ## converged values
#              ρ_ref = info.ρ
#              H_ref = info.ham
#              egval_ref = info.eigenvalues
#              nrj_ref = info.energies.total

#              ## filling residuals and errors
#              err_ref_list = []
#              norm_res_list = []
#              nrj_ref_list = []
#              if test_newton
#                  err_newton_list = []
#                  norm_δφ_list = []
#              end

#              ## preconditioner for changing norm if asked so
#              if change_norm
#                  Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
#                  for ik = 1:length(Pks)
#                      DFTK.precondprep!(Pks[ik], φ_ref[ik])
#                  end
#                  norm_Pkres_list = []
#                  err_Pkref_list = []
#              else
#                  Pks = nothing
#              end

#              println("Computing residual...")
#              for i in 1:ite
#                  println("   iteration $(i)")
#                  φ = φ_list[i]
#                  for ik = 1:Nk
#                      φ[ik] = φ[ik][:,1:N]
#                  end

#                  res = compute_residual(basis, φ, occupation)
#                  err = compute_error(basis, φ, φ_ref)
#                  append!(err_ref_list, norm(err))
#                  append!(norm_res_list, norm(res))
#                  append!(nrj_ref_list, abs(nrj_list[i] - nrj_ref))

#                  if change_norm
#                      append!(norm_Pkres_list, norm(apply_inv_sqrt(Pks, res)))
#                      append!(err_Pkref_list, norm(apply_sqrt_M(Pks, err)))
#                  end

#                  if test_newton
#                      φ_newton, δφ = newton_step(basis, φ, res, occupation;
#                                                 tol_krylov=tol_krylov, φproj=φ_ref)
#                      append!(err_newton_list, dm_distance(φ_newton[1], φ_ref[1]))
#                      append!(norm_δφ_list, norm(δφ))
#                  end
#              end

#              ## error estimates
#              println("--------------------------------")
#              println("Computing operator norms...")
#              gap = egval_ref[1][N+1] - egval_ref[1][N]
#              println("--> gap $(gap)")
#              normop_ΩpK, svd_min_ΩpK, svd_max_ΩpK = compute_normop_invΩpK(basis, φ_ref, occupation;
#                                                                           tol_krylov=tol_krylov, Pks=nothing)
#              err_estimator = normop_ΩpK .* norm_res_list
#              if change_norm
#                  normop_invΩ, svd_min_Ω, svd_max_Ω = compute_normop_invΩ(basis, φ_ref, occupation;
#                                                                          tol_krylov=tol_krylov, Pks=Pks)
#                  normop_invε, svd_min_ε, svd_max_ε = compute_normop_invε(basis, φ_ref, occupation;
#                                                                          tol_krylov=tol_krylov, Pks=Pks)
#                  err_Pk_estimator = normop_invε .* normop_invΩ .* norm_Pkres_list
#              end

#              h5open(system*"_SCF.h5", "w") do file
#                  file["Ecut_ref"] = Ecut_ref
#                  file["ite"] = ite
#                  file["kgrid"] = kgrid
#                  file["N"] = N
#                  file["nl"] = nl
#                  file["gap"] = gap
#                  file["normop_invΩpK"] = normop_ΩpK
#                  file["svd_min_ΩpK"] = svd_min_ΩpK
#                  file["svd_max_ΩpK"] = svd_max_ΩpK
#                  file["normop_invΩ_kin"] = normop_invΩ
#                  file["svd_min_Ω_kin"] = svd_min_Ω
#                  file["svd_max_Ω_kin"] = svd_max_Ω
#                  file["normop_invε_kin"] = normop_invε
#                  file["svd_min_ε_kin"] = svd_min_ε
#                  file["svd_max_ε_kin"] = svd_max_ε
#                  file["norm_err_list"] = Float64.(err_ref_list)
#                  file["norm_res_list"] = Float64.(norm_res_list)
#                  file["err_estimator"] = Float64.(err_estimator)
#                  file["norm_Pk_kin_err_list"] = Float64.(err_Pkref_list)
#                  file["norm_Pk_kin_res_list"] = Float64.(norm_Pkres_list)
#                  file["err_Pk_estimator"] = Float64.(err_Pk_estimator)
#                  file["nrj_ref_list"] = Float64.(nrj_ref_list)
#              end

#          else
#              ite += 1
#              append!(φ_list, [info.ψ])
#              append!(nrj_list, info.energies.total)
#          end
#      end
#      callback
#  end
