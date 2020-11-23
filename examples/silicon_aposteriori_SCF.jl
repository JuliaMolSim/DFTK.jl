# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# when the error we look at is the SCF error : P* is computed as the converged
# density matrix and then we measure the error P-P* and the residual along the
# iterations. To this end, we defined a custom callback function
# (currently, only Nk = 1 kpt only is supported)
#
#            !!! NOT OPTIMIZED YET, WE USE PLAIN DENSITY MATRICES !!!
#

# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
#  atoms = [Si => [ones(3)/8 .- 1e-3, -ones(3)/8]]
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# define different models
n_el = 2
modelLDA = model_LDA(lattice, atoms, n_electrons=n_el)
modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()], n_electrons=n_el)
modelAtomic = model_atomic(lattice, atoms, n_electrons=n_el)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-12
tol_krylov = 1e-15
Ecut = 15           # kinetic energy cutoff in Hartree

## number of kpoints
Nk = 0
N = 0
filled_occ = 0

model_list = ["Atomic", "Hartree", "LDA"]
k = 0
ite = 0

## custom callback to follow estimators
function callback_estimators()

    φ_list = []                 # list of ψ^k
    res_φ_list = []             # list of residuals as Hψ - λψ
    global ite
    ite = 0

    function callback(info)
        global N, Nk, filled_occ, k, ite

        if info.stage == :finalize
            ## converged values
            φref = similar(info.ψ)
            for ik = 1:Nk
                φref[ik] = info.ψ[ik][:,1:N]
            end
            basis = info.basis
            ρ = info.ρ
            H = info.ham
            egval = info.eigenvalues
            T = typeof(ρ.real[1])
            occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

            ## orthogonalisation routine
            ortho(ψk) = Matrix(qr(ψk).Q)

            ## vec and unpack
            # length of ψ0[ik]
            lengths = [length(φref[ik]) for ik = 1:Nk]
            starts = copy(lengths)
            starts[1] = 1
            for ik = 1:Nk-1
                starts[ik+1] = starts[ik] + lengths[ik]
            end
            pack(ψ) = vcat(Base.vec.(ψ)...)
            unpack(ψ) = [@views reshape(ψ[starts[ik]:starts[ik]+lengths[ik]-1], size(φref[ik]))
                         for ik = 1:Nk]

            # random starting point for eigensolvers
            ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
                  for kpt in basis.kpoints]
            packed_proj!(ϕ,ψ) = proj!(unpack(ϕ), unpack(ψ))

            # eigsolve
            function f(x)
                δφ = unpack(x)
                δφ = proj!(δφ, φref)
                ΩpKx = ΩplusK(basis, δφ, φref, ρ, H, egval, occupation)
                ΩpKx = proj!(ΩpKx, φref)
                pack(ΩpKx)
            end
            vals_ΩpK, _ = eigsolve(f, pack(ψ0), 3, :SR;
                                   tol=tol_krylov, verbosity=0, eager=true,
                                   orth=OrthogonalizeAndProject(packed_proj!, pack(φref)))

            # svd solve
            function g(x,flag)
                f(x)
            end
            svds_ΩpK, _ = svdsolve(g, pack(ψ0), 3, :SR;
                                   tol=tol_krylov, verbosity=0, eager=true,
                                   orth=OrthogonalizeAndProject(packed_proj!, pack(φref)))

            normop = 1. / svds_ΩpK[1]
            println("--> plus petite valeur propre $(real(vals_ΩpK[1]))")
            println("--> plus petite valeur singulière $(svds_ΩpK[1])")
            println("--> normop $(normop)")
            println("--> gap $(egval[1][N+1] - egval[1][N])")

            # evaluating errors
            norm_res_φ_list = norm.(res_φ_list)
            err_ref_list = [ abs(sqrt(Complex(2*N - 2*norm(φ_list[i][1]'φref[1])^2))) for i in 1:ite]
            err_φ_estimator = normop .* norm_res_φ_list

            figure()
            title("error estimators vs iteration for $(model_list[k]), N = $(N)")
            semilogy(1:ite, err_ref_list, "x-", label="|P-Pref|")
            #  semilogy(1:ite, norm_res_φ_list, "x-", label="|res_φ|")
            #  semilogy(1:ite, err_φ_estimator, "x-", label="|(Ω+K)^{-1}|*|res_φ|")

            # plotting newton iterations
            err_newton = []
            err_newton_step_φ = []
            for i in 1:ite
                res_φ = proj!(res_φ_list[i], φref)
                δφ_packed, info = linsolve(f, pack(res_φ);
                                           tol=1e-15, verbosity=1,
                                           orth=OrthogonalizeAndProject(packed_proj!, pack(φref)))
                δφ = unpack(δφ_packed)
                δφ = proj!(δφ, φref)


                φ_newton = deepcopy(φ_list[i])
                for ik = 1:Nk
                    for j = 1:N
                        φ_newton[ik][:,j] = φ_newton[ik][:,j] - δφ[ik][:,j]
                    end
                    φ_newton[ik] = ortho(φ_newton[ik])
                end

                append!(err_newton, abs(sqrt(Complex(2*N - 2*norm(φ_newton[1]'φref[1])^2))))
                append!(err_newton_step_φ, norm(δφ))
            end
            semilogy(1:ite, err_newton, "x-", label="|P_newton-Pref|")
            semilogy(1:ite, err_newton_step_φ, "x-", label="|δφ|")
            legend()
            xlabel("iterations")

        else
            ite += 1
            φ = info.ψ
            occ = info.occupation
            basis = info.basis
            ρ = compute_density(basis, φ, occ)
            _, ham = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])
            # compute residual
            res_φ = similar(φ)
            for ik = 1:Nk
                φk = φ[ik]
                Hk = ham.blocks[ik]
                egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:length(occ[ik])]
                rk = Hk*φk - hcat([egvalk[i] * φk[:,i] for i = 1:length(occ[ik])]...)
                res_φ[ik] = rk[:,1:N]
            end
            append!(res_φ_list, [res_φ])
            φφ = similar(φ)
            for ik = 1:Nk
                φφ[ik] = φ[ik][:,1:N]
            end
            append!(φ_list, [φφ])
        end
    end
    callback
end


for model in [modelAtomic, modelHartree, modelLDA]
    println("--------------------------------")
    println("--------------------------------")
    global k
    k += 1
    println(model_list[k])

    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    global N, Nk, filled_occ
    ## number of kpoints
    Nk = length(basis.kpoints)
    ## number of eigenvalue/eigenvectors we are looking for
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    scfres = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol),
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-14),
                                   callback=callback_estimators())
end
