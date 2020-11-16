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
atoms = [Si => [ones(3)/8, -ones(3)/8]]

modelLDA = model_LDA(lattice, atoms)
modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()])
modelAtomic = model_atomic(lattice, atoms)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-12
tol_krylov = 1e-15
Ecut = 10           # kinetic energy cutoff in Hartree

## number of kpoints
Nk = 0
N = 0
filled_occ = 0

model_list = ["Atomic", "Hartree", "LDA"]
k = 0
ite = 0

## custom callback to follow estimators
function callback_estimators()

    P_list = []                 # list of P^k
    res_list = []               # list of residuals
    global ite
    ite = 0

    function callback(info)
        global N, Nk, filled_occ, k, ite

        if info.stage == :finalize
            ## converged values
            φ = similar(info.ψ)
            for ik = 1:Nk
                φ[ik] = info.ψ[ik][:,1:N]
            end
            basis = info.basis
            ρ = info.ρ
            H = info.ham
            egval = info.eigenvalues
            T = typeof(ρ.real[1])
            occupation = [filled_occ * ones(T, N) for ik = 1:Nk]
            Pref = φ[1][:,1:N]*φ[1][:,1:N]'

            ## orthogonalisation routine
            ortho(ψk) = Matrix(qr(ψk).Q)

            ## vec and unpack
            # length of ψ0[ik]
            lengths = [length(φ[ik]) for ik = 1:Nk]
            starts = copy(lengths)
            starts[1] = 1
            for ik = 1:Nk-1
                starts[ik+1] = starts[ik] + lengths[ik]
            end
            pack(ψ) = vcat(Base.vec.(ψ)...) # TODO as an optimization, do that lazily? See LazyArrays
            unpack(ψ) = [@views reshape(ψ[starts[ik]:starts[ik]+lengths[ik]-1], size(φ[ik]))
                         for ik = 1:Nk]

            # random starting point
            ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
                  for kpt in basis.kpoints]
            packed_proj!(ϕ,ψ) = proj!(unpack(ϕ), unpack(ψ))

            # eigsolve
            function f(x)
                δφ = unpack(x)
                δφ = proj!(δφ, φ)
                ΩpKx = ΩplusK(basis, δφ, φ, ρ, H, egval, occupation)
                ΩpKx = proj!(ΩpKx, φ)
                pack(ΩpKx)
            end
            vals_ΩpK, _ = eigsolve(f, pack(ψ0), 3, :SR;
                                   tol=tol_krylov, verbosity=0, eager=true,
                                   orth=OrthogonalizeAndProject(packed_proj!, pack(φ)))

            # svd solve
            function g(x,flag)
                f(x)
            end
            svds_ΩpK, _ = svdsolve(g, pack(ψ0), 3, :SR;
                                   tol=tol_krylov, verbosity=0, eager=true,
                                   orth=OrthogonalizeAndProject(packed_proj!, pack(φ)))

            normop = 1. / svds_ΩpK[1]
            println("--> plus petite valeur propre $(real(vals_ΩpK[1]))")
            println("--> plus petite valeur singulière $(svds_ΩpK[1])")
            println("--> norm_L2 $(normop)")
            println("--> gap $(egval[1][5] - egval[1][4])")

            err_P_list = [norm(P_list[i+1] - P_list[i]) for i = 1:(length(P_list)-1)]
            err_ref_list = [norm(P_list[i]-Pref) for i in 1:length(P_list)]
            err_estimator = normop .* res_list

            figure()
            title("error estimators vs iteration for $(model_list[k])")
            semilogy(1:ite, err_ref_list, "x-", label="|P-Pref|")
            semilogy(1:ite, err_estimator, "x-", label="|(Ω+K)^{-1}|*|res|")
            legend()
            xlabel("iterations")

        else
            ite += 1
            φ = info.ψ
            ham = info.ham
            # compute density matrix
            P = φ[1][:,1:N]*φ[1][:,1:N]'
            append!(P_list, [P])
            # compute residual
            H = Matrix(ham.blocks[1])
            res = P*H + H*P - 2*P*H*P
            append!(res_list, norm(res))
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
    Nk = length(basis.kpoints)
    ## number of eigenvalue/eigenvectors we are looking for
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    scfres = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol),
                                   callback=callback_estimators())
end
