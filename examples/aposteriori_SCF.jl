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
Ecut_ref = 30           # kinetic energy cutoff in Hartree

Ecut_list = 5:5:25
model_list = ["Atomic", "Hartree", "LDA"]
k = 0


for model in [modelAtomic, modelHartree, modelLDA]
    println("--------------------------------")
    println("--------------------------------")
    global k
    k += 1
    println(model_list[k])
    ## reference density matrix
    println("--------------------------------")
    println("reference computation")
    basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol),
                                       callback=info->nothing)
    T = typeof(scfres_ref.ρ.real[1])
    ## number of kpoints
    Nk = length(basis_ref.kpoints)
    ## number of eigenvalue/eigenvectors we are looking for
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    ## reference matrix density
    Pref = scfres_ref.ψ[1][:,1:N]*scfres_ref.ψ[1][:,1:N]'

    ## error lists
    err_list = []
    res_list = []
    opres_list = []

    for Ecut in Ecut_list

        println("--------------------------------")
        println("Ecut = $(Ecut)")

        # compute solution
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        scfres = self_consistent_field(basis, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol),
                                       callback=info->nothing)
        ## converged values
        φ = similar(scfres.ψ)
        for ik = 1:Nk
            φ[ik] = scfres.ψ[ik][:,1:N]
        end
        ρ = scfres.ρ
        H = scfres.ham
        egval = scfres.eigenvalues
        occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

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
        println("--> gap $(scfres.eigenvalues[1][5] - scfres.eigenvalues[1][4])")

        # compute errors after interpolating to the reference basis (_ref is here
        # to remember which basis quantities live on, except for Pref which is
        # the reference density matrix)
        φ_ref = DFTK.interpolate_blochwave(φ, basis, basis_ref)
        P = φ_ref[1][:,1:N]*φ_ref[1][:,1:N]'
        ρ_ref = compute_density(basis_ref, φ_ref, occupation)
        _, H_ref = energy_hamiltonian(basis_ref, φ_ref, occupation; ρ=ρ_ref[1])
        H_ref = Matrix(H_ref.blocks[1])
        res = P*H_ref + H_ref*P - 2*P*H_ref*P
        append!(err_list, norm(P - Pref))
        append!(res_list, norm(res))
        append!(opres_list, normop*norm(res))
    end

    figure()
    title("error estimators vs Ecut for $(model_list[k])")
    semilogy(Ecut_list, err_list, "x-", label="|P-Pref|")
    semilogy(Ecut_list, res_list, "x-", label="|res|")
    semilogy(Ecut_list, opres_list, "x-", label="|(Ω+K)^{-1}|*|res|")
    legend()
end
