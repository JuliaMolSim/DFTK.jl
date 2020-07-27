using LinearAlgebra
using DFTK
using KrylovKit
using Printf

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 10

# Silicon lattice
a = 10.26
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# change the symmetry to compute the dielectric operator with and without symmetries
modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()], symmetry=:off)
modelLDA = model_LDA(lattice, atoms; symmetry=:off)

basis = PlaneWaveBasis(modelHartree, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-12, callback=info->nothing)

## callback function
function callback_density()
    ρin_list = []
    Vsρin_list = []
    ρout_list = []
    Vsρout_list = []
    ρnext_list = []
    Vsρnext_list = []
    ρinout_list = []
    Vsρinout_list = []
    ρinnext_list = []
    Vsρinnext_list = []
    function callback(info)
        if info.stage == :finalize
            ρ = info.ρ.fourier
            Vsρ = DFTK.apply_kernel_sqrt(basis.terms[6], info.ρ).fourier

            subplot(121)
            title("L2 norm")
            ρin_norm = norm.([ρin_list[i] - ρ for i in 1:length(ρin_list)])
            ρout_norm = norm.([ρout_list[i] - ρ for i in 1:length(ρout_list)])
            ρnext_norm = norm.([ρnext_list[i] - ρ for i in 1:length(ρnext_list)])
            ρinout_norm = norm.([ρinout_list[i] for i in 1:length(ρinout_list)])
            ρinnext_norm = norm.([ρinnext_list[i] for i in 1:length(ρinnext_list)])
            semilogy(ρin_norm, "x-", label="|ρin - ρ|")
            #  semilogy(ρout_norm, "x-", label="|ρout - ρ|")
            #  semilogy(ρnext_norm, "x-", label="|ρnext - ρ|")
            semilogy(ρinout_norm, "x-", label="|ρin - ρout|")
            #  semilogy(ρinnext_norm, "x-", label="|ρin - ρnext|")
            xlabel("iteration")
            ylabel("error")
            legend()

            subplot(122)
            title("Vc norm")
            Vsρin_norm = norm.([Vsρin_list[i] - Vsρ for i in 1:length(Vsρin_list)])
            Vsρout_norm = norm.([Vsρout_list[i] - Vsρ for i in 1:length(Vsρout_list)])
            Vsρnext_norm = norm.([Vsρnext_list[i] - Vsρ for i in 1:length(Vsρnext_list)])
            Vsρinout_norm = norm.([Vsρinout_list[i] for i in 1:length(ρinout_list)])
            Vsρinnext_norm = norm.([Vsρinnext_list[i] for i in 1:length(ρinnext_list)])
            semilogy(Vsρin_norm, "x-", label="|ρin - ρ|_Vc")
            #  semilogy(Vsρout_norm, "x-", label="|ρout - ρ|_Vc")
            #  semilogy(Vsρnext_norm, "x-", label="|ρnext - ρ|_Vc")
            semilogy(Vsρinout_norm, "x-", label="|ρin - ρout|_Vc")
            #  semilogy(Vsρinnext_norm, "x-", label="|ρin - ρnext|_Vc")
            xlabel("iteration")
            ylabel("error")
            legend()

        else
            ρin = info.ρin
            ρout = info.ρout
            ρnext = info.ρnext

            push!(ρin_list, ρin.fourier)
            push!(Vsρin_list, DFTK.apply_kernel_sqrt(basis.terms[6], ρin).fourier)
            push!(ρout_list, ρout.fourier)
            push!(Vsρout_list, DFTK.apply_kernel_sqrt(basis.terms[6], ρout).fourier)
            push!(ρnext_list, ρnext.fourier)
            push!(Vsρnext_list, DFTK.apply_kernel_sqrt(basis.terms[6], ρnext).fourier)
            push!(ρinout_list, ρin.fourier - ρout.fourier)
            push!(Vsρinout_list, DFTK.apply_kernel_sqrt(basis.terms[6], ρin - ρout).fourier)
            push!(ρinnext_list, ρin.fourier - ρnext.fourier)
            push!(Vsρinnext_list, DFTK.apply_kernel_sqrt(basis.terms[6], ρin - ρnext).fourier)
        end
    end
    callback
end


## eigenvalue problem with KrylovKit

# apply ϵ = 1 - χ0 K
function ϵ(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Kdρ = apply_kernel(basis, from_real(basis, dρ); ρ=scfres.ρ)
    χ0Kdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, Kdρ)
    vec(dρ - χ0Kdρ.real)
end

# apply ϵ'
function ϵt(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    χ0dρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues,
                    from_real(basis, dρ))
    Kχ0dρ = apply_kernel(basis, χ0dρ; ρ=scfres.ρ)
    vec(dρ - Kχ0dρ.real)
end

# apply Vc^(1/2) ϵ Vc(-1/2)
function VsϵVs(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Vsdρ = DFTK.apply_kernel_invsqrt(basis.terms[6], from_real(basis, dρ))
    KVsdρ = apply_kernel(basis, Vsdρ; ρ=scfres.ρ)
    χ0KVsdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, KVsdρ)
    Vsχ0KVsdρ = DFTK.apply_kernel_sqrt(basis.terms[6], χ0KVsdρ)
    vec(dρ - Vsχ0KVsdρ.real)
end

# apply Vc^(-1/2) ϵ' Vc(1/2)
function VsϵVst(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Vsdρ = DFTK.apply_kernel_sqrt(basis.terms[6], from_real(basis, dρ))
    χ0Vsdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, Vsdρ)
    Kχ0Vsdρ = apply_kernel(basis, χ0Vsdρ; ρ=scfres.ρ)
    VsKχ0Vsdρ = DFTK.apply_kernel_invsqrt(basis.terms[6], Kχ0Vsdρ)
    vec(dρ - VsKχ0Vsdρ.real)
end

start = vec(randn(size(scfres.ρ.real)))
norm_L2 = []
norm_Vc = []

for model in [modelHartree, modelLDA]

    global basis, scfres

    figure()
    suptitle("Silicon " * string(typeof(model.term_types[end])))

    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=tol, solver=scf_nlsolve_solver(m=10),
                                   callback=callback_density(),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

    #  # flagged function for SVD in L2 norm
    #  function G(x, flag)
    #      if flag
    #          y = ϵt(x)
    #      else
    #          y = ϵ(x)
    #      end
    #      return y
    #  end
    #  vals_L2, _ = svdsolve(G, start, 1, :SR;  tol=1e-4, verbosity=verb, eager=true)

    #  # flagged function for SVD in Vc norm
    #  function F(x, flag)
    #      if flag
    #          y = VsϵVst(x)
    #      else
    #          y = VsϵVs(x)
    #      end
    #      return y
    #  end
    #  vals_Vc, _ = svdsolve(F, start, 1, :SR;  tol=1e-4, verbosity=verb, eager=true)

    #  global norm_L2, norm_Vc
    #  push!(norm_L2, 1 / vals_L2[1])
    #  push!(norm_Vc, 1 / vals_Vc[1])

    #  println("\nsingval L2 = ", vals_L2[1])
    #  println("singval Vc = ", vals_Vc[1], "\n")
end

#  println("\n        |   L2  |   Vc  |")
#  println("-------------------------")
#  @printf "Hartree | %.3f | %.3f |\n" norm_L2[1] norm_Vc[1]
#  @printf "LDA     | %.3f | %.3f |\n" norm_L2[2] norm_Vc[2]
#  println("-------------------------\n")
