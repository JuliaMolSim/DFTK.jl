using LinearAlgebra
using DFTK

# Calculation parameters
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 5           # kinetic energy cutoff in Hartree
verb = 2

# LiCu lattice
lattice = Matrix(Diagonal([9.124450004207551, 9.891041973775085, 8.270832367071698]))

Li = ElementPsp(:Li, psp=load_psp("hgh/lda/li-q3.hgh"))
Cu = ElementPsp(:Cu, psp=load_psp("hgh/lda/cu-q11.hgh"))
atoms = [Li => [[0.015272, 0.0, 0.0], [0.332794, 0.0, 0.5],
                [0.811602, 0.264698, 0.5], [0.811602, 0.735301, 0.5]],
         Cu => [[0.518469, 0.248638, 0.0], [0.017306, 0.499999, 0.0],
                [0.518469, 0.751363, 0.0], [0.307825, 0.499999, 0.5]]]

modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()],
                            symmetry=:off, temperature=0.001)
modelLDA = model_LDA(lattice, atoms; symmetry=:off, temperature=0.001)

basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-12, solver=scf_nlsolve_solver(m=10),
                              callback=info->nothing)



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
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=1e-12, solver=scf_nlsolve_solver(m=10))

    # flagged function for SVD in L2 norm
    function G(x, flag)
        if flag
            y = ϵt(x)
        else
            y = ϵ(x)
        end
        return y
    end
    vals_L2, _ = svdsolve(G, start, 1, :SR;  tol=1e-4, verbosity=verb, eager=true)

    # flagged function for SVD in Vc norm
    function F(x, flag)
        if flag
            y = VsϵVst(x)
        else
            y = VsϵVs(x)
        end
        return y
    end
    vals_Vc, _ = svdsolve(F, start, 1, :SR;  tol=1e-4, verbosity=verb, eager=true)

    global norm_L2, norm_Vc
    push!(norm_L2, 1 / vals_L2[1])
    push!(norm_Vc, 1 / vals_Vc[1])

    println("\nsingval L2 = ", vals_L2[1])
    println("singval Vc = ", vals_Vc[1], "\n")
end

println("\n        |   L2  |   Vc  |")
println("-------------------------")
@printf "Hartree | %.3f | %.3f |\n" norm_L2[1] norm_Vc[1]
@printf "LDA     | %.3f | %.3f |\n" norm_L2[2] norm_Vc[2]
println("-------------------------\n")
