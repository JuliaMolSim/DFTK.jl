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

model = model_LDA(lattice, atoms, n_electrons=8)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)

function test_newton(Ecut_list, Ecut_ref)

    println("\n------------------------------------")
    println("Solving for Ecutref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
    scfres_ref = self_consistent_field(basis_ref, tol=1e-10,
                                       callback=info->nothing)

    nrj_ref = scfres_ref.energies.total
    ψ1_ref = scfres_ref.ψ[1][:,1]

    println("Energy : $(nrj_ref)")

    nrj_list = []
    ψ1_list = []

    nrj_corr_list = []
    ψ1_corr_list = []

    for Ecut in Ecut_list
        println("\n------------------------------------")
        println("Solving for Ecut = $(Ecut)")
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        scfres = self_consistent_field(basis, tol=1e-10,
                                      callback=info->nothing)

        H = scfres.ham
        ψ = scfres.ψ
        egval = scfres.eigenvalues
        occ = scfres.occupation
        nrj = scfres.energies.total

        ψ_itp = DFTK.interpolate_blochwave(ψ, basis, basis_ref)
        ψ1 = ψ_itp[1][:,1]

        println("Energy : $(nrj)")
        append!(nrj_list, abs(nrj - nrj_ref))
        append!(ψ1_list, norm(ψ1 - (ψ1_ref'ψ1) * ψ1_ref))

        println("Computing the residual...")
        Hψ = H * ψ
        R = similar(ψ)
        for ik in 1:length(basis.kpoints)
            N = length([l for l in occ[ik] if l != 0.0])
            ψk = ψ[ik]
            R[ik] = similar(ψk[:,1:N])
            for i in 1:N
                R[ik][:,i] = Hψ[ik][:,i] - ψk[:,i] * egval[ik][i]
            end
            R[ik] = proj!(R[ik], ψk[:,1:N])
        end

        x0 = similar(ψ)
        for ik = 1:length(basis.kpoints)
            N = size(R[ik], 2)
            x0[ik] = generate_δφ(ψ[ik][:,1:N])
        end

        # we want to solve (Ω+K)δψ = R, and then do ψ = ψ + δψ
        for ik = 1:length(basis.kpoints)

            kpt = basis.kpoints[ik]
            egvalk = egval[ik]
            Rk = R[ik]
            Hk = H.blocks[ik]

            N = size(Rk, 2)
            ψk = ψ[ik][:,1:N]

            b = vec(Rk)

            function f(x)
                x = reshape(x, size(ψk))
                x = proj!(x, ψk)
                x0[ik] = x

                δρ = compute_density(basis, x0, [o[1:N] for o in occ])
                Kδρ = apply_kernel(basis, δρ[1]; ρ=scfres.ρ)

                ΩpKx = ΩplusK_kpt(scfres, kpt, x, Kδρ[ik], ψk, Hk, egvalk)
                ΩpKx = proj!(ΩpKx, ψk)
                vec(ΩpKx)
            end

            println("Solve linear system and apply correction...")
            δφk, info = linsolve(f, b, vec(x0[ik]);
                                 tol=1e-4, verbosity=1,
                                 orth=OrthogonalizeAndProject(proj!, ψk))
            δφk = reshape(δφk, size(ψk))
            ψk .-= δφk

            for i=1:N
                ψ[ik][:,i] = ψk[:,i] / norm(ψk[:,i])
            end
        end

        ψ_itp = DFTK.interpolate_blochwave(ψ, basis, basis_ref)
        ψ1_corr = ψ_itp[1][:,1]
        ρ = compute_density(basis, ψ, occ)
        nrj_corr, _ = energy_hamiltonian(basis, ψ, occ; ρ=ρ[1])
        nrj_corr = nrj_corr.total

        println("Energy corrected : $(nrj_corr)")
        append!(nrj_corr_list, abs(nrj_corr - nrj_ref))
        append!(ψ1_corr_list, norm(ψ1_corr - (ψ1_ref'ψ1_corr) * ψ1_ref))

    end

    figure()
    title("Error on the energy")
    semilogy(Ecut_list, nrj_list, "+-", label="Solution for Ecut")
    semilogy(Ecut_list, nrj_corr_list, "+-", label="Correction for Ecut")
    legend()

    figure()
    title("Error on the 1st eigenvector")
    semilogy(Ecut_list, ψ1_list, "+-", label="Solution for Ecut")
    semilogy(Ecut_list, ψ1_corr_list, "+-", label="Correction for Ecut")
    legend()

end

Ecut_ref = 20
Ecut_list = 5:5:15

test_newton(Ecut_list, Ecut_ref)
