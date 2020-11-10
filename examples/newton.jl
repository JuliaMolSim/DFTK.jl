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

#  model = model_atomic(lattice, atoms, n_electrons=2)
#  model = model_atomic(lattice, atoms, n_electrons=2, extra_terms=[Hartree()])
model = model_LDA(lattice, atoms, n_electrons=2)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 5           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

tol = 1e-12
scfres = self_consistent_field(basis, tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol))

# newton algorithm
function newton()

    err_list = []
    err_ref_list = []
    k_list = []

    err = 1
    k = 0
    max_iter = 100

    kpt = basis.kpoints[1]
    Hk = scfres.ham.blocks[1]
    occ = scfres.occupation
    ψk = scfres.ψ[1][:,1]

    φ = deepcopy(scfres.ψ)
    φk = ψk + rand(typeof(ψk[1,1]), size(ψk))*1e-2
    φk /= norm(φk)
    φ[1][:,1] = φk
    φ[1][:,2:end] .= 0

    δφ = deepcopy(φ)
    δφ[1][:,2:end] .= 0


    while err > tol && k < max_iter
        k += 1
        println("Iteration $(k)")

        # compute residual
        φ[1][:,1] = φk
        ρ = compute_density(basis, φ, occ)
        E, ham = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])
        hamk = ham.blocks[1]
        egvalk = [φk'*(hamk*φk)]
        rk = hamk*φk - egvalk[1]*φk
        rk = proj!(rk, φk)

        # solve linear system
        function f(x)
            x = proj!(x, φk)

            δφ[1][:,1] = x

            δρ = DFTK.compute_density(basis, φ, δφ, occ)
            Kδρ = apply_kernel(basis, δρ[1]; ρ=ρ[1])

            ΩpKx = ΩplusK_kpt(scfres, kpt, x, Kδρ[1], φk, hamk, egvalk)
            ΩpKx = proj!(ΩpKx, φk)
            vec(ΩpKx)
        end

        δφk, info = linsolve(f, rk;
                             tol=1e-12, verbosity=1,
                             orth=OrthogonalizeAndProject(proj!, φk))

        φk_prec = deepcopy(φk)
        φk = (φk - δφk) / norm(φk - δφk)

        err = norm(φk - φk_prec)
        err_ref = norm(φk - (ψk'φk)*ψk)
        println(" ---> err     = $(err)")
        println(" ---> err_ref = $(err_ref)")
        append!(err_list, err)
        append!(err_ref_list, err_ref)
        append!(k_list, k)
    end
    figure()
    semilogy(k_list, err_list, "x-", label="iter")
    semilogy(k_list, err_ref_list, "x-", label="ref")
end


newton()
