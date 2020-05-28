using DFTK
using PyPlot
using LinearAlgebra

function test_forces_convergence()
    # Calculation parameters
    kgrid = [4, 4, 4]       # k-Point grid
    supercell = [1, 1, 1]   # Lattice supercell
    n_bands = 8             # number of bands to plot in the bandstructure
    tol = 5e-15

    # Setup silicon lattice
    a = 10.263141334305942  # Silicon lattice constant in Bohr
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [.008 .+ ones(3)/8, -ones(3)/8]]

    # Make a supercell if desired
    pystruct = pymatgen_structure(lattice, atoms)
    pystruct.make_supercell(supercell)
    lattice = load_lattice(pystruct)
    atoms = [Si => [s.frac_coords for s in pystruct.sites]]

    model = model_LDA(lattice, atoms)

    forces_list = []
    Ecut_list = 5:5:15

    Ecut_ref = 20
    println("--------------------------------------")
    println("Ecut ref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid, enable_bzmesh_symmetry=false)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
    Fref = Dict()
    for term in basis_ref.terms
        key = typeof(term)
        Fref[key] = forces(term, scfres_ref.ψ, scfres_ref.occupation, ρ=scfres_ref.ρ)
    end

    F = Dict()
    for term in basis_ref.terms
        key = typeof(term)
        F[key] = []
    end

    for Ecut in Ecut_list
        println("--------------------------------------")
        println("Ecut = $(Ecut)")
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)
        scfres = self_consistent_field(basis, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol))
        for term in basis.terms
            key = typeof(term)
            ft = forces(term, scfres.ψ, scfres.occupation, ρ=scfres.ρ)
            push!(F[key], ft)
        end
    end

    PyPlot.figure(figsize=(10,10))

    for term in basis_ref.terms
        key = typeof(term)
        if Fref[key] != nothing
            error_list = norm.([ff - Fref[key] for ff in F[key]])
            semilogy(Ecut_list, error_list, "+-", label="$(typeof(term))")
        end
    end
    total_forces_ref = sum([Fref[typeof(term)] for term in basis_ref.terms
                            if Fref[typeof(term)] != nothing])
    total_forces = [sum(F[typeof(term)][i] for term in basis_ref.terms
                        if F[typeof(term)][i] != nothing)
                    for i in 1:length(Ecut_list)]
    error_list = norm.([ff - total_forces_ref for ff in total_forces])
    PyPlot.semilogy(Ecut_list, error_list, "x-", label="total forces")
    legend()
    PyPlot.xlabel("Ecut")
    PyPlot.ylabel("error")
    PyPlot.title("Decomposed forces")
    PyPlot.savefig("Ecutref_$(Ecut_ref)_forces.pdf")

    display(F[DFTK.TermEwald])
end


test_forces_convergence()
