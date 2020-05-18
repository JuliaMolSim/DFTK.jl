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
    Ecut_list = 5:5:60

    Ecut_ref = 100
    println("--------------------------------------")
    println("Ecut ref = $(Ecut_ref)")
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)
    scfres = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
    Fref = forces(scfres)

    for Ecut in Ecut_list
        println("--------------------------------------")
        println("Ecut = $(Ecut)")
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)
        scfres = self_consistent_field(basis, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol))

        # compute forces
        F = forces(scfres)
        println(F)
        push!(forces_list, norm(F - Fref))
    end

    semilogy(Ecut_list, forces_list)
    legend()
    savefig("forces.pdf")
end

test_forces_convergence()
