using DFTK
using StaticArrays
using HDF5
using PyPlot
using LinearAlgebra
using FFTW

BLAS.set_num_threads(16)
FFTW.set_num_threads(16)

function test_forces_convergence()
    # Calculation parameters
    kgrid = [1, 1, 1]       # k-Point grid
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

    #  model = model_LDA(lattice, atoms)
    model = model_DFT(lattice, atoms, [])

    forces_list = []
    Ecut_list = 5:1:6

    Ecut_ref = 8
    println("--------------------------------------")
    println("Ecut ref = $(Ecut_ref)")
    basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid, enable_bzmesh_symmetry=false)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
    Fref = Dict()
    key_list = []
    for term in basis_ref.terms
        key = string(typeof(term))
        ftref = forces(term, scfres_ref.ψ, scfres_ref.occupation, ρ=scfres_ref.ρ)
        if ftref !== nothing
            Fref[key] = Array.(ftref[1])
            push!(key_list, key)
        end
    end

    F = Dict()
    for term in basis_ref.terms
        key = string(typeof(term))
        F[key] = []
    end

    for Ecut in Ecut_list
        println("--------------------------------------")
        println("Ecut = $(Ecut)")
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)
        scfres = self_consistent_field(basis, tol=tol,
                                       is_converged=DFTK.ScfConvergenceDensity(tol))
        for term in basis.terms
            key = string(typeof(term))
            if key in key_list
                ft = forces(term, scfres.ψ, scfres.occupation, ρ=scfres.ρ)
                push!(F[key], Array.(ft[1]))
            end
        end
    end

    key_list = String.(key_list)
    h5open("forces_rHF.h5", "w") do file
        Ecut_list = collect(Ecut_list)
        @write file Ecut_list
        @write file key_list
        for key in key_list
            file["Fref/$(key)"] = hcat(Fref[key]...)
            Fk = reshape(hcat(hcat(F[key]...)...), 3, 2, length(Ecut_list))
            file["F/$(key)"] = Float64.(Fk)
        end
    end
end

test_forces_convergence()
