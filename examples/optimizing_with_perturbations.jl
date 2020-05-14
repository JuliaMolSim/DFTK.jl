using DFTK
using PyPlot
using DataFrames
using GLM
import Statistics: mean
using Optim

include("perturbations.jl")

aref = 10.263141334305942
Eref = -7.924456699632 # computed with silicon.jl for kgrid = [4,4,4], Ecut = 100
tol = 1e-6

"""
compute scfres for a given lattice constant
"""
function E(a, Ecut)
    ### Setting the model
    # Calculation parameters
    kgrid = [4, 4, 4]       # k-Point grid
    supercell = [1, 1, 1]   # Lattice supercell

    # Setup silicon lattice
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [0.008 .+ ones(3)/8, -ones(3)/8]]

    # Make a supercell if desired
    pystruct = pymatgen_structure(lattice, atoms)
    pystruct.make_supercell(supercell)
    lattice = load_lattice(pystruct)
    atoms = [Si => [s.frac_coords for s in pystruct.sites]]

    # precize the number of electrons on build the model
    Ne = 8
    model = model_LDA(lattice, atoms; n_electrons=Ne)
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)
    scfres = self_consistent_field(basis, tol=tol/10,
                                   callback=info->nothing)
    sum(values(scfres.energies))
end


"""
optimize the lattice constant for a fixed Ecut
"""
function optimize_a(Ecut)

    function f(x)
        E(x, Ecut)
    end

    res = optimize(f, 0.98*aref, 1.02*aref, GoldenSection())
                   # Optim.Options(show_trace = true, f_tol=tol))
    #  amin = Optim.minimizer(res)
end

function optimize_a()
    a_list = []
    Ecut_list = 10:5:40
    for Ecut in Ecut_list
        println("Ecut=$(Ecut)")
        res = optimize_a(Ecut)
        amin = Optim.minimizer(res)
        println(amin)
        push!(a_list, amin)
    end
    plot(Ecut_list, amin, "-+")
end

"""
graphical optimization of the lattice constant
"""
function optimize_a_graphic(Ecut)
    arange = range(0.95*aref, 1.05*aref, length=21)

    E_list = []
    for a in arange
        println("Solving for a=$(a)")
        push!(E_list, E(a, Ecut))
    end
    plot(arange, E_list, "+-", label="Energy with Ecut=$(Ecut)")

    (Emin, idmin) = findmin(E_list)
    amin = arange[idmin]
end

#  for Ecut in 10:5:30
#      println("--------------------------------")
#      println("Ecut=$(Ecut)")
#      amin = optimize_a_graphic(Ecut)
#      println(amin)
#  end
#  plot(aref, Eref)
#  legend()






