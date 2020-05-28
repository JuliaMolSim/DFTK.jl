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
    atoms = [Si => [ones(3)/8, -ones(3)/8]]

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
compute scfres with perturbation coef α
"""
function E_perturbed(a, Ecut, α)
    ### Setting the model
    # Calculation parameters
    kgrid = [4, 4, 4]       # k-Point grid
    supercell = [1, 1, 1]   # Lattice supercell

    # Setup silicon lattice
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]

    # Make a supercell if desired
    pystruct = pymatgen_structure(lattice, atoms)
    pystruct.make_supercell(supercell)
    lattice = load_lattice(pystruct)
    atoms = [Si => [s.frac_coords for s in pystruct.sites]]

    # precize the number of electrons on build the model
    Ne = 8
    model = model_LDA(lattice, atoms; n_electrons=Ne)
    kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.lattice, model.atoms)
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
    scfres = self_consistent_field(basis, tol=tol/10,
                                   callback=info->nothing)
    avg = true
    Ep_fine, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, false)
    sum(values(Ep_fine))
end


"""
optimize the lattice constant for a fixed Ecut
"""
function optimize_a(Ecut)

    function f(x)
        E(x, Ecut)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref, GoldenSection()))
    #  amin = Optim.minimizer(res)
end

"""
optimize the lattice constant for a fixed Ecut with perturbation
"""
function optimize_a_perturbed(Ecut, α)

    function f(x)
        E_perturbed(x, Ecut, α)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref, GoldenSection()))
    #  amin = Optim.minimizer(res)
end

"""
compare optimization time for non-perturbed and perturbed scf
"""
function compare()

    res_ref = optimize_a(30)

    Ecut_list = 5:5:25
    time_list = []
    alloc_list = []
    err_list = []

    Ecutp_list = 5:5:10
    timep_list = []
    allocp_list = []
    errp_list = []

    for Ecut in Ecut_list

        println("----------------------")
        println("Ecut = $(Ecut)")

        res = optimize_a(Ecut)
        push!(err_list, abs(res[1] - aref))
        push!(time_list, res[2])
        push!(alloc_list, res[3])
    end

    for Ecutp in Ecutp_list

        println("----------------------")
        println("Ecutp = $(Ecutp)")

        resp = optimize_a_perturbed(Ecutp, 2.5)
        push!(errp_list, abs(resp[1] - aref))
        push!(timep_list, resp[2])
        push!(allocp_list, resp[3])
    end

    figure(figsize=(20,10))
    suptitle("Performance")

    subplot(121)
    PyPlot.semilogy(time_list, err_list, "+-", label="Non-perturbed")
    PyPlot.semilogy(timep_list, errp_list, "x-", label="Perturbed")
    xlabel("time")
    ylabel("error")
    legend()

    subplot(122)
    PyPlot.semilogy(alloc_list, err_list, "+-", label="Non-perturbed")
    PyPlot.semilogy(allocp_list, errp_list, "x-", label="Perturbed")
    xlabel("bytes allocated")
    ylabel("error")
    legend()

    PyPlot.savefig("compare.pdf")

    expo = [Ecut_list time_list alloc_list err_list]
    expop = [Ecutp_list timep_list allocp_list errp_list]

    writedlm("./optim_nonperturbed.csv", expo, ',')
    writedlm("./optim_perturbed.csv", expop, ',')
end
compare()
STOP
############################# Graphical ########################################

"""
graphical optimization of the lattice constant
"""
function optimize_a_graphic(Ecut)
    arange = range(0.95*aref, 1.05*aref, length=11)

    E_list = []
    Ep_list = []
    for a in arange
        println("Solving for a=$(a)")
        push!(E_list, E(a, Ecut))
        push!(Ep_list, E_perturbed(a, Ecut, 2.5))
    end
    PyPlot.plot(arange, E_list, "+-", label="Energy with Ecut=$(Ecut)")
    PyPlot.plot(arange, Ep_list, "o-", label="Perturbed energy with Ecut=$(Ecut)")
end

#  for Ecut in 10:5:20
#      println("--------------------------------")
#      println("Ecut=$(Ecut)")
#      amin = optimize_a_graphic(Ecut)
#      println(amin)
#  end
#  legend()






